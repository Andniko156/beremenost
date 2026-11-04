"""Explicit research LoRA adaptation; never replaces the serving model automatically."""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .evaluation import metrics
from .forecast import MODEL_ID, MODEL_REVISION, QUANTILES
from .research_data import check_parent_holdouts, evaluation_windows, load_prepared, sha256, training_windows, write_json


def _parameter_hash(model, trainable_only=False):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if trainable_only and not value.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate_models(windows, profile, pipelines):
    histories = [row["values"][:profile.context_steps] for row in windows]
    mean_window = min(profile.context_steps, max(1, 60 // profile.interval_minutes))
    forecasts = {
        "persistence": [np.column_stack([np.full(profile.prediction_steps, np.nan),
                                          np.repeat(x[-1], profile.prediction_steps),
                                          np.full(profile.prediction_steps, np.nan)]) for x in histories],
        "rolling_mean": [np.column_stack([np.full(profile.prediction_steps, np.nan),
                                           np.repeat(np.mean(x[-mean_window:]), profile.prediction_steps),
                                           np.full(profile.prediction_steps, np.nan)]) for x in histories],
    }
    for name, pipeline in pipelines.items():
        qs, _ = pipeline.predict_quantiles(histories, prediction_length=profile.prediction_steps,
                                           context_length=profile.context_steps, quantile_levels=QUANTILES,
                                           batch_size=8, cross_learning=False)
        forecasts[name] = [q.detach().cpu().numpy()[0] for q in qs]
    rows = []
    for name, predictions in forecasts.items():
        if len(predictions) != len(windows):
            raise ValueError("Model returned an unexpected number of predictions")
        for window, prediction in zip(windows, predictions):
            prediction = np.asarray(prediction)
            if prediction.shape != (profile.prediction_steps, 3) or not np.isfinite(prediction[:, 1]).all():
                raise ValueError("Invalid forecast shape or nonfinite point prediction")
            if name in pipelines and (not np.isfinite(prediction).all() or (np.diff(prediction, axis=1) < 0).any()):
                raise ValueError("Invalid or crossing quantiles")
            for step, (actual, quantiles) in enumerate(zip(window["values"][profile.context_steps:], prediction), 1):
                rows.append({"backend": name, "subject_id": window["subject_id"],
                             "segment_id": window["segment_id"], "origin_window": window["origin_window"],
                             "horizon_step": step, "actual": float(actual),
                             **dict(zip(("p10", "p50", "p90"), map(float, quantiles)))})
    frame = pd.DataFrame(rows)
    report = {"aggregate": {name: metrics(g) for name, g in frame.groupby("backend")},
              "per_horizon_minutes": {str(step * profile.interval_minutes):
                  {name: metrics(g) for name, g in group.groupby("backend")}
                  for step, group in frame.groupby("horizon_step")},
              "subject_macro_mae_bpm": {name: float(group.groupby("subject_id").apply(
                  lambda g: np.abs(g.p50 - g.actual).mean(), include_groups=False).mean())
                  for name, group in frame.groupby("backend")}}
    return frame, report


def finetune(dataset_dir, output_dir, *, init_model=MODEL_ID, num_steps=100, batch_size=4,
             learning_rate=1e-5, device="cpu", cpu_threads=4, seed=42, offline=False):
    import torch
    # Upstream silently falls back to full fine-tuning without PEFT. This command must fail instead.
    try:
        import peft  # noqa: F401
        from chronos import Chronos2Pipeline
        from transformers import TrainerCallback, set_seed
    except ImportError as exc:
        raise RuntimeError("Install .[finetune]; PEFT is required and full-training fallback is forbidden") from exc

    if type(num_steps) is not int or num_steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Training steps and batch size must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested compute device is unavailable")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Training output directory must be new or empty")
    manifest, records, profile = load_prepared(dataset_dir)
    # A local *merged* parent is required for subsequent stages. New LoRA then has an explicit parent.
    parent = Path(init_model)
    if init_model != MODEL_ID and not (parent.is_dir() and (parent / "config.json").exists()):
        raise ValueError("Continuation requires a local merged-model directory with config.json")
    if parent.is_dir() and (parent / "adapter_config.json").exists():
        raise ValueError("Export the parent adapter as a merged model before continuing fine-tuning")
    parent_lineage = {}
    if parent.is_dir():
        lineage_path = parent / "training_lineage.json"
        if not lineage_path.is_file():
            raise ValueError("Continuation requires training_lineage.json from the previous stage")
        parent_lineage = json.loads(lineage_path.read_text())
        check_parent_holdouts(manifest, parent_lineage)
    train = training_windows(records, set(manifest["splits"]["train"]), profile)
    val_windows = evaluation_windows(records, set(manifest["splits"]["validation"]), profile)
    test_windows = evaluation_windows(records, set(manifest["splits"]["test"]), profile)
    validation = [row["values"] for row in val_windows]
    if not train or not validation or not test_windows:
        raise ValueError("All three splits require eligible windows")
    torch.set_num_threads(cpu_threads)
    set_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dataset_manifest.json", manifest)
    model_kwargs = {"device_map": device, "dtype": torch.float32, "local_files_only": offline}
    if init_model == MODEL_ID:
        model_kwargs["revision"] = MODEL_REVISION
    else:
        init_model = str(parent.resolve())
    if offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
    started = time.perf_counter()
    base = Chronos2Pipeline.from_pretrained(init_model, **model_kwargs)
    # Make a resumed adapter point to the actual merged parent, never silently to the original base.
    base.model.config._name_or_path = init_model
    initial_base_hash = _parameter_hash(base.model)
    events = {}

    class LoraGuard(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            if not trainable or not hasattr(model, "peft_config") or any("lora_" not in n for n, _ in trainable):
                raise RuntimeError("Only LoRA parameters may be trained")
            events["trainable_parameters"] = sum(p.numel() for _, p in trainable)
            events["total_parameters"] = sum(p.numel() for p in model.parameters())
            events["initial_adapter_sha256"] = _parameter_hash(model, trainable_only=True)

        def on_train_end(self, args, state, control, **kwargs):
            events["completed_steps"] = state.global_step
            events["best_validation_checkpoint"] = state.best_model_checkpoint
            events["best_validation_loss"] = state.best_metric
            write_json(out / "trainer_history.json", state.log_history)

    eval_steps = max(1, min(25, num_steps))
    tuned = base.fit(train, prediction_length=profile.prediction_steps, validation_inputs=validation,
                     context_length=profile.context_steps, min_past=profile.context_steps,
                     finetune_mode="lora", lora_config={"r": 8, "lora_alpha": 16,
                        "revision": MODEL_REVISION if init_model == MODEL_ID else None,
                        "target_modules": ["self_attention.q", "self_attention.v", "self_attention.k",
                                           "self_attention.o", "output_patch_embedding.output_layer"]},
                     learning_rate=learning_rate, num_steps=num_steps, batch_size=batch_size,
                     output_dir=out, finetuned_ckpt_name="adapter", callbacks=[LoraGuard()],
                     optim="adamw_torch", eval_steps=eval_steps, save_steps=eval_steps,
                     logging_steps=max(1, min(10, num_steps)), disable_tqdm=True, report_to="none",
                     seed=seed, data_seed=seed)
    events["final_adapter_sha256"] = _parameter_hash(tuned.model, trainable_only=True)
    if events.get("completed_steps") != num_steps or events["initial_adapter_sha256"] == events["final_adapter_sha256"]:
        raise RuntimeError("The requested adaptation did not complete or adapter parameters did not change")
    events["base_pipeline_unchanged"] = _parameter_hash(base.model) == initial_base_hash
    if not events["base_pipeline_unchanged"]:
        raise RuntimeError("The original baseline pipeline was unexpectedly modified")
    training_seconds = time.perf_counter() - started
    # Evaluate the test women only after model selection on validation; never select using test metrics.
    report = {"dataset": manifest["source"], "prepared_manifest_sha256": sha256(Path(dataset_dir) / "manifest.json"),
              "profile": asdict(profile), "seed": seed, "init_model": init_model,
              "base_revision": MODEL_REVISION if init_model == MODEL_ID else None,
              "initial_model_parameters_sha256": initial_base_hash,
              "parent_config_sha256": sha256(parent / "config.json") if parent.is_dir() else None,
              "training": {"mode": "lora", "steps": num_steps, "batch_size": batch_size,
                           "learning_rate": learning_rate, "cpu_threads": cpu_threads,
                           "device": device, "seconds_including_model_load": training_seconds, **events},
              "counts": {"subjects": {k: len(v) for k, v in manifest["splits"].items()},
                         "training_windows": len(train), "validation_windows": len(validation),
                         "test_windows": len(test_windows)},
              "runtime": {n: version(n) for n in ("torch", "chronos-forecasting", "transformers", "peft")},
              "clinical_validity": "not_evaluated", "serving_model_changed": False,
              "automatic_promotion": False}
    for split, windows in (("validation", val_windows), ("test", test_windows)):
        predictions, scores = evaluate_models(windows, profile, {"chronos2_base": base, "chronos2_lora": tuned})
        predictions.to_csv(out / f"{split}_predictions.csv", index=False)
        report[split] = scores
    report["evaluation_seconds"] = time.perf_counter() - started - training_seconds
    write_json(out / "report.json", report)
    # A full local checkpoint provides an unambiguous parent for another medical-data adaptation stage.
    merged = tuned.model.merge_and_unload()
    merged.save_pretrained(out / "merged-model", safe_serialization=True)
    lineage = {
        "report_sha256": sha256(out / "report.json"), "parent": init_model,
        "parent_base_revision": report["base_revision"], "source": manifest["source"],
        "profile": asdict(profile), "clinical_validity": "not_evaluated",
        "reserved_holdouts": [*parent_lineage.get("reserved_holdouts", []), {
            "subject_namespace": manifest.get("subject_namespace", manifest["source"].get("dataset_id", "unknown")),
            "validation": manifest["splits"]["validation"], "test": manifest["splits"]["test"]}]}
    write_json(out / "training_lineage.json", lineage)
    write_json(out / "merged-model" / "training_lineage.json", lineage)
    print(json.dumps({"output_dir": str(out), "training": report["training"],
                      "test": report["test"]["aggregate"]}, ensure_ascii=False, indent=2))
    return report


def export_adapter(adapter_dir, output_dir, device="cpu", offline=False):
    """Reconstruct a portable stage-one adapter bundle into a merged parent checkpoint."""
    import torch
    from chronos import Chronos2Pipeline

    source, out = Path(adapter_dir), Path(output_dir)
    if not (source / "adapter_config.json").is_file():
        raise ValueError("Expected an adapter directory")
    config = json.loads((source / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") == MODEL_ID and config.get("revision") != MODEL_REVISION:
        raise ValueError("Adapter must reference the pinned base model revision")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export output directory must be new or empty")
    pipeline = Chronos2Pipeline.from_pretrained(str(source), device_map=device,
                                               dtype=torch.float32, local_files_only=offline)
    pipeline.save_pretrained(out, safe_serialization=True)
    lineage = source.parent / "training_lineage.json"
    if lineage.is_file():
        (out / "training_lineage.json").write_bytes(lineage.read_bytes())
