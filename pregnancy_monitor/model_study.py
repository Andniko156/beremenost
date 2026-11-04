"""Nested subject evaluation on development women; the original test split stays excluded."""

from dataclasses import asdict
import gc
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .evaluation import metrics
from .finetuning import _parameter_hash, evaluate_models, fit_lora_model
from .forecast import MODEL_ID, MODEL_REVISION
from .research_data import evaluation_windows, load_prepared, sha256, write_json

BASE = "chronos2_base"
POLICY = "selected_policy"
CONFIGURATIONS = (
    {"name": "lora_r4_lr1e-5", "rank": 4, "learning_rate": 1e-5},
    {"name": "lora_r4_lr5e-5", "rank": 4, "learning_rate": 5e-5},
)
BLEND_WEIGHTS = (0.25, 0.5, 1.0)
KEYS = ["subject_id", "segment_id", "origin_window", "horizon_step"]


def development_records(manifest, records):
    subjects = set(manifest["splits"]["train"]) | set(manifest["splits"]["validation"])
    if subjects.intersection(manifest["splits"]["test"]):
        raise ValueError("Development subjects overlap the reserved test split")
    selected = [r for r in records if r["subject_id"] in subjects]
    counts = pd.Series([r["subject_id"] for r in selected]).value_counts()
    if set(counts.index) != subjects or not counts.eq(1).all():
        raise ValueError("This study requires exactly one eligible recording per development woman")
    return subjects, selected


def nested_folds(subjects, folds=3, inner_validation=4, seed=20261104):
    subjects = sorted(set(subjects))
    if type(folds) is not int or folds < 2 or type(inner_validation) is not int or inner_validation < 2:
        raise ValueError("At least two outer folds and two inner-validation women are required")
    order = np.random.default_rng(seed).permutation(subjects)
    assessments = np.array_split(order, folds)
    result = []
    for i, assessment in enumerate(assessments):
        others = sorted(set(subjects) - set(assessment))
        if len(assessment) < 2 or len(others) - inner_validation < 3:
            raise ValueError("Too few women for the requested nested partition")
        inner = np.random.default_rng(seed + i + 1).permutation(others).tolist()
        result.append({"fold": i + 1, "train": sorted(inner[inner_validation:]),
                       "validation": sorted(inner[:inner_validation]), "assessment": sorted(assessment.tolist())})
    return result


def subject_errors(rows):
    return rows.assign(error=(rows.p50 - rows.actual).abs()).groupby("subject_id").error.mean()


def blend_rows(base_rows, adapted_rows, weight, name):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Blend weight must be between zero and one")
    merged = base_rows.merge(adapted_rows, on=KEYS, suffixes=("_base", "_adapted"), validate="one_to_one")
    if len(merged) != len(base_rows) or len(merged) != len(adapted_rows):
        raise ValueError("Blend forecasts must cover exactly the same observations")
    if not np.array_equal(merged.actual_base.to_numpy(), merged.actual_adapted.to_numpy()):
        raise ValueError("Blend targets do not match")
    result = merged[KEYS].copy()
    result["actual"] = merged.actual_base
    result["backend"] = name
    for column in ("p10", "p50", "p90"):
        result[column] = (1 - weight) * merged[column + "_base"] + weight * merged[column + "_adapted"]
    return result


def select_candidate(validation_rows, minimum_relative_improvement=0.01):
    """Pure inner-validation selection. No outer scores or original test inputs are accepted."""
    if not np.isfinite(minimum_relative_improvement) or not 0 <= minimum_relative_improvement < 1:
        raise ValueError("Minimum improvement must be in [0, 1)")
    groups = {name: subject_errors(g) for name, g in validation_rows.groupby("backend")}
    if BASE not in groups or len(groups[BASE]) < 2:
        raise ValueError("Base forecasts on at least two validation women are required")
    reference = groups[BASE]
    base_mae = float(reference.mean())
    reference_rows = validation_rows[validation_rows.backend == BASE]
    scores = {}
    eligible = []
    for name, error in sorted(groups.items()):
        if set(error.index) != set(reference.index):
            raise ValueError("Every selection candidate must cover the same validation women")
        candidate_rows = validation_rows[validation_rows.backend == name]
        paired = reference_rows.merge(candidate_rows, on=KEYS, suffixes=("_reference", "_candidate"), validate="one_to_one")
        if (len(paired) != len(reference_rows) or len(paired) != len(candidate_rows)
                or not np.array_equal(paired.actual_reference.to_numpy(), paired.actual_candidate.to_numpy())):
            raise ValueError("Selection candidates must predict identical validation targets")
        difference = reference - error.reindex(reference.index)
        mae = float(error.mean())
        improvement = (base_mae - mae) / base_mae if base_mae else 0.0
        fraction_better = float((difference > 0).mean())
        scores[name] = {"subject_macro_mae_bpm": mae, "relative_improvement_vs_base": improvement,
                        "fraction_women_with_lower_mae": fraction_better, "subjects": len(error)}
        if name != BASE and improvement >= minimum_relative_improvement and fraction_better >= 0.5:
            eligible.append(name)
    selected = min(eligible, key=lambda name: (scores[name]["subject_macro_mae_bpm"], name)) if eligible else BASE
    return selected, scores


def paired_subject_bootstrap(rows, repetitions=2000, seed=20261104):
    if type(repetitions) is not int or repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    base = subject_errors(rows[rows.backend == BASE])
    candidate = subject_errors(rows[rows.backend == POLICY])
    if len(base) < 3 or set(base.index) != set(candidate.index):
        raise ValueError("Paired scores for at least three women are required")
    differences = (candidate.reindex(base.index) - base).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(repetitions, len(differences)), replace=True).mean(axis=1)
    return {"definition": "selected_policy MAE minus base MAE; negative favours selected policy",
            "resampling_unit": "woman, not forecast point", "subjects": len(differences),
            "mean_difference_bpm": float(differences.mean()),
            "conditional_percentile_95_interval_bpm": np.quantile(draws, [.025, .975]).tolist(),
            "repetitions": repetitions,
            "limitation": "Conditional on fitted folds; models are not retrained within bootstrap"}


class BlendedPipeline:
    def __init__(self, base, adapted, weight):
        self.base, self.adapted, self.weight = base, adapted, weight

    def predict_quantiles(self, histories, **kwargs):
        base, _ = self.base.predict_quantiles(histories, **kwargs)
        adapted, _ = self.adapted.predict_quantiles(histories, **kwargs)
        if len(base) != len(adapted) or any(a.shape != b.shape for a, b in zip(base, adapted)):
            raise ValueError("Blend model output shapes differ")
        return [(1 - self.weight) * a + self.weight * b for a, b in zip(base, adapted)], None


def run_study(dataset_dir, output_dir, *, num_steps=100, seed=20261104, cpu_threads=4,
              minimum_relative_improvement=0.01, offline=False):
    import torch
    from chronos import Chronos2Pipeline
    from transformers import set_seed

    if type(num_steps) is not int or num_steps < 1 or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("Steps and CPU threads must be positive integers")
    if not 0 <= minimum_relative_improvement < 1:
        raise ValueError("Invalid minimum improvement")
    manifest, records, profile = load_prepared(dataset_dir)
    subjects, records = development_records(manifest, records)
    folds = nested_folds(subjects, seed=seed)
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Study output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    plan = {"protocol": "three outer subject folds with a separate inner holdout",
            "seed": seed, "source_manifest_sha256": sha256(Path(dataset_dir) / "manifest.json"),
            "profile": asdict(profile), "configurations": list(CONFIGURATIONS),
            "blend_weights": list(BLEND_WEIGHTS), "steps_per_fit": num_steps,
            "minimum_relative_improvement": minimum_relative_improvement,
            "selection_rule": "inner MAE, >= minimum gain and >= half women improve; otherwise base",
            "checkpoint_metric": "inner forecast MAE", "folds": folds,
            "reserved_original_test_subjects": sorted(manifest["splits"]["test"]),
            "training_sampler": "uniform full visits, one eligible visit per development woman",
            "initialization": "original pinned Chronos-2 for every fit, never the stage-one adapter"}
    write_json(out / "study_plan.json", plan)  # Frozen before model loading and all fitting.
    torch.set_num_threads(cpu_threads)
    set_seed(seed)
    started = time.perf_counter()
    base = Chronos2Pipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                           device_map="cpu", dtype=torch.float32, local_files_only=offline)
    initial_hash = _parameter_hash(base.model)
    assessments, fold_reports = [], []
    for fold in folds:
        index = fold["fold"]
        print(f"Outer fold {index}/3: {len(fold['train'])} train, {len(fold['validation'])} inner validation, {len(fold['assessment'])} assessment", flush=True)
        train = [np.asarray(r["values"], dtype=np.float32) for r in records if r["subject_id"] in fold["train"]]
        val_windows = evaluation_windows(records, set(fold["validation"]), profile)
        validation = [r["values"] for r in val_windows]
        inner, _ = evaluate_models(val_windows, profile, {BASE: base})
        base_rows = inner[inner.backend == BASE]
        candidates = [base_rows]
        pipelines, fit_audits = {}, {}
        for config in CONFIGURATIONS:
            name = config["name"]
            set_seed(seed + index)
            pipeline, audit = fit_lora_model(
                base, train, validation, profile, out / f"fold-{index}" / name,
                num_steps=num_steps, rank=config["rank"], learning_rate=config["learning_rate"],
                seed=seed + index, selection_metric="eval_forecast_mae",
                validation_subjects=[row["subject_id"] for row in val_windows])
            pipelines[name] = pipeline
            audit["best_validation_checkpoint"] = Path(audit["best_validation_checkpoint"]).name
            fit_audits[name] = audit
            predicted, _ = evaluate_models(val_windows, profile, {name: pipeline})
            adapted_rows = predicted[predicted.backend == name]
            for weight in BLEND_WEIGHTS:
                candidates.append(blend_rows(base_rows, adapted_rows, weight, f"{name}@{weight:g}"))
        selected, selection_scores = select_candidate(pd.concat(candidates, ignore_index=True), minimum_relative_improvement)
        selected_pipeline = base
        if selected != BASE:
            model_name, weight = selected.split("@")
            selected_pipeline = BlendedPipeline(base, pipelines[model_name], float(weight))
        # Outer observations are first forecast only after the inner decision has been finalized.
        outer_windows = evaluation_windows(records, set(fold["assessment"]), profile)
        rows, outer_scores = evaluate_models(outer_windows, profile, {BASE: base, POLICY: selected_pipeline})
        rows["outer_fold"] = index
        assessments.append(rows)
        fold_reports.append({"fold": index, "train_subjects": len(fold["train"]),
                             "inner_validation_subjects": len(fold["validation"]),
                             "assessment_subjects": len(fold["assessment"]),
                             "selected": selected, "inner_selection_scores": selection_scores,
                             "outer_scores": outer_scores, "fits": fit_audits})
        write_json(out / "progress.json", fold_reports)
        print(f"Fold {index} selected {selected}; assessment MAE base={outer_scores['aggregate'][BASE]['mae_bpm']:.4f}, policy={outer_scores['aggregate'][POLICY]['mae_bpm']:.4f}", flush=True)
        del pipelines, pipeline, selected_pipeline
        gc.collect()
    rows = pd.concat(assessments, ignore_index=True)
    if set(rows.subject_id) != subjects or set(rows.subject_id).intersection(manifest["splits"]["test"]):
        raise RuntimeError("Outer coverage mismatch or reserved-test leakage")
    if rows.duplicated(["backend", *KEYS]).any():
        raise RuntimeError("Repeated outer target in assessment")
    unchanged = _parameter_hash(base.model) == initial_hash
    if not unchanged:
        raise RuntimeError("The original model was modified during study")
    rows.to_csv(out / "development_oof_predictions.csv", index=False)
    report = {"protocol": plan["protocol"], "study_plan_sha256": sha256(out / "study_plan.json"),
              "dataset": manifest["source"], "profile": asdict(profile), "seed": seed,
              "model": MODEL_ID, "revision": MODEL_REVISION, "development_subjects": len(subjects),
              "original_test_subjects_excluded": len(manifest["splits"]["test"]),
              "original_test_evaluated": False, "base_parameters_unchanged": unchanged,
              "initial_model_parameters_sha256": initial_hash, "folds": fold_reports,
              "aggregate": {name: metrics(g) for name, g in rows.groupby("backend")},
              "per_horizon_minutes": {str(step * profile.interval_minutes):
                    {name: metrics(g) for name, g in frame.groupby("backend")}
                    for step, frame in rows.groupby("horizon_step")},
              "paired_subject_bootstrap": paired_subject_bootstrap(rows, seed=seed),
              "elapsed_seconds": time.perf_counter() - started,
              "runtime": {n: version(n) for n in ("torch", "chronos-forecasting", "transformers", "peft")},
              "serving_model_changed": False, "clinical_validity": "not_evaluated",
              "interpretation": "Assessment of a selection procedure, not a single deployable checkpoint or external clinical validation"}
    write_json(out / "report.json", report)
    print(json.dumps({"output_dir": str(out), "aggregate": report["aggregate"],
                      "bootstrap": report["paired_subject_bootstrap"]}, indent=2), flush=True)
    return report
