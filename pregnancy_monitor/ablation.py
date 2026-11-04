"""Development-only context, past-HRV and regularized-model experiments."""

from dataclasses import asdict
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .evaluation import metrics
from .finetuning import _parameter_hash, fit_lora_model, validation_forecasts
from .forecast import MODEL_ID, MODEL_REVISION
from .model_study import BASE, KEYS, POLICY, development_records, nested_folds, select_candidate, subject_errors
from .research_data import HRV29_SHA256, ResearchProfile, load_prepared, sha256, subject_split, write_json

COVARIATES = ("rmssd_ibi", "sdnn_ibi")
SEEDS = (20261104, 20261105, 20261106)
LORA_CONFIGS = (
    {"name": "lora_hr_r2", "hrv": False, "rank": 2, "learning_rate": 1e-5},
    {"name": "lora_hrv_r2", "hrv": True, "rank": 2, "learning_rate": 1e-5},
    {"name": "lora_hr_r4", "hrv": False, "rank": 4, "learning_rate": 1e-5},
    {"name": "lora_hr_r2_lr1e-4", "hrv": False, "rank": 2, "learning_rate": 1e-4},
)
RIDGE_ALPHAS = (1.0, 10.0, 100.0)


def require_original_holdout(manifest, records):
    subjects = {r["subject_id"] for r in records}
    if len(subjects) != 29 or manifest["splits"] != subject_split(subjects, seed=42):
        raise ValueError("Ablation requires the original 29-woman seed-42 split; do not redefine the old test women")


def attach_past_hrv(workbook, records, allowed_subjects):
    """Join real source-window covariates without changing HR eligibility or imputing."""
    if sha256(workbook) != HRV29_SHA256:
        raise ValueError("Workbook differs from the reviewed HRV29 source")
    frame = pd.read_excel(workbook, sheet_name="merged_visit1", engine="openpyxl")
    frame["participant_id"] = frame.participant_id.astype(str)
    frame = frame[frame.participant_id.isin(allowed_subjects)]
    if frame.duplicated(["participant_id", "Data_collection_window"]).any():
        raise ValueError("Duplicate HRV source windows")
    source = frame.set_index(["participant_id", "Data_collection_window"])
    result = []
    for record in records:
        if record["subject_id"] not in allowed_subjects:
            raise ValueError("Reserved subject reached covariate preparation")
        if record["recording_id"] != "merged_visit1":
            raise ValueError("Only reviewed control recordings support this experiment")
        covariates = {key: [] for key in COVARIATES}
        for i, actual in enumerate(record["values"]):
            row = source.loc[(record["subject_id"], record["first_window"] + i)]
            if not np.isclose(float(row.hr_ibi), actual, rtol=0, atol=1e-5):
                raise ValueError("HRV covariates are not aligned with the prepared HR target")
            for key in COVARIATES:
                value = pd.to_numeric(row[key], errors="coerce")
                covariates[key].append(float(value) if np.isfinite(value) and value >= 0 else None)
        result.append(dict(record, past_covariates=covariates))
    return result


def last_windows(records, context, horizon):
    profile = ResearchProfile(context_steps=context, prediction_steps=horizon)
    size = context + horizon
    return [dict(r, values=np.asarray(r["values"][-size:], dtype=np.float32),
                 past_covariates={k: np.asarray(v[-size:], dtype=np.float32)
                                  for k, v in r["past_covariates"].items()},
                 origin_window=r["first_window"] + len(r["values"]) - horizon - 1)
            for r in records if len(r["values"]) >= size], profile


def model_inputs(records, use_hrv):
    if not use_hrv:
        return [np.asarray(r["values"], dtype=np.float32) for r in records]
    return [{"target": np.asarray(r["values"], dtype=np.float32),
             "past_covariates": {k: np.asarray(r["past_covariates"][k], dtype=np.float32)
                                 for k in COVARIATES}} for r in records]


def forecast_rows(windows, predictions, name, horizon):
    if len(windows) != len(predictions):
        raise ValueError("Predictions do not cover all evaluation windows")
    rows = []
    for record, prediction in zip(windows, predictions):
        q = np.asarray(prediction)
        if q.ndim == 1:
            q = np.column_stack([np.full(horizon, np.nan), q, np.full(horizon, np.nan)])
        if q.shape != (horizon, 3) or not np.isfinite(q[:, 1]).all() or (q[:, 1] <= 0).any():
            raise ValueError("Invalid point forecast")
        point_only = np.isnan(q[:, [0, 2]]).all()
        if not point_only and (not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any()):
            raise ValueError("Invalid forecast intervals")
        for step, (actual, interval) in enumerate(zip(record["values"][-horizon:], q), 1):
            rows.append({"backend": name, "subject_id": record["subject_id"],
                         "segment_id": record["segment_id"], "origin_window": record["origin_window"],
                         "horizon_step": step, "actual": float(actual),
                         **dict(zip(("p10", "p50", "p90"), map(float, interval)))})
    return pd.DataFrame(rows)


def chronos_rows(pipeline, windows, profile, use_hrv, name):
    predictions, _ = validation_forecasts(pipeline, model_inputs(windows, use_hrv), profile)
    return forecast_rows(windows, predictions, name, profile.prediction_steps)


def lag_features(values, covariates, context, use_hrv):
    """Features must receive history only; predict changes from the last observed HR."""
    history = np.asarray(values, dtype=np.float64)[-context:]
    if len(history) != context or not np.isfinite(history).all():
        raise ValueError("Complete HR context is required")
    last = history[-1]
    pieces = [history - last, np.asarray([last])]
    if use_hrv:
        for key in COVARIATES:
            x = np.asarray(covariates[key], dtype=np.float64)[-context:]
            if len(x) != context:
                raise ValueError("HRV and HR history lengths differ")
            pieces.append(np.log1p(np.where(np.isfinite(x) & (x >= 0), x, np.nan)))
    return np.concatenate(pieces)


def ridge_samples(records, profile, use_hrv, all_windows):
    X, y, last_values, subjects = [], [], [], []
    context, horizon = profile.context_steps, profile.prediction_steps
    for r in records:
        values = np.asarray(r["values"], dtype=np.float64)
        origins = range(context, len(values) - horizon + 1) if all_windows else [len(values) - horizon]
        for origin in origins:
            if origin < context:
                raise ValueError("Record is too short for the requested Ridge task")
            cov = {k: np.asarray(v[:origin], dtype=np.float64) for k, v in r["past_covariates"].items()}
            X.append(lag_features(values[:origin], cov, context, use_hrv))
            last_values.append(values[origin - 1])
            y.append(values[origin:origin + horizon] - values[origin - 1])
            subjects.append(r["subject_id"])
    return np.asarray(X), np.asarray(y), np.asarray(last_values), subjects


def fit_ridge(train, validation, profile, use_hrv):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, _, subjects = ridge_samples(train, profile, use_hrv, True)
    counts = pd.Series(subjects).value_counts()
    weights = np.asarray([1 / counts[s] for s in subjects])
    val_X, val_y, last, val_subjects = ridge_samples(validation, profile, use_hrv, False)
    candidates, scores = {}, {}
    for alpha in RIDGE_ALPHAS:
        pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(X, y, standardscaler__sample_weight=weights, ridge__sample_weight=weights)
        predictions = pipe.predict(val_X) + last[:, None]
        errors = np.abs(predictions - (val_y + last[:, None])).mean(axis=1)
        scores[str(alpha)] = float(pd.Series(errors).groupby(val_subjects).mean().mean())
        candidates[alpha] = pipe
    best = min(RIDGE_ALPHAS, key=lambda a: (scores[str(a)], -a))
    return candidates[best], {"alpha": best, "inner_mae_by_alpha": scores,
                              "training_windows": len(X), "training_subjects": len(counts),
                              "preprocessing_fit_on": "training women only", "equal_subject_weights": True}


def ridge_rows(pipeline, windows, profile, use_hrv, name):
    X, _, last, _ = ridge_samples(windows, profile, use_hrv, False)
    return forecast_rows(windows, pipeline.predict(X) + last[:, None], name, profile.prediction_steps)


def baselines(windows, profile):
    c, h = profile.context_steps, profile.prediction_steps
    return pd.concat([
        forecast_rows(windows, [np.repeat(r["values"][c - 1], h) for r in windows], "persistence", h),
        forecast_rows(windows, [np.repeat(np.mean(r["values"][:c]), h) for r in windows], "rolling_mean", h),
    ], ignore_index=True)


def summarize(rows):
    result = {}
    for name, frame in rows.groupby("backend"):
        result[name] = {**metrics(frame), "subject_macro_mae_bpm": float(subject_errors(frame).mean()),
                        "unique_subjects": frame.subject_id.nunique()}
        if "coverage_p10_p90" in result[name]:
            result[name]["mean_interval_score_80_bpm"] = float((frame.p90 - frame.p10
                + 10 * np.maximum(frame.p10 - frame.actual, 0)
                + 10 * np.maximum(frame.actual - frame.p90, 0)).mean())
    return result


def paired_comparison(rows, candidate, reference=BASE, seed=20261104):
    a, b = rows[rows.backend == reference], rows[rows.backend == candidate]
    keys = ["seed", *KEYS] if "seed" in rows else KEYS
    paired = a.merge(b, on=keys, suffixes=("_base", "_candidate"), validate="one_to_one")
    if len(paired) != len(a) or len(paired) != len(b) or not np.array_equal(paired.actual_base, paired.actual_candidate):
        raise ValueError("Paired models must cover identical targets")
    paired["difference"] = (paired.p50_candidate - paired.actual_base).abs() - (paired.p50_base - paired.actual_base).abs()
    differences = paired.groupby("subject_id").difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, (2000, len(differences)), replace=True).mean(axis=1)
    return {"candidate": candidate, "reference": reference, "subjects": len(differences),
            "mean_difference_mae_bpm": float(differences.mean()),
            "conditional_percentile_95_interval_bpm": np.quantile(draws, [.025, .975]).tolist(),
            "fraction_women_with_lower_mae": float((differences < 0).mean()),
            "limitation": "Seeds are averaged within woman; conditional on fitted folds, without retraining bootstrap"}


def run_ablation(dataset_dir, workbook, output_dir, *, seeds=SEEDS, num_steps=100, cpu_threads=4,
                 offline=False):
    if offline:
        # PEFT checkpoint saves also probe the Hub; local_files_only on the loader is insufficient.
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from chronos import Chronos2Pipeline
    from transformers import set_seed

    if type(num_steps) is not int or num_steps < 1 or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("Steps and CPU threads must be positive integers")
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    manifest, prepared, _ = load_prepared(dataset_dir)
    require_original_holdout(manifest, prepared)
    subjects, records = development_records(manifest, prepared)
    records = attach_past_hrv(workbook, records, subjects)
    if any(len(r["values"]) < 18 for r in records):
        raise ValueError("Every development woman needs 12 context + 6 target points")
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Ablation output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    profile = ResearchProfile(context_steps=12, prediction_steps=6)
    partitions = {str(seed): nested_folds(subjects, seed=seed) for seed in seeds}
    plan = {"protocol": "repeated outer subject folds with separate inner holdouts; development only",
            "source_manifest_sha256": sha256(Path(dataset_dir) / "manifest.json"),
            "source_workbook_sha256": sha256(workbook), "profile": asdict(profile), "seeds": list(seeds),
            "folds": partitions, "excluded_original_test": sorted(manifest["splits"]["test"]),
            "lora_configurations": list(LORA_CONFIGS), "steps": num_steps,
            "evaluation_interval_steps": min(10, num_steps), "ridge_alphas": list(RIDGE_ALPHAS),
            "covariates": list(COVARIATES), "primary_reference": BASE,
            "fixed_candidates": [BASE, "chronos2_context6", "chronos2_hrv", "persistence", "rolling_mean"],
            "ridge_candidates": ["ridge_hr", "ridge_hrv"], "women_per_training_batch": 4,
            "initialization": "original pinned Chronos-2 independently for every adapter",
            "primary_targets": "last six observations per eligible control recording; identical across models",
            "selection": "inner subject MAE; >=1% gain and >=half women improve; otherwise base",
            "secondary": ["context16 at 30min on common eligible subset", "60min fixed-origin horizon sweep"],
            "scope": "previously examined development cohort; no confirmatory test or automatic promotion",
            "method_sources": ["https://scikit-learn.org/stable/common_pitfalls.html",
                               "https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/notebooks/chronos-2-quickstart.ipynb"],
            "reviewed_ml_guidance_blob": "9e26cff8bd282c523b645719c817c1415e0ddb3f"}
    write_json(out / "study_plan.json", plan)
    torch.set_num_threads(cpu_threads)
    set_seed(seeds[0])
    started = time.perf_counter()
    base = Chronos2Pipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION, device_map="cpu",
                                           dtype=torch.float32, local_files_only=offline)
    base.model.config._name_or_path = MODEL_ID
    initial_hash = _parameter_hash(base.model)
    all_windows, _ = last_windows(records, 12, 6)
    short_windows, short_profile = last_windows(records, 6, 6)
    fixed_rows = pd.concat([baselines(all_windows, profile),
        chronos_rows(base, all_windows, profile, False, BASE),
        chronos_rows(base, all_windows, profile, True, "chronos2_hrv"),
        chronos_rows(base, short_windows, short_profile, False, "chronos2_context6")], ignore_index=True)
    # Secondary comparisons are fixed beforehand and never used for selection.
    subset = [r for r in records if len(r["values"]) >= 22]
    context_rows = []
    for context in (6, 12, 16):
        windows, p = last_windows(subset, context, 6)
        context_rows.append(chronos_rows(base, windows, p, False, f"context{context}"))
    hour_windows, hour_profile = last_windows(records, 6, 12)
    hour_rows = pd.concat([baselines(hour_windows, hour_profile),
        chronos_rows(base, hour_windows, hour_profile, False, BASE),
        chronos_rows(base, hour_windows, hour_profile, True, "chronos2_hrv")], ignore_index=True)
    fold_reports, outer_frames = [], []
    restored_adapter_check = None
    for seed in seeds:
        for fold in partitions[str(seed)]:
            idx = fold["fold"]
            path = out / f"seed-{seed}" / f"fold-{idx}"
            path.mkdir(parents=True)
            train = [r for r in records if r["subject_id"] in fold["train"]]
            val_records = [r for r in records if r["subject_id"] in fold["validation"]]
            val, _ = last_windows(val_records, 12, 6)
            outer = [w for w in all_windows if w["subject_id"] in fold["assessment"]]
            inner = [fixed_rows[fixed_rows.subject_id.isin(fold["validation"])].copy()]
            outer_candidates = [fixed_rows[fixed_rows.subject_id.isin(fold["assessment"])].copy()]
            audits, ridge_models = {}, {}
            for hrv in (False, True):
                name = "ridge_hrv" if hrv else "ridge_hr"
                pipeline, audit = fit_ridge(train, val, profile, hrv)
                import joblib
                joblib.dump(pipeline, path / f"{name}.joblib")
                audits[name] = audit
                ridge_models[name] = (pipeline, hrv)
                inner.append(ridge_rows(pipeline, val, profile, hrv, name))
            for config in LORA_CONFIGS:
                name = config["name"]
                print(f"seed={seed} fold={idx}/3 fitting {name}", flush=True)
                set_seed(seed + idx)
                tuned, audit = fit_lora_model(base, model_inputs(train, config["hrv"]),
                    model_inputs(val, config["hrv"]), profile, path / name,
                    rank=config["rank"], learning_rate=config["learning_rate"], num_steps=num_steps,
                    batch_size=4 * (1 + len(COVARIATES) if config["hrv"] else 1),
                    seed=seed + idx, selection_metric="eval_forecast_mae",
                    validation_subjects=[w["subject_id"] for w in val], eval_steps=min(10, num_steps))
                audit["women_per_training_batch"] = 4
                audit["series_per_training_batch"] = 4 * (1 + len(COVARIATES) if config["hrv"] else 1)
                audit["best_validation_checkpoint"] = Path(audit["best_validation_checkpoint"]).name
                audits[name] = audit
                inner.append(chronos_rows(tuned, val, profile, config["hrv"], name))
                if restored_adapter_check is None and config["hrv"]:
                    restored = Chronos2Pipeline.from_pretrained(str(path / name / "adapter"),
                        device_map="cpu", dtype=torch.float32, local_files_only=offline)
                    original_q, _ = validation_forecasts(tuned, model_inputs(val, True), profile)
                    restored_q, _ = validation_forecasts(restored, model_inputs(val, True), profile)
                    error = float(np.max(np.abs(np.asarray(original_q) - np.asarray(restored_q))))
                    if error > 1e-3:
                        raise RuntimeError("Saved HRV adapter forecasts differ after reload")
                    restored_adapter_check = {"maximum_quantile_difference_bpm": error,
                                              "past_covariate_adapter_reloaded": True}
                    del restored
                del tuned
                gc.collect()
            inner = pd.concat(inner, ignore_index=True)
            selected, scores = select_candidate(inner)
            write_json(path / "selection.json", {"selected": selected, "scores": scores})
            # No fitted candidate forecasts assessment women before the inner decision is saved.
            for name, (pipeline, hrv) in ridge_models.items():
                outer_candidates.append(ridge_rows(pipeline, outer, profile, hrv, name))
            for config in LORA_CONFIGS:
                restored = Chronos2Pipeline.from_pretrained(str(path / config["name"] / "adapter"),
                    device_map="cpu", dtype=torch.float32, local_files_only=offline)
                outer_candidates.append(chronos_rows(restored, outer, profile, config["hrv"], config["name"]))
                del restored
                gc.collect()
            rows = pd.concat(outer_candidates, ignore_index=True)
            selected_rows = rows[rows.backend == selected].assign(backend=POLICY)
            rows = pd.concat([rows, selected_rows], ignore_index=True).assign(seed=seed, outer_fold=idx)
            outer_frames.append(rows)
            fold_reports.append({"seed": seed, "fold": idx, "train_subjects": len(train),
                "validation_subjects": len(val), "assessment_subjects": len(outer),
                "selected": selected, "inner_selection_scores": scores, "fits": audits,
                "outer": summarize(rows)})
            write_json(out / "progress.json", fold_reports)
            print(f"Completed seed={seed} fold={idx}; selected={selected}", flush=True)
    rows = pd.concat(outer_frames, ignore_index=True)
    if set(rows.subject_id) != subjects or set(rows.subject_id).intersection(manifest["splits"]["test"]):
        raise RuntimeError("Reserved subject leakage or incomplete cohort")
    if rows.duplicated(["seed", "backend", *KEYS]).any():
        raise RuntimeError("Repeated assessment target within seed")
    if _parameter_hash(base.model) != initial_hash:
        raise RuntimeError("The original base model changed")
    rows.to_csv(out / "development_predictions.csv", index=False)
    pd.concat(context_rows).to_csv(out / "context_subset_predictions.csv", index=False)
    hour_rows.to_csv(out / "hour_predictions.csv", index=False)
    # The preparation manifest describes the earlier HR-only experiment. Its
    # input-use note must not incorrectly describe this past-HRV experiment.
    source = dict(manifest["source"])
    source["limitations"] = [note for note in source.get("limitations", [])
        if note != "Cortisol, HRV and stress labels are not model inputs"]
    source["limitations"].append("Past RMSSD/SDNN are inputs only for declared HRV variants; future HRV, cortisol and stress labels are not inputs")
    report = {"protocol": plan["protocol"], "study_plan_sha256": sha256(out / "study_plan.json"),
        "dataset": source, "profile": asdict(profile), "seeds": list(seeds),
        "unique_subjects": len(subjects), "unique_targets_per_model_per_seed": len(subjects) * 6,
        "missing_covariate_points": {key: sum(v is None for r in records for v in r["past_covariates"][key])
                                     for key in COVARIATES},
        "original_test_subjects_excluded": len(manifest["splits"]["test"]), "original_test_evaluated": False,
        "base_model_revision": MODEL_REVISION, "base_parameters_unchanged": True,
        "completed_lora_fits": len(fold_reports) * len(LORA_CONFIGS), "folds": fold_reports,
        "aggregate": summarize(rows), "per_seed": {str(seed): summarize(g) for seed, g in rows.groupby("seed")},
        "per_horizon_minutes": {str(step * 5): summarize(g) for step, g in rows.groupby("horizon_step")},
        "paired_comparisons": {name: paired_comparison(rows, name) for name in sorted(set(rows.backend) - {BASE})},
        "context16_common_subset": {"subjects": len(subset), "aggregate": summarize(pd.concat(context_rows))},
        "hour_fixed_origin": {"profile": asdict(hour_profile), "aggregate": summarize(hour_rows),
            "cumulative_horizons_minutes": {str(h * 5): summarize(hour_rows[hour_rows.horizon_step <= h])
                                            for h in (1, 3, 6, 12)}},
        "adapter_reload_check": restored_adapter_check,
        "runtime": {n: version(n) for n in ("torch", "numpy", "pandas", "chronos-forecasting", "transformers", "peft", "scikit-learn")},
        "elapsed_seconds": time.perf_counter() - started, "serving_model_changed": False,
        "offline": offline,
        "clinical_validity": "not_evaluated", "confirmatory_evaluation": False,
        "interpretation": "Exploratory comparisons on previously examined development women; seeds are not new patients"}
    write_json(out / "report.json", report)
    print(json.dumps({"output_dir": str(out), "aggregate": report["aggregate"],
                      "completed_lora_fits": report["completed_lora_fits"]}, indent=2), flush=True)
    return report
