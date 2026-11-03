"""Chronological evaluation: observations after the origin are never model inputs."""

import numpy as np
import pandas as pd

from .forecast import PersistenceForecaster, forecast_frame


def metrics(rows):
    error = rows["p50"].to_numpy() - rows["actual"].to_numpy()
    result = {"n": len(rows), "mae_bpm": float(np.abs(error).mean()),
              "rmse_bpm": float(np.sqrt(np.square(error).mean()))}
    if rows[["p10", "p90"]].notna().all().all():
        result["coverage_p10_p90"] = float(
            ((rows.actual >= rows.p10) & (rows.actual <= rows.p90)).mean())
        result["mean_interval_width_bpm"] = float((rows.p90 - rows.p10).mean())
        result["pinball_loss_bpm"] = {}
        for q, column in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
            residual = rows["actual"].to_numpy() - rows[column].to_numpy()
            result["pinball_loss_bpm"][str(q)] = float(np.maximum(q * residual, (q - 1) * residual).mean())
    return result


def backtest(validated, config, forecaster, folds=3, baselines=None):
    if type(folds) is not int or folds < 1:
        raise ValueError("folds must be a positive integer")
    required = config.history_steps + folds * config.horizon_steps
    groups = list(validated.groupby("series_id", sort=True))
    if any(len(group) < required for _, group in groups):
        raise ValueError(f"Backtest needs at least {required} observations per series")
    records = []
    if baselines is None:
        baselines = [PersistenceForecaster()]
    models = list({model.name: model for model in [*baselines, forecaster]}.values())
    for fold in range(folds):
        contexts, targets = [], []
        for _, group in groups:
            end = len(group) - (folds - fold) * config.horizon_steps
            contexts.append(group.iloc[end - config.history_steps:end])
            targets.append(group.iloc[end:end + config.horizon_steps])
        context = pd.concat(contexts, ignore_index=True)
        actual = pd.concat(targets, ignore_index=True).rename(columns={"heart_rate_bpm": "actual"})
        for model in models:
            forecast = forecast_frame(context, config, model)
            joined = forecast.merge(actual, on=["series_id", "timestamp"], how="inner", validate="one_to_one")
            if len(joined) != len(forecast):
                raise ValueError("Forecast and held-out timestamps do not match")
            joined["fold"] = fold + 1
            records.append(joined)
    rows = pd.concat(records, ignore_index=True)
    report = {"folds": folds, "method": "rolling origins, non-overlapping forecast windows",
              "aggregate": {name: metrics(group) for name, group in rows.groupby("backend")},
              "per_series": {str(series): {name: metrics(group) for name, group in series_rows.groupby("backend")}
                             for series, series_rows in rows.groupby("series_id")}}
    report["models"] = [model.metadata() for model in models]
    report["per_horizon_minutes"] = {
        str(step * config.interval_minutes): {name: metrics(group) for name, group in step_rows.groupby("backend")}
        for step, step_rows in rows.groupby("horizon_step")
    }
    report["per_fold"] = {
        str(fold): {name: metrics(group) for name, group in fold_rows.groupby("backend")}
        for fold, fold_rows in rows.groupby("fold")
    }
    baseline_names = [model.name for model in models if model.name != forecaster.name]
    if baseline_names:
        scores = report["aggregate"]
        best = min(baseline_names, key=lambda name: scores[name]["mae_bpm"])
        reference = scores[best]["mae_bpm"]
        candidate = scores[forecaster.name]["mae_bpm"]
        report["comparison"] = {
            "candidate": forecaster.name, "best_baseline_on_this_sample": best,
            "mae_improvement_fraction": (reference - candidate) / reference if reference > 0 else None,
            "candidate_has_lower_mae": candidate < reference,
            "automatic_model_selection": False,
        }
    return rows, report
