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
    return result


def backtest(validated, config, forecaster, folds=3):
    if type(folds) is not int or folds < 1:
        raise ValueError("folds must be a positive integer")
    required = config.history_steps + folds * config.horizon_steps
    groups = list(validated.groupby("series_id", sort=True))
    if any(len(group) < required for _, group in groups):
        raise ValueError(f"Backtest needs at least {required} observations per series")
    records = []
    models = [forecaster] if forecaster.name == "persistence" else [forecaster, PersistenceForecaster()]
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
    return rows, report
