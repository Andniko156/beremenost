"""Strict input contract: no implicit resampling, imputation or medical thresholds."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUMNS = ["series_id", "timestamp", "heart_rate_bpm"]


@dataclass(frozen=True)
class ExperimentConfig:
    interval_minutes: int = 5
    history_steps: int = 288
    horizon_steps: int = 12

    def __post_init__(self):
        for name in ("interval_minutes", "history_steps", "horizon_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.history_steps < 2 or self.history_steps > 8192:
            raise ValueError("history_steps must be between 2 and 8192")
        if self.horizon_steps > 1024:
            raise ValueError("horizon_steps must not exceed 1024")

    @property
    def interval(self):
        return pd.Timedelta(minutes=self.interval_minutes)


def validate_frame(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if set(frame.columns) != set(COLUMNS) or len(frame.columns) != len(COLUMNS):
        raise ValueError(f"CSV must contain exactly {COLUMNS}; extra columns are not covariates")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Empty input or missing values; no automatic imputation is performed")
    result = frame[COLUMNS].copy()
    if not result["series_id"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("series_id must be a nonempty pseudonymous string")
    # Require an explicit UTC offset so local device time is never guessed.
    timestamps = result["timestamp"].astype(str)
    if not timestamps.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).all():
        raise ValueError("Every timestamp must include Z or an explicit offset such as +03:00")
    try:
        result["timestamp"] = pd.to_datetime(timestamps, format="ISO8601", utc=True, errors="raise")
        values = pd.to_numeric(result["heart_rate_bpm"], errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid timestamp or numeric heart_rate_bpm") from exc
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("heart_rate_bpm must be finite and positive; this is not a clinical range check")
    result["heart_rate_bpm"] = values
    if result.duplicated(["series_id", "timestamp"]).any():
        raise ValueError("Duplicate timestamps within a series; resolve duplicates at the source")
    result = result.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    for _, group in result.groupby("series_id", sort=True):
        if len(group) < config.history_steps:
            raise ValueError(f"Each series needs at least {config.history_steps} observations")
        if not group["timestamp"].diff().iloc[1:].eq(config.interval).all():
            raise ValueError("Irregular timestamps or gaps; provide a documented regular series")
    return result


def read_csv(path, config):
    return validate_frame(pd.read_csv(path, dtype={"series_id": str}), config)


def synthetic_data(days=3, series_count=2, seed=42, interval_minutes=5):
    """Engineering fixture only; not a simulator of pregnancy physiology."""
    if days < 1 or series_count < 1 or interval_minutes < 1:
        raise ValueError("days, series_count and interval_minutes must be positive")
    rng = np.random.default_rng(seed)
    count = days * 24 * 60 // interval_minutes
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=count, freq=f"{interval_minutes}min")
    hours = np.arange(count) * interval_minutes / 60
    frames = []
    for i in range(series_count):
        noise = np.zeros(count)
        for j in range(1, count):
            noise[j] = 0.85 * noise[j - 1] + rng.normal(0, 0.8)
        values = 75 + 3 * i + 6 * np.sin(2 * np.pi * hours / 24) + noise
        # A deterministic activity-like excursion exercises changing trends.
        values += 8 * np.exp(-0.5 * ((hours % 24 - 17) / 0.5) ** 2)
        frames.append(pd.DataFrame({
            "series_id": f"synthetic_{i + 1:03d}",
            "timestamp": timestamps,
            "heart_rate_bpm": values.round(3),
        }))
    return pd.concat(frames, ignore_index=True)
