"""Technical eligibility for a live forecast; these are not medical thresholds."""

from dataclasses import dataclass

import pandas as pd

from .data import validate_frame


class DataQualityError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class QualityPolicy:
    max_age_minutes: int = 15
    max_future_skew_seconds: int = 60

    def __post_init__(self):
        for name in ("max_age_minutes", "max_future_skew_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def live_history(frame, qualities, config, policy, now):
    if len(qualities) != len(frame) or any(value != "valid" for value in qualities):
        raise DataQualityError("signal_quality_rejected", "Every submitted sample must have valid signal quality")
    try:
        validated = validate_frame(frame, config)
    except ValueError as exc:
        raise DataQualityError("invalid_history", str(exc)) from exc
    if validated.series_id.nunique() != 1:
        raise DataQualityError("multiple_series", "A live request must contain exactly one series")
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("The server clock must include a timezone")
    origin = validated.timestamp.iloc[-1]
    if origin > now + pd.Timedelta(seconds=policy.max_future_skew_seconds):
        raise DataQualityError("future_observations", "The latest sample is ahead of the server clock")
    if origin < now - pd.Timedelta(minutes=policy.max_age_minutes):
        raise DataQualityError("stale_observations", "The latest sample is too old for a live forecast")
    return validated
