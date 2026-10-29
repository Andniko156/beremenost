"""Adapters keep real Chronos inference distinct from the persistence baseline."""

from pathlib import Path
import os

import numpy as np
import pandas as pd

from .data import ExperimentConfig

MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
QUANTILES = [0.1, 0.5, 0.9]


class PersistenceForecaster:
    name = "persistence"

    def metadata(self):
        return {"backend": self.name, "rule": "repeat the last observed value", "intervals": False}

    def predict(self, histories, horizon):
        # A point forecast has no estimated uncertainty interval.
        return [np.column_stack((np.full(horizon, np.nan), np.full(horizon, x[-1]),
                                 np.full(horizon, np.nan))) for x in histories]


class ChronosForecaster:
    name = "chronos2"

    def __init__(self, model=MODEL_ID, revision=MODEL_REVISION, device="cpu", offline=False):
        if offline:
            # Also covers the upstream adapter-config probe before weight loading.
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise RuntimeError("Install the model dependencies: python -m pip install -e '.[chronos]'") from exc
        self.model = model
        self.revision = None if Path(model).is_dir() else revision
        self.device = device
        self.offline = offline
        self.cpu_threads = torch.get_num_threads()
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu")
        kwargs = {"device_map": device, "local_files_only": offline, "dtype": torch.float32}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        self.pipeline = Chronos2Pipeline.from_pretrained(model, **kwargs)

    def metadata(self):
        from importlib.metadata import version

        return {"backend": self.name, "model": self.model, "revision": self.revision,
                "device": self.device, "offline": self.offline,
                "cpu_threads": self.cpu_threads,
                "chronos_forecasting": version("chronos-forecasting"),
                "torch": version("torch"), "transformers": version("transformers")}

    def predict(self, histories, horizon):
        # Separate univariate items; never enable cross-patient cross-learning.
        quantiles, _ = self.pipeline.predict_quantiles(
            [np.asarray(x, dtype=np.float32) for x in histories],
            prediction_length=horizon, quantile_levels=QUANTILES,
            context_length=max(len(x) for x in histories), cross_learning=False,
        )
        # Chronos-2 returns (n_variates, horizon, n_quantiles) per item.
        return [q.detach().cpu().numpy()[0] for q in quantiles]


def forecast_frame(validated: pd.DataFrame, config: ExperimentConfig, forecaster):
    groups = list(validated.groupby("series_id", sort=True))
    histories = [g["heart_rate_bpm"].to_numpy()[-config.history_steps:] for _, g in groups]
    predictions = forecaster.predict(histories, config.horizon_steps)
    if len(predictions) != len(groups):
        raise ValueError("Model returned an unexpected number of series")
    frames = []
    for (series_id, group), prediction in zip(groups, predictions):
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != (config.horizon_steps, 3):
            raise ValueError("Model returned an unexpected forecast shape")
        if not np.isfinite(prediction[:, 1]).all():
            raise ValueError("Model returned a non-finite point forecast")
        if forecaster.name != "persistence":
            if not np.isfinite(prediction).all() or (np.diff(prediction, axis=1) < 0).any():
                raise ValueError("Model returned invalid or crossing quantiles; do not publish the forecast")
        origin = group["timestamp"].iloc[-1]
        frames.append(pd.DataFrame({
            "series_id": series_id,
            "timestamp": pd.date_range(origin + config.interval, periods=config.horizon_steps,
                                       freq=config.interval),
            "p10": prediction[:, 0], "p50": prediction[:, 1], "p90": prediction[:, 2],
            "backend": forecaster.name,
        }))
    return pd.concat(frames, ignore_index=True)
