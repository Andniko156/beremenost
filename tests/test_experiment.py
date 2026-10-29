import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pregnancy_monitor.cli import main
from pregnancy_monitor.data import ExperimentConfig, synthetic_data, validate_frame
from pregnancy_monitor.evaluation import backtest, metrics
from pregnancy_monitor.forecast import PersistenceForecaster, forecast_frame


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(history_steps=6, horizon_steps=3)
        self.raw = synthetic_data(days=1, series_count=2)
        self.raw["timestamp"] = self.raw.timestamp.astype(str)
        self.frame = validate_frame(self.raw, self.config)

    def test_missing_and_irregular_observations_rejected(self):
        for frame in (self.raw.drop(index=2), self.raw.assign(heart_rate_bpm=np.nan)):
            with self.assertRaises(ValueError):
                validate_frame(frame, self.config)

    def test_duplicates_and_extra_columns_rejected(self):
        for frame in (pd.concat([self.raw, self.raw.iloc[:1]]), self.raw.assign(patient_name="x")):
            with self.assertRaises(ValueError):
                validate_frame(frame, self.config)

    def test_timezone_required_and_offsets_normalized(self):
        raw = self.raw.copy()
        raw["timestamp"] = pd.to_datetime(raw.timestamp).dt.tz_convert("Europe/Moscow").astype(str)
        pd.testing.assert_frame_equal(validate_frame(raw, self.config), self.frame)
        raw["timestamp"] = pd.to_datetime(raw.timestamp).dt.tz_localize(None).astype(str)
        with self.assertRaises(ValueError):
            validate_frame(raw, self.config)

    def test_nonfinite_zero_and_negative_values_rejected(self):
        for value in (float("inf"), -1, 0, "bad"):
            raw = self.raw.copy()
            raw["heart_rate_bpm"] = raw.heart_rate_bpm.astype(object)
            raw.loc[0, "heart_rate_bpm"] = value
            with self.assertRaises(ValueError):
                validate_frame(raw, self.config)

    def test_minimum_history_and_configuration(self):
        with self.assertRaises(ValueError):
            validate_frame(self.raw.iloc[:5], self.config)
        for kwargs in ({"history_steps": 1}, {"horizon_steps": 1025}, {"interval_minutes": 0}):
            with self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)

    def test_sorting_does_not_change_forecast(self):
        shuffled = validate_frame(self.raw.sample(frac=1, random_state=1), self.config)
        pd.testing.assert_frame_equal(self.frame, shuffled)

    def test_horizon_and_series_isolation(self):
        result = forecast_frame(self.frame, self.config, PersistenceForecaster())
        for series_id, group in self.frame.groupby("series_id"):
            forecast = result[result.series_id == series_id]
            self.assertEqual(len(forecast), 3)
            self.assertEqual(forecast.timestamp.iloc[0], group.timestamp.iloc[-1] + self.config.interval)
            self.assertEqual(forecast.timestamp.iloc[-1], group.timestamp.iloc[-1] + 3 * self.config.interval)
            self.assertTrue(forecast.p50.eq(group.heart_rate_bpm.iloc[-1]).all())
            self.assertTrue(forecast[["p10", "p90"]].isna().all().all())

    def test_backtest_never_exposes_future_and_uses_exact_history(self):
        frame = self.frame.copy()
        for _, group in frame.groupby("series_id"):
            frame.loc[group.index, "heart_rate_bpm"] = np.arange(1, len(group) + 1, dtype=float)
        observed = []

        class Spy(PersistenceForecaster):
            def predict(self, histories, horizon):
                observed.append([x.copy() for x in histories])
                return super().predict(histories, horizon)

        rows, report = backtest(frame, self.config, Spy(), folds=3)
        for fold, histories in enumerate(observed):
            end = 288 - (3 - fold) * 3
            for history in histories:
                np.testing.assert_array_equal(history, np.arange(end - 5, end + 1))
        # The unseen next values are 1, 2, 3 above the last context value.
        self.assertEqual(report["aggregate"]["persistence"]["mae_bpm"], 2.0)
        self.assertEqual(len(rows), 2 * 3 * 3)

    def test_backtest_requires_enough_data(self):
        with self.assertRaises(ValueError):
            backtest(self.frame.iloc[:10], self.config, PersistenceForecaster(), folds=3)

    def test_errors_and_crossing_quantiles_are_not_hidden_by_fallback(self):
        class Broken:
            name = "chronos2"

            def predict(self, histories, horizon):
                raise RuntimeError("inference failed")

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            forecast_frame(self.frame, self.config, Broken())

        class Crossing:
            name = "chronos2"

            def predict(self, histories, horizon):
                return [np.tile([90, 80, 70], (horizon, 1)) for _ in histories]

        with self.assertRaisesRegex(ValueError, "crossing quantiles"):
            forecast_frame(self.frame, self.config, Crossing())

    def test_interval_metrics(self):
        result = metrics(pd.DataFrame({"actual": [70, 80], "p10": [60, 60],
                                       "p50": [70, 70], "p90": [75, 75]}))
        self.assertEqual(result["mae_bpm"], 5)
        self.assertEqual(result["coverage_p10_p90"], 0.5)

    def test_cli_outputs_and_invalid_input_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            data = str(Path(tmp) / "synthetic.csv")
            output = str(Path(tmp) / "result")
            self.assertEqual(main(["generate", "--output", data]), 0)
            self.assertEqual(main(["backtest", "--input", data, "--output-dir", output,
                                   "--backend", "persistence", "--data-kind", "synthetic"]), 0)
            report = json.loads((Path(output) / "report.json").read_text())
            self.assertEqual(report["horizon_minutes"], 60)
            self.assertEqual(report["data_kind"], "synthetic")
            self.assertTrue((Path(output) / "predictions.csv").is_file())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["forecast", "--input", data, "--interval-minutes", "10",
                                       "--backend", "persistence"]), 2)


if __name__ == "__main__":
    unittest.main()
