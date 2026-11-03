import unittest

import numpy as np
import pandas as pd

from pregnancy_monitor.data import ExperimentConfig, SCENARIOS, synthetic_data, validate_frame
from pregnancy_monitor.evaluation import backtest, metrics
from pregnancy_monitor.forecast import (PersistenceForecaster, RollingMeanForecaster,
                                       SeasonalForecaster, benchmark_baselines, forecast_frame)


class BenchmarkTests(unittest.TestCase):
    def test_baselines_use_only_their_requested_history(self):
        result = RollingMeanForecaster(3).predict([np.array([900, 3, 6, 9])], 2)[0]
        np.testing.assert_array_equal(result[:, 1], [6, 6])
        self.assertTrue(np.isnan(result[:, [0, 2]]).all())
        result = SeasonalForecaster(3).predict([np.array([900, 3, 6, 9])], 5)[0]
        np.testing.assert_array_equal(result[:, 1], [3, 6, 9, 3, 6])
        with self.assertRaises(ValueError):
            SeasonalForecaster(4).predict([np.array([1, 2, 3])], 2)

    def test_benchmark_aligns_models_and_reports_each_horizon(self):
        config = ExperimentConfig()
        raw = synthetic_data()
        raw["timestamp"] = raw.timestamp.astype(str)
        data = validate_frame(raw, config)
        rows, report = backtest(data, config, PersistenceForecaster(), folds=12,
                                baselines=benchmark_baselines(config))
        self.assertEqual(set(report["aggregate"]), {"persistence", "rolling_mean", "seasonal"})
        self.assertEqual(set(report["per_horizon_minutes"]), {str(i) for i in range(5, 61, 5)})
        self.assertEqual(len(rows), 2 * 12 * 12 * 3)
        self.assertTrue((rows.timestamp > rows.forecast_origin).all())
        for model, group in rows.groupby("backend"):
            self.assertEqual(report["aggregate"][model]["n"], 288)
            self.assertTrue((group.timestamp - group.forecast_origin ==
                             pd.to_timedelta(group.horizon_step * 5, unit="min")).all())
        self.assertFalse(report["comparison"]["automatic_model_selection"])

    def test_zero_baseline_error_does_not_cause_division_by_zero(self):
        config = ExperimentConfig()
        raw = synthetic_data(days=2, series_count=1)
        raw["timestamp"] = raw.timestamp.astype(str)
        raw["heart_rate_bpm"] = 70
        _, report = backtest(validate_frame(raw, config), config, PersistenceForecaster(),
                             baselines=benchmark_baselines(config))
        self.assertIsNone(report["comparison"]["mae_improvement_fraction"])
        self.assertFalse(report["comparison"]["candidate_has_lower_mae"])

    def test_quantile_loss_matches_hand_calculation(self):
        report = metrics(pd.DataFrame({"actual": [10, 20], "p10": [5, 5], "p50": [10, 10], "p90": [15, 15]}))
        self.assertAlmostEqual(report["pinball_loss_bpm"]["0.1"], 1)
        self.assertAlmostEqual(report["pinball_loss_bpm"]["0.5"], 2.5)
        self.assertAlmostEqual(report["pinball_loss_bpm"]["0.9"], 2.5)

    def test_all_scenarios_are_reproducible_engineering_fixtures(self):
        for scenario in SCENARIOS:
            pd.testing.assert_frame_equal(synthetic_data(scenario=scenario), synthetic_data(scenario=scenario))
        self.assertFalse(synthetic_data(scenario="daily").equals(synthetic_data(scenario="shift")))

    def test_nonphysical_point_forecast_is_refused_without_clipping(self):
        class Negative(PersistenceForecaster):
            def predict(self, histories, horizon):
                return super().predict([np.array([-1]) for _ in histories], horizon)

        config = ExperimentConfig()
        raw = synthetic_data(days=1, series_count=1)
        raw["timestamp"] = raw.timestamp.astype(str)
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            forecast_frame(validate_frame(raw, config), config, Negative())


if __name__ == "__main__":
    unittest.main()
