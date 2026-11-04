import contextlib
import importlib.util
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pregnancy_monitor.ablation import (
    BASE, COVARIATES, POLICY, attach_past_hrv, fit_ridge, forecast_rows, last_windows,
    model_inputs, paired_comparison, ridge_rows, ridge_samples, run_ablation, summarize,
)
from pregnancy_monitor.finetuning import validation_forecasts
from pregnancy_monitor.research_data import HRV29_SHA256, ResearchProfile, save_prepared


def record(subject, length=24, start=80):
    return {"subject_id": subject, "recording_id": "merged_visit1", "segment_id": subject + "/merged_visit1/1",
            "first_window": 0, "values": [start + .001 * j + .0001 * np.sin(j) for j in range(length)],
            "past_covariates": {key: [20.0 + j for j in range(length)] for key in COVARIATES}}


class AblationTests(unittest.TestCase):
    def test_context_comparison_uses_identical_future_targets_and_origin(self):
        records = [record("a", 24), record("b", 21)]
        short, _ = last_windows(records, 6, 6)
        long, _ = last_windows(records, 12, 6)
        for a, b in zip(short, long):
            self.assertEqual(a["origin_window"], b["origin_window"])
            np.testing.assert_array_equal(a["values"][-6:], b["values"][-6:])
            for key in COVARIATES:
                np.testing.assert_array_equal(a["past_covariates"][key][-6:], b["past_covariates"][key][-6:])

    def test_join_keeps_missing_hrv_without_losing_hr_or_inventing_values(self):
        r = record("a", 18)
        frame = pd.DataFrame({"participant_id": ["a"] * 18, "Data_collection_window": range(18),
                              "hr_ibi": r["values"], "rmssd_ibi": [np.nan] + list(range(17)),
                              "sdnn_ibi": list(range(18))})
        with patch("pregnancy_monitor.ablation.sha256", return_value=HRV29_SHA256), \
                patch("pandas.read_excel", return_value=frame):
            joined = attach_past_hrv("unused", [r], {"a"})
        self.assertEqual(len(joined[0]["values"]), 18)
        self.assertIsNone(joined[0]["past_covariates"]["rmssd_ibi"][0])
        with patch("pregnancy_monitor.ablation.sha256", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "reviewed"):
                attach_past_hrv("unused", [r], {"a"})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires optional model dependencies")
    def test_validation_never_exposes_future_target_or_future_hrv(self):
        import torch
        windows, p = last_windows([record("a")], 12, 6)
        windows[0]["values"][-6:] = 99999
        for key in COVARIATES:
            windows[0]["past_covariates"][key][-6:] = 88888
        seen = []

        class Spy:
            def predict_quantiles(self, inputs, **kwargs):
                seen.extend(inputs)
                self_kwargs = kwargs
                assert self_kwargs["cross_learning"] is False
                return [torch.tensor(np.tile([60, 80, 100], (1, 6, 1)))], None

        _, targets = validation_forecasts(Spy(), model_inputs(windows, True), p)
        self.assertTrue((targets == 99999).all())
        self.assertEqual(len(seen[0]["target"]), 12)
        self.assertNotIn(99999, seen[0]["target"])
        self.assertNotIn("future_covariates", seen[0])
        for x in seen[0]["past_covariates"].values():
            self.assertEqual(len(x), 12)
            self.assertNotIn(88888, x)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Requires optional study dependencies")
    def test_ridge_preprocessing_uses_training_women_and_equal_subject_weights(self):
        p = ResearchProfile(context_steps=6, prediction_steps=6)
        train = [record("short", 18, 70), record("long", 30, 90)]
        val, _ = last_windows([record("v", 24, 500)], 6, 6)
        pipe, audit = fit_ridge(train, val, p, True)
        X, _, _, subjects = ridge_samples(train, p, True, True)
        counts = pd.Series(subjects).value_counts()
        weights = [1 / counts[s] for s in subjects]
        self.assertAlmostEqual(pipe.named_steps["standardscaler"].mean_[6], np.average(X[:, 6], weights=weights))
        self.assertLess(pipe.named_steps["standardscaler"].mean_[6], 100)
        self.assertEqual(audit["training_subjects"], 2)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Requires optional study dependencies")
    def test_ridge_forecast_is_invariant_to_future_hr_and_hrv_poisoning(self):
        p = ResearchProfile(context_steps=12, prediction_steps=6)
        train = [record("t" + str(i), start=70 + i) for i in range(6)]
        val, _ = last_windows([record("v")], 12, 6)
        pipe, _ = fit_ridge(train, val, p, True)
        before = ridge_rows(pipe, val, p, True, "ridge")
        val[0]["values"][-6:] = 12345
        for key in COVARIATES:
            val[0]["past_covariates"][key][-6:] = 54321
        after = ridge_rows(pipe, val, p, True, "ridge")
        np.testing.assert_array_equal(before.p50, after.p50)
        self.assertTrue((after.actual == 12345).all())
        self.assertNotIn("coverage_p10_p90", summarize(after)["ridge"])

    def test_repeated_seeds_are_not_counted_as_new_women(self):
        windows, _ = last_windows([record("a"), record("b")], 12, 6)
        base = forecast_rows(windows, [np.repeat(80, 6)] * 2, BASE, 6)
        candidate = base.assign(backend=POLICY, p50=base.p50 + 1)
        rows = pd.concat([base.assign(seed=1), candidate.assign(seed=1),
                          base.assign(seed=2), candidate.assign(seed=2)])
        self.assertEqual(summarize(rows)[BASE]["unique_subjects"], 2)
        self.assertEqual(paired_comparison(rows, POLICY)["subjects"], 2)
        with self.assertRaisesRegex(ValueError, "identical targets"):
            paired_comparison(rows.iloc[:-1], POLICY)

    @unittest.skipUnless(importlib.util.find_spec("chronos") and importlib.util.find_spec("sklearn"),
                         "Requires optional study dependencies")
    def test_whole_protocol_excludes_old_test_and_saves_selection_before_assessment(self):
        import torch
        records = [record(str(i), start=100 + i) for i in range(29)]
        observed = []

        class FakePipeline:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1)
                self.model.config = type("Config", (), {})()

            def predict_quantiles(self, inputs, **kwargs):
                targets = [x["target"] if isinstance(x, dict) else x for x in inputs]
                observed.extend(int(x[0]) - 100 for x in targets)
                h = kwargs["prediction_length"]
                return [torch.tensor(np.tile([x[-1] - 10, x[-1], x[-1] + 10], (1, h, 1)),
                                     dtype=torch.float32) for x in targets], None

        def fake_fit(base, train, validation, profile, path, **kwargs):
            for item in [*train, *validation]:
                x = item["target"] if isinstance(item, dict) else item
                observed.append(int(x[0]) - 100)
            return FakePipeline(), {"best_validation_checkpoint": "checkpoint-1"}

        def load_model(path, **kwargs):
            if "adapter" in path:
                parent = Path(path).parents[1]
                # Inner restore check is allowed; assessment restores follow saved selection.
                if (parent / "selection.json").is_file():
                    observed.append(-1000)
            return FakePipeline()

        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "out"
            workbook = Path(tmp) / "fixture.xlsx"
            workbook.write_bytes(b"fixture; source join mocked")
            manifest = save_prepared(records, data, ResearchProfile(), {"dataset_id": "fixture"})
            sealed = set(map(int, manifest["splits"]["test"]))
            with patch("pregnancy_monitor.ablation.attach_past_hrv", side_effect=lambda w, r, s: r), \
                    patch("pregnancy_monitor.ablation.fit_lora_model", side_effect=fake_fit) as fit, \
                    patch("chronos.Chronos2Pipeline.from_pretrained", side_effect=load_model), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = run_ablation(data, workbook, out, seeds=(7,), num_steps=1)
            self.assertEqual(fit.call_count, 12)
            self.assertFalse(set(observed) & sealed)
            self.assertEqual(observed.count(-1000), 12)
            self.assertEqual(report["aggregate"][POLICY]["unique_subjects"], 24)
            self.assertFalse(report["original_test_evaluated"])
            self.assertFalse(report["serving_model_changed"])


if __name__ == "__main__":
    unittest.main()
