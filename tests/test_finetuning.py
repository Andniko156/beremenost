import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pregnancy_monitor.finetuning import evaluate_models
from pregnancy_monitor.research_data import (
    ResearchProfile, check_parent_holdouts, evaluation_windows, load_prepared, save_prepared,
    segment_visits, sha256, subject_split, training_windows, write_json,
)


class ResearchDataTests(unittest.TestCase):
    def visits(self):
        return {visit: pd.DataFrame([
            {"participant_id": str(subject), "Data_collection_window": i,
             "hr_ibi": 70 + subject + i / 10}
            for subject in range(10) for i in range(length)])
            for visit, length in (("control", 24), ("stress", 6))}

    def test_missing_values_and_time_gaps_split_visits_without_imputation(self):
        frame = pd.DataFrame({"participant_id": ["s"] * 6,
                              "Data_collection_window": [0, 1, 2, 4, 5, 6],
                              "hr_ibi": [70, 71, np.nan, 74, 0, 76]})
        rows, audit = segment_visits({"visit": frame})
        self.assertEqual([r["values"] for r in rows], [[70, 71], [74], [76]])
        self.assertEqual([r["first_window"] for r in rows], [0, 4, 6])
        self.assertEqual(audit["excluded_invalid_hr_rows"], 2)

    def test_duplicate_and_noninteger_relative_times_are_rejected(self):
        frame = self.visits()["control"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            segment_visits({"v": pd.concat([frame, frame.iloc[:1]])})
        with self.assertRaisesRegex(ValueError, "integers"):
            segment_visits({"v": frame.assign(Data_collection_window=0.5)})

    def test_visits_are_never_joined_to_create_fake_long_histories(self):
        rows, _ = segment_visits(self.visits())
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(len(r["values"]) >= 18 for r in rows), 10)
        self.assertEqual(max(len(r["values"]) for r in rows), 24)

    def test_subject_split_is_deterministic_and_disjoint(self):
        a = subject_split(map(str, range(29)), 42)
        self.assertEqual(a, subject_split(map(str, reversed(range(29))), 42))
        self.assertEqual({k: len(v) for k, v in a.items()}, {"train": 20, "validation": 4, "test": 5})
        self.assertEqual(len(set().union(*map(set, a.values()))), 29)

    def test_prepared_data_detects_tampering_and_split_overlap(self):
        rows, _ = segment_visits(self.visits())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prepared"
            manifest = save_prepared(rows, out, ResearchProfile(), {"dataset_id": "fixture"})
            load_prepared(out)
            manifest["splits"]["train"].append(manifest["splits"]["test"][0])
            write_json(out / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_prepared(out)
            manifest["splits"]["train"].pop()
            write_json(out / "manifest.json", manifest)
            with (out / "segments.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(out)

    def test_train_and_test_windows_contain_distinct_subjects(self):
        rows, _ = segment_visits(self.visits())
        profile = ResearchProfile()
        train = training_windows(rows, {"0"}, profile)
        heldout = evaluation_windows(rows, {"1"}, profile)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(heldout), 1)
        self.assertEqual(heldout[0]["subject_id"], "1")
        self.assertEqual(heldout[0]["origin_window"], 11)
        np.testing.assert_allclose(train[0], np.arange(18) / 10 + 70)
        np.testing.assert_allclose(heldout[0]["values"], np.arange(6, 24) / 10 + 71)

    def test_evaluation_exposes_context_only_and_baselines_have_no_fake_intervals(self):
        import torch
        observed = []

        class Spy:
            def predict_quantiles(self, histories, **kwargs):
                observed.extend(x.copy() for x in histories)
                self.kwargs = kwargs
                return [torch.tensor(np.tile([60, 70, 90], (1, 12, 1)), dtype=torch.float32)
                        for _ in histories], None

        records, _ = segment_visits(self.visits())
        windows = evaluation_windows(records, {"1"}, ResearchProfile())
        spy = Spy()
        rows, report = evaluate_models(windows, ResearchProfile(), {"spy": spy})
        np.testing.assert_array_equal(observed[0], windows[0]["values"][:6])
        self.assertFalse(spy.kwargs["cross_learning"])
        self.assertEqual(len(rows), 36)
        self.assertNotIn("coverage_p10_p90", report["aggregate"]["persistence"])
        self.assertEqual(report["aggregate"]["spy"]["n"], 12)

    def test_model_forecast_with_crossing_quantiles_is_rejected(self):
        import torch

        class Crossing:
            def predict_quantiles(self, histories, **kwargs):
                return [torch.tensor(np.tile([90, 70, 60], (1, 12, 1))) for _ in histories], None

        rows, _ = segment_visits(self.visits())
        with self.assertRaisesRegex(ValueError, "crossing"):
            evaluate_models(evaluation_windows(rows, {"0"}, ResearchProfile()), ResearchProfile(), {"bad": Crossing()})

    def test_prepared_dataset_requires_maternal_pregnancy_signal(self):
        rows, _ = segment_visits(self.visits())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prepared"
            manifest = save_prepared(rows, out, ResearchProfile(), {})
            manifest["signal"] = "fetal_heart_rate"
            write_json(out / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "maternal"):
                load_prepared(out)

    def test_continuation_keeps_former_test_women_out_of_selection(self):
        manifest = {"subject_namespace": "clinic", "source": {},
                    "splits": {"train": ["a"], "validation": ["b"], "test": ["c"]}}
        lineage = {"reserved_holdouts": [{"subject_namespace": "clinic", "validation": ["b"], "test": ["c"]}]}
        check_parent_holdouts(manifest, lineage)
        manifest["splits"]["validation"] = ["c"]
        with self.assertRaisesRegex(ValueError, "held-out"):
            check_parent_holdouts(manifest, lineage)
        manifest["splits"]["validation"] = ["b"]
        manifest["splits"]["train"] = ["b"]
        with self.assertRaisesRegex(ValueError, "held-out"):
            check_parent_holdouts(manifest, lineage)


if __name__ == "__main__":
    unittest.main()
