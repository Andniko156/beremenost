import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pregnancy_monitor.model_study import (
    BASE, POLICY, blend_rows, development_records, nested_folds,
    paired_subject_bootstrap, run_study, select_candidate,
)
from pregnancy_monitor.research_data import ResearchProfile, save_prepared


def forecast_rows(backend, errors, points=12):
    return pd.DataFrame([dict(backend=backend, subject_id=subject, segment_id=subject + "/v",
                              origin_window=5, horizon_step=step, actual=80.0,
                              p10=60 + error, p50=80 + error, p90=100 + error)
                         for subject, error in errors.items() for step in range(1, points + 1)])


class ModelStudyTests(unittest.TestCase):
    def test_cli_forwards_forecast_default_and_historical_loss_selection(self):
        from pregnancy_monitor.cli import build_parser, run

        for extra, expected in (([], "eval_forecast_mae"),
                                (["--selection-metric", "eval_loss"], "eval_loss")):
            args = build_parser().parse_args(["finetune", "--dataset", "unused",
                                              "--output-dir", "unused", *extra])
            with patch("pregnancy_monitor.finetuning.finetune") as train:
                run(args)
            self.assertEqual(train.call_args.kwargs["selection_metric"], expected)

    def test_nested_subject_partitions_are_disjoint_and_cover_every_development_woman_once(self):
        subjects = list(map(str, range(24)))
        plan = nested_folds(subjects)
        self.assertEqual(plan, nested_folds(list(reversed(subjects))))
        seen = []
        for fold in plan:
            a, b, c = [set(fold[k]) for k in ("train", "validation", "assessment")]
            self.assertEqual((len(a), len(b), len(c)), (12, 4, 8))
            self.assertFalse(a & b or a & c or b & c)
            self.assertEqual(a | b | c, set(subjects))
            seen.extend(c)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(subjects))

    def test_reserved_test_values_do_not_enter_development(self):
        manifest = {"splits": {"train": ["a"], "validation": ["b"], "test": ["sealed"]}}
        records = [{"subject_id": "a", "values": [70]}, {"subject_id": "b", "values": [80]},
                   {"subject_id": "sealed", "values": [999999]}]
        subjects, selected = development_records(manifest, records)
        self.assertEqual(subjects, {"a", "b"})
        self.assertEqual([r["values"] for r in selected], [[70], [80]])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            development_records(manifest, records + [records[0]])

    def test_selection_uses_inner_mae_and_retains_base_for_negligible_gain(self):
        base = forecast_rows(BASE, dict.fromkeys("abcd", 10))
        weak = forecast_rows("weak", dict.fromkeys("abcd", 9.99))
        selected, _ = select_candidate(pd.concat([base, weak]))
        self.assertEqual(selected, BASE)
        good = forecast_rows("good", dict(zip("abcd", [8, 9, 8, 9])))
        selected, scores = select_candidate(pd.concat([base, weak, good]))
        self.assertEqual(selected, "good")
        self.assertAlmostEqual(scores["good"]["relative_improvement_vs_base"], 0.15)

    def test_selection_rejects_candidate_missing_difficult_horizons(self):
        base = forecast_rows(BASE, dict.fromkeys("abcd", 10))
        incomplete = forecast_rows("incomplete", dict.fromkeys("abcd", 1)).iloc[:-1]
        with self.assertRaisesRegex(ValueError, "identical validation targets"):
            select_candidate(pd.concat([base, incomplete]))

    def test_gain_on_only_one_woman_cannot_win_selection(self):
        base = forecast_rows(BASE, dict(zip("abcd", [100, 10, 10, 10])))
        outlier_only = forecast_rows("outlier_only", dict(zip("abcd", [1, 11, 11, 11])))
        selected, _ = select_candidate(pd.concat([base, outlier_only]))
        self.assertEqual(selected, BASE)

    def test_quantile_blend_preserves_order_and_checks_alignment(self):
        base = forecast_rows(BASE, {"a": 2})
        adapted = forecast_rows("adapted", {"a": 6})
        blended = blend_rows(base, adapted, .25, "blend")
        np.testing.assert_allclose(blended.p50, 83)
        self.assertTrue((np.diff(blended[["p10", "p50", "p90"]], axis=1) >= 0).all())
        with self.assertRaises(ValueError):
            blend_rows(base, adapted.iloc[:-1], .25, "blend")
        with self.assertRaises(ValueError):
            blend_rows(base, adapted.assign(actual=90), .25, "blend")

    def test_bootstrap_resamples_women_not_points(self):
        base = pd.concat([forecast_rows(BASE, {"a": 0}, 100), forecast_rows(BASE, {"b": 0}, 2),
                          forecast_rows(BASE, {"c": 0}, 1)])
        selected = pd.concat([forecast_rows(POLICY, {"a": 1}, 100), forecast_rows(POLICY, {"b": 2}, 2),
                              forecast_rows(POLICY, {"c": 3}, 1)])
        result = paired_subject_bootstrap(pd.concat([base, selected]))
        self.assertEqual(result["mean_difference_bpm"], 2)
        self.assertEqual(result["subjects"], 3)
        self.assertEqual(result, paired_subject_bootstrap(pd.concat([base, selected])))

    def test_identical_policies_have_zero_paired_difference(self):
        base = forecast_rows(BASE, dict(zip("abc", [1, 2, 3])))
        result = paired_subject_bootstrap(pd.concat([base, base.assign(backend=POLICY)]))
        self.assertEqual(result["conditional_percentile_95_interval_bpm"], [0, 0])

    @unittest.skipUnless(importlib.util.find_spec("chronos"), "Requires optional model dependencies")
    def test_full_study_excludes_original_test_from_every_fit_and_forecast(self):
        import torch
        records = [{"subject_id": str(i), "recording_id": "control", "segment_id": str(i) + "/control",
                    "first_window": 0, "values": [100 + i + j / 10 for j in range(24)]} for i in range(29)]
        fitted, forecasted = [], []

        class FakePipeline:
            def __init__(self, offset=0):
                self.offset = offset
                self.model = torch.nn.Linear(1, 1)

            def predict_quantiles(self, histories, **kwargs):
                forecasted.extend(int(x[0]) - 100 for x in histories)
                return [torch.tensor(np.tile([x[-1] - 10, x[-1] + self.offset, x[-1] + 10],
                                             (1, 12, 1)), dtype=torch.float32) for x in histories], None

        def fake_fit(base, train, validation, *args, **kwargs):
            fitted.extend(int(x[0]) - 100 for x in [*train, *validation])
            self.assertEqual(kwargs["selection_metric"], "eval_forecast_mae")
            return FakePipeline(.2), {"best_validation_checkpoint": "checkpoint-1"}

        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "out"
            manifest = save_prepared(records, data, ResearchProfile(), {"dataset_id": "fixture"})
            sealed = set(map(int, manifest["splits"]["test"]))
            with patch("chronos.Chronos2Pipeline.from_pretrained", return_value=FakePipeline()), \
                    patch("pregnancy_monitor.model_study.fit_lora_model", side_effect=fake_fit) as fit, \
                    contextlib.redirect_stdout(io.StringIO()):
                report = run_study(data, out, num_steps=1)
            self.assertEqual(fit.call_count, 6)
            self.assertFalse(set(fitted) & sealed)
            self.assertFalse(set(forecasted) & sealed)
            self.assertFalse(report["original_test_evaluated"])
            self.assertEqual(report["aggregate"][POLICY]["n"], 24 * 12)
            self.assertTrue((out / "study_plan.json").is_file())


if __name__ == "__main__":
    unittest.main()
