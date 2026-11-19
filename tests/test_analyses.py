import json
import unittest
import numpy as np
import pandas as pd

from vessel_power.configuration import ROOT, PARAMETERS
from vessel_power.model_comparison import make_model, SimpleBaseline, selection_scores, DEFINITIONS
from vessel_power.change_analysis import (
    bh,
    permutation_test,
    block_permutation_test,
    compare_table,
    SETTINGS,
)


class FrozenDefinitions(unittest.TestCase):
    def test_original_seven_definitions(self):
        self.assertEqual(
            set(DEFINITIONS["models"]),
            {"mean", "speed_cubic", "rpm_cubic", "ridge_1", "ridge_100", "hgb_15", "hgb_31"},
        )
        for name, alpha in [("ridge_1", 1), ("ridge_100", 100)]:
            self.assertEqual(make_model(name).named_steps["ridge"].alpha, alpha)
        for name, leaves in [("hgb_15", 15), ("hgb_31", 31)]:
            p = make_model(name).get_params()
            self.assertEqual(p["max_leaf_nodes"], leaves)
            self.assertEqual(p["max_iter"], 120)
            self.assertEqual(p["min_samples_leaf"], 80)
            self.assertEqual(p["l2_regularization"], 10)
            self.assertEqual(p["learning_rate"], 0.08)
            self.assertEqual(p["random_state"], 731)
            self.assertFalse(p["early_stopping"])

    def test_simple_baselines_training_arithmetic(self):
        x = pd.DataFrame({"stw": [1.0, 2.0], "engine_rpm_abs": [10.0, 20.0]})
        np.testing.assert_array_equal(
            SimpleBaseline("mean").fit(x, [1.0, 3.0]).predict(x), [2.0, 2.0]
        )
        self.assertAlmostEqual(SimpleBaseline("rpm_cubic").fit(x, [10.0, 80.0]).value, 0.01)
        self.assertAlmostEqual(SimpleBaseline("speed_cubic").fit(x, [10.0, 80.0]).value, 10.0)

    def test_selection_uses_only_chronological_selection_rows(self):
        idx = pd.to_datetime(["2024-06-02T12:00Z", "2024-06-02T13:00Z", "2024-09-01T12:00Z"])
        z = pd.DataFrame(
            {
                "eval_label_eligible": True,
                "served": True,
                "split": ["select", "select", "evaluate"],
                "inference_regime": ["active", "idle", "active"],
                "power": [10.0, 0.0, 10.0],
                "a": [12.0, 0.0, 1.0],
                "b": [11.0, 3.0, 10.0],
            },
            index=idx,
        )
        a = selection_scores(z, ["a", "b"])
        z.loc[idx[-1], ["power", "a", "b"]] = [1e12, -1e12, 5e12]
        pd.testing.assert_frame_equal(a, selection_scores(z, ["a", "b"]))
        self.assertEqual(a.iloc[0].model, "a")
        self.assertEqual(a.iloc[0].selection_score, 1.0)

    def test_bh_known_values(self):
        np.testing.assert_allclose(bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])

    def test_permutations_are_deterministic(self):
        a = np.arange(14.0)
        b = np.arange(14.0) + 10
        self.assertEqual(permutation_test(a, b), permutation_test(a, b))
        d = np.r_[a, b]
        d[2] = np.nan
        self.assertEqual(block_permutation_test(d), block_permutation_test(d))

    def test_block_schedule_and_frozen_settings(self):
        with self.assertRaises(AssertionError):
            block_permutation_test(np.arange(27.0))
        self.assertEqual(SETTINGS["permutations"], 999)
        self.assertEqual(SETTINGS["block_days"], 3)
        self.assertEqual(SETTINGS["side_window_days"], 14)
        self.assertEqual(SETTINGS["fdr"], 0.05)

    def test_evidence_comparison_does_not_hide_differences(self):
        a = pd.DataFrame({"boundary": ["x"], "delta": [1.0], "q": [0.01]})
        self.assertTrue(compare_table(a, a.copy())["passed"])
        b = a.copy()
        b["q"] = 0.05
        self.assertFalse(compare_table(a, b)["passed"])


@unittest.skipUnless(
    (ROOT / "results/model_comparison/verification.json").exists(),
    "Run optional compare command first",
)
class CandidateReproduction(unittest.TestCase):
    def test_v2_training_and_no_reselection(self):
        v = json.loads((ROOT / "results/model_comparison/verification.json").read_text())
        self.assertEqual(v["training_labels"], 180673)
        self.assertEqual(v["unresolved_training_labels_not_reintroduced"], 2)
        self.assertFalse(v["final_model_reselected"])
        self.assertTrue(v["final_artifacts_unchanged"])
        self.assertEqual(v["final_selected_model"], PARAMETERS["selected_family"])
        self.assertFalse(v["historical_scores_exact"])

    def test_hyperparameters_and_roundoff_report(self):
        v = json.loads((ROOT / "results/model_comparison/verification.json").read_text())
        self.assertTrue(v["hyperparameters_unchanged"])
        self.assertLess(abs(v["rpm_coefficient_difference_vs_final"]), 1e-14)
        states = json.loads((ROOT / "results/model_comparison/fit_states.json").read_text())
        self.assertEqual(states["hgb_15"]["iterations"], 120)
        self.assertEqual(states["hgb_31"]["iterations"], 120)

    def test_current_and_historical_evaluation_population_equal(self):
        m = pd.read_csv(ROOT / "results/model_comparison/evaluation_parity.csv")
        self.assertTrue(m.n_v2_refit.equals(m.n_historical))
        self.assertTrue(m[m.regime.eq("active")].n_v2_refit.eq(101998).all())


@unittest.skipUnless(
    (ROOT / "results/change_analysis/verification.json").exists(),
    "Run optional changes command first",
)
class InvestigationReproduction(unittest.TestCase):
    def test_all_change_statistics_and_adjustments_reproduce(self):
        v = json.loads((ROOT / "results/change_analysis/verification.json").read_text())
        self.assertTrue(v["passed"])
        self.assertEqual(v["scheduled_tests"], 392)
        self.assertEqual(v["testable_tests"], 204)
        self.assertEqual(v["dependence_sensitive_candidates"], 5)
        self.assertEqual(v["power_candidates_passing_both_screens"], 0)
        for check in v["parity"].values():
            self.assertTrue(check["passed"])
            self.assertTrue(
                all(diff == 0 for diff in check["maximum_numeric_differences"].values())
            )

    def test_transition_timestamp_and_missing_boundary_semantics(self):
        t = pd.read_csv(ROOT / "results/change_analysis/transitions.csv").set_index("event")
        self.assertEqual(
            pd.Timestamp(t.loc["june_zero_onset", "verified_raw_transition"]),
            pd.Timestamp("2024-06-10T10:40:45.003Z"),
        )
        self.assertEqual(
            pd.Timestamp(t.loc["august_zero_onset", "verified_raw_transition"]),
            pd.Timestamp("2024-08-27T20:01:30.007Z"),
        )
        self.assertTrue(pd.isna(t.loc["june_zero_run_end_gap", "verified_raw_transition"]))
        self.assertEqual(t.loc["june_zero_run_end_gap", "recorded_state_after"], "missing_boundary")

    def test_independent_evidence_and_alignment(self):
        s = pd.read_csv(ROOT / "results/change_analysis/transition_evidence.csv")
        self.assertEqual(len(s), 96)
        self.assertTrue(s[s.channel.eq("fuel_flow")].source.eq("torquemeter").all())
        self.assertTrue(s[s.channel.eq("engine_rpm")].source.eq("separately_registered").all())
        a = pd.read_csv(ROOT / "results/change_analysis/alignment.csv")
        self.assertEqual(len(a), 56)
        self.assertTrue(a.max_absolute_offset_seconds.le(1).all())


if __name__ == "__main__":
    unittest.main()
