import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vessel_power.configuration import ROOT, PARAMETERS, CHANNELS
from vessel_power.preprocessing import (
    clean_records,
    minute_end,
    identity_kw,
    read_month,
    align_months,
)
from vessel_power.reliability import (
    FEATURES,
    NEAR_ZERO,
    assess_reliability,
    label_eligibility,
    features,
    inference_scope,
    inference_gate,
    split_of,
)
from vessel_power.modeling import (
    coefficient,
    score,
    historical_members,
    updates,
    FIRST,
    DELAY,
    METHODS,
)

OUTPUT = ROOT / "results"
PROCESSED = ROOT / "data/processed/measurements.pkl"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = pd.read_pickle(PROCESSED)
        cls.z = pd.read_pickle(OUTPUT / "estimates.pkl")
        cls.x = features(cls.d)
        cls.state = json.loads((OUTPUT / "model.json").read_text())

    def example(self):
        m = self.d.power_min.gt(0) & self.d.power_max.le(100) & self.d.engine_rpm.abs().gt(20)
        return self.d.loc[[self.d.index[m][0]]].copy()

    def reason(self, d, primary, tokens):
        q = label_eligibility(d)
        self.assertEqual(q.exclusion_reason.iloc[0], primary)
        detail = set(q.diagnostic_reason_codes.iloc[0].split("|"))
        self.assertTrue(set(tokens) <= detail)
        self.assertTrue((set(q.previous_reason_codes.iloc[0].split("|")) - {""}) <= detail)
        return q

    def test_positive_near_zero_is_unresolved(self):
        q = self.reason(self.example(), NEAR_ZERO, [NEAR_ZERO])
        self.assertEqual(q.recomputed_status.iloc[0], "UNRESOLVED")
        self.assertFalse(q.eval_label_eligible.any())

    def test_active_zero_primary_and_complete_evidence(self):
        d = self.example()
        for c in ["power", "torque", "shaft_rpm"]:
            for s in ["", "_min", "_max"]:
                d[c + s] = 0
        q = self.reason(
            d,
            "active_engine_zero",
            ["zero_with_engine_activity", NEAR_ZERO, "mechanical_zero_with_engine_activity"],
        )
        self.assertEqual(q.recomputed_status.iloc[0], "SUSPECT")

    def test_negative_sign_reason(self):
        d = self.example()
        d[["power", "power_min", "power_max"]] = -50
        self.reason(d, "negative_sign_unresolved", ["negative_with_engine_activity", NEAR_ZERO])

    def test_mixed_sign_reason(self):
        for lo, hi in [(-10, 50), (0, 50)]:
            d = self.example()
            d["power"] = 20
            d["power_min"] = lo
            d["power_max"] = hi
            self.reason(d, "mixed_power_sign_unresolved", ["mixed_power_sign", NEAR_ZERO])

    def test_contradictions_take_precedence(self):
        for field, reason in [
            ("magnitude_bad", "magnitude_identity_inconsistent"),
            ("conflicting_keys", "source_conflict"),
        ]:
            d = self.example()
            d[field] = 1
            q = self.reason(d, reason, [reason, NEAR_ZERO])
            self.assertEqual(q.recomputed_status.iloc[0], "SUSPECT")

    def test_missing_support_is_specific(self):
        d = self.example()
        d["scavenge"] = np.nan
        d["scavenge_count"] = 0
        self.reason(
            d,
            "insufficient_samples",
            ["insufficient_samples", "independent_inputs_incomplete", NEAR_ZERO],
        )

    def test_eligible_label_reason(self):
        d = self.example()
        d[["power", "power_min", "power_max"]] = 200
        q = self.reason(d, "eligible_conditional_proxy", [])
        self.assertTrue(q.eval_label_eligible.all())

    def test_stationary_idle_zero_retained(self):
        d = self.example()
        for c in ["power", "torque", "shaft_rpm", "engine_rpm", "turbo", "fuel_load", "stw", "sog"]:
            for s in ["", "_min", "_max"]:
                d[c + s] = 0
        d[["magnitude_bad", "scaled_bad", "registered_bad"]] = 0
        self.assertTrue(label_eligibility(d).eval_label_eligible.all())

    def test_sign_convention_reason(self):
        d = self.example()
        d["scaled_bad"] = 1
        d["magnitude_bad"] = 0
        self.reason(
            d, "sign_convention_unresolved", ["identity_sign_convention_unresolved", NEAR_ZERO]
        )

    def test_exact_near_zero_boundary(self):
        d = self.example()
        d[["power", "power_min", "power_max"]] = 100
        self.assertFalse(label_eligibility(d).eval_label_eligible.any())
        d["power_max"] = 100.001
        self.assertTrue(label_eligibility(d).eval_label_eligible.all())

    def test_exact_completed_minute_boundaries(self):
        t = pd.Series(
            pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:00:59.999Z", "2024-01-01T00:01:00Z"],
                format="ISO8601",
            )
        )
        self.assertEqual(
            minute_end(t).tolist(),
            pd.to_datetime(
                ["2024-01-01T00:01Z", "2024-01-01T00:01Z", "2024-01-01T00:02Z"]
            ).tolist(),
        )

    def test_timezone_grid_exposes_gap_without_imputing_values(self):
        idx = pd.to_datetime(["2024-01-01T00:01Z", "2024-01-01T00:03Z"])
        frame = pd.DataFrame({"power": [10.0, 20.0], "conflicting_keys": 0}, index=idx)
        for c in [
            "power",
            "torque",
            "shaft_rpm",
            "engine_rpm",
            "turbo",
            "fuel_load",
            "scavenge",
            "stw",
            "sog",
            "turn",
        ]:
            frame[c + "_count"] = 4
        result = align_months([frame])
        self.assertEqual(len(result), 3)
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertTrue(pd.isna(result.power.iloc[1]))
        self.assertEqual(result.power_count.iloc[1], 0)
        self.assertEqual(result.power.iloc[2], 20.0)
        with self.assertRaises(ValueError):
            align_months([frame, frame])

    def test_exact_split_boundaries(self):
        idx = pd.to_datetime(
            ["2024-06-01T00:00Z", "2024-06-01T00:01Z", "2024-09-01T00:00Z", "2024-09-01T00:01Z"]
        )
        self.assertEqual(split_of(idx).tolist(), ["fit", "select", "calibrate", "evaluate"])

    def test_duplicate_and_conflicting_key_handling(self):
        t = pd.Timestamp("2024-01-01", tz="UTC")
        d = pd.DataFrame(
            {
                "ts": [t, t, t, t],
                "channel": ["power", "power", "torque", "torque"],
                "value": [1.0, 1.0, 2.0, 3.0],
            }
        )
        cleaned, conflicts = clean_records(d)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(conflicts), 2)

    def test_original_format_reader(self):
        reverse = {v: q for q, v, *_ in CHANNELS}
        values = {
            "power": 200.0,
            "torque": 200 / (30 * 2 * np.pi / 60),
            "shaft_rpm": 30.0,
            "engine_rpm": 30.0,
            "turbo": 2500.0,
            "fuel_load": 20.0,
            "scavenge": 0.1,
            "stw": 2.0,
            "sog": 2.0,
            "turn": 0.0,
        }
        with tempfile.TemporaryDirectory(dir=OUTPUT, prefix="source_test_") as td:
            rows = []
            for second in [0, 15, 30, 45]:
                for channel, value in values.items():
                    rows.append([f"2024-01-01T00:00:{second:02d}.000Z", reverse[channel], value])
            rows.append(rows[0])
            path = Path(td) / "1.csv"
            pd.DataFrame(rows).to_csv(path, header=False, index=False)
            d, inventory = read_month(path)
            self.assertEqual(inventory["exact_duplicate_copies"], 1)
            self.assertEqual(int(d.power_count.iloc[0]), 4)
            self.assertTrue(label_eligibility(d).eval_label_eligible.all())

    def test_shared_torquemeter_sources_are_excluded_features(self):
        for _, name, source, _, _ in CHANNELS:
            if source == "Instrument Torquemeter":
                self.assertNotIn(name, FEATURES)

    def test_torque_scale_is_only_an_internal_hypothesis(self):
        self.assertAlmostEqual(identity_kw(100, 60, 1000) / identity_kw(100, 60), 1000)
        self.assertFalse(self.z.physical_error_available.any())

    def test_hiding_target_does_not_change_features_or_gate(self):
        d = self.d.iloc[::1001].copy()
        h = d.copy()
        for c in ["power", "torque", "shaft_rpm"]:
            for s in ["", "_min", "_max", "_count"]:
                h[c + s] = np.nan
        h["status"] = "SUSPECT"
        h[NEAR_ZERO] = True
        pd.testing.assert_frame_equal(features(d), features(h))
        c, r = inference_scope(d)
        cc, rr = inference_scope(h)
        pd.testing.assert_series_equal(
            inference_gate(features(d), PARAMETERS["domain"], c, r),
            inference_gate(features(h), PARAMETERS["domain"], cc, rr),
        )

    def test_accepted_label_counts(self):
        expected = {
            "fit": 180673,
            "select": 54535,
            "calibrate": 36265,
            "evaluate": 115973,
            "unlabeled": 0,
        }
        self.assertEqual(self.z.groupby("split").eval_label_eligible.sum().to_dict(), expected)

    def test_no_negative_or_uncertain_zero_scored(self):
        h = self.z[self.z.proxy_error_allowed]
        self.assertTrue(h.power.ge(0).all())
        self.assertFalse(h.zero_with_engine_activity.any())

    def test_january_abstention(self):
        h = self.z[self.z.split.eq("unlabeled")]
        self.assertTrue(h.power.isna().all())
        self.assertFalse(h.served.any())
        self.assertTrue(h[METHODS].isna().all().all())

    def test_model_initialization_respects_assumed_arrival(self):
        self.assertLessEqual(pd.Timestamp(self.state["training_last"]) + DELAY, FIRST)
        self.assertLessEqual(pd.Timestamp(self.state["selector_last"]) + DELAY, FIRST)
        self.assertTrue(self.z.loc[:FIRST, METHODS].isna().all().all())

    def test_update_membership_and_arrival(self):
        ledger = pd.read_pickle(OUTPUT / "update_history.pkl")
        for r in ledger.itertuples():
            h = historical_members(self.z, r.update_at)
            self.assertEqual(len(h), r.labels)
            self.assertEqual(hashlib.sha256(h.index.asi8.tobytes()).hexdigest(), r.member_sha256)
            self.assertTrue((h.index + DELAY <= r.update_at).all())
            self.assertFalse(h.previous_reason_codes.str.contains(NEAR_ZERO).any())

    def test_future_labels_do_not_change_current_update(self):
        day = pd.Timestamp("2024-10-10", tz="UTC")
        z = self.z.loc["2024-09-29":"2024-10-11"].copy()
        x = self.x.loc[z.index]
        p, q, _ = updates(z, x, self.state["coefficient"], [day])
        m = z.index > day - DELAY
        z.loc[m, "power"] = 1e12
        z.loc[m, "train_label_eligible"] = False
        pp, qq, _ = updates(z, x, self.state["coefficient"], [day])
        pd.testing.assert_series_equal(p, pp)
        pd.testing.assert_series_equal(q, qq)

    def test_missing_history_falls_back(self):
        day = pd.Timestamp("2024-10-10", tz="UTC")
        z = self.z.loc["2024-09-29":"2024-10-11"].copy()
        x = self.x.loc[z.index]
        z["train_label_eligible"] = False
        _, _, ledger = updates(z, x, self.state["coefficient"], [day])
        self.assertFalse(ledger.rolling_updated.any())
        self.assertEqual(ledger.coefficient.iloc[0], self.state["coefficient"])

    def test_coefficient_and_metric_known_values(self):
        self.assertAlmostEqual(coefficient([10, 20], [10, 80])[0], 0.01)
        self.assertEqual(coefficient([0, 0], [0, 0], 0.02)[0], 0.02)
        m = score([1, 2, 3], [2, 3, 4])
        self.assertEqual(m["mae"], 1)
        self.assertAlmostEqual(m["pearson_r"], 1)
        self.assertAlmostEqual(m["r_squared"], -0.5)
        with self.assertRaises(ValueError):
            score([1, np.nan], [1, 2])

    def test_archive_access_is_blocked(self):
        with self.assertRaises(PermissionError):
            (ROOT / "archive/forbidden_runtime_dependency").open()


if __name__ == "__main__":
    unittest.main()
