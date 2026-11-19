import numpy as np
import pandas as pd

POLICY_VERSION = "professor_aligned_v2"
SOURCES = ["engine_rpm", "turbo", "fuel_load", "scavenge", "stw", "sog", "turn"]
FEATURES = ["engine_rpm_abs", "turbo", "fuel_load", "scavenge", "stw", "sog", "turn_abs"]
RPM_ABS = 4.188168054399988
NEAR_ZERO = "near_zero_with_engine_activity"
METADATA_REVISION = "primary_reason_priority_v1"


def assess_reliability(measurements, rpm_abs=2, rel=0.05):
    """Recompute from measurements; never consume stored quality decisions."""
    d = measurements.copy()
    core = ["power", "shaft_rpm", "torque", "engine_rpm", "turbo", "fuel_load"]
    d["insufficient_support"] = False
    for c in core:
        d["insufficient_support"] |= d[c + "_count"].fillna(0).lt(3)
    d["source_conflict"] = d.conflicting_keys.gt(0)
    d["active_engine"] = d.engine_rpm.abs().gt(20) & d.turbo.gt(1000) & d.fuel_load.gt(5)
    d["idle_engine"] = d.engine_rpm.abs().le(2) & d.turbo.abs().le(1000) & d.fuel_load.abs().le(1)
    d["mixed_power_sign"] = (
        d.power_min.lt(0) & d.power_max.gt(0)
        | d.power_min.eq(0) & d.power_max.gt(0)
        | d.power_min.lt(0) & d.power_max.eq(0)
    )
    d["power_sign"] = np.select(
        [
            d.power_count.eq(0),
            d.mixed_power_sign,
            d.power_max.lt(0),
            d.power_min.eq(0) & d.power_max.eq(0),
            d.power_min.gt(0),
        ],
        ["missing", "mixed", "negative", "zero", "positive"],
        default="unresolved",
    )
    d["registered_identity_inconsistent"] = d.registered_bad.gt(0)
    d["scaled_identity_inconsistent"] = d.scaled_bad.gt(0)
    d["magnitude_identity_inconsistent"] = d.magnitude_bad.gt(0)
    d["identity_sign_convention_unresolved"] = (
        d.scaled_identity_inconsistent & ~d.magnitude_identity_inconsistent
    )
    d["paired_coverage_low"] = d.paired_count.fillna(0).lt(
        d.power_count.fillna(0) * 0.75
    ) | d.paired_count.fillna(0).lt(3)
    d["rpm_disagreement"] = (
        (d.shaft_rpm.abs() - d.engine_rpm.abs())
        .abs()
        .gt(np.maximum(rpm_abs, rel * d.engine_rpm.abs()))
        & d.shaft_rpm.notna()
        & d.engine_rpm.notna()
    )
    d["rotation_direction_disagreement"] = (
        (d.shaft_rpm * d.engine_rpm).lt(0) & d.shaft_rpm.abs().gt(20) & d.engine_rpm.abs().gt(20)
    )
    d["zero_with_engine_activity"] = d.power_sign.eq("zero") & d.active_engine
    d["negative_with_engine_activity"] = d.power_min.lt(0) & d.active_engine
    d[NEAR_ZERO] = d.power_min.abs().le(100) & d.power_max.abs().le(100) & d.active_engine
    d["mechanical_zero_with_engine_activity"] = (
        d.zero_with_engine_activity
        & d.shaft_rpm_min.eq(0)
        & d.shaft_rpm_max.eq(0)
        & d.torque_min.eq(0)
        & d.torque_max.eq(0)
    )
    d["positive_with_idle_engine"] = d.power_sign.eq("positive") & d.idle_engine
    d["navigation_motion_at_zero"] = d.power_sign.eq("zero") & (
        d.sog.abs().gt(3) | d.stw.abs().gt(3)
    )
    review = [
        "source_conflict",
        "magnitude_identity_inconsistent",
        "identity_sign_convention_unresolved",
        "rpm_disagreement",
        "rotation_direction_disagreement",
        "zero_with_engine_activity",
        "negative_with_engine_activity",
        "positive_with_idle_engine",
        "mixed_power_sign",
    ]
    strong = (
        d.source_conflict
        | d.magnitude_identity_inconsistent
        | d.mechanical_zero_with_engine_activity
    )
    ambiguous = (
        d.rpm_disagreement
        | d.rotation_direction_disagreement
        | d.identity_sign_convention_unresolved
        | d.zero_with_engine_activity
        | d.negative_with_engine_activity
        | d.positive_with_idle_engine
        | d.mixed_power_sign
        | d[NEAR_ZERO]
    )
    d["review_evidence"] = d[review + [NEAR_ZERO]].any(axis=1)
    d["status"] = np.select(
        [strong, d.power_count.eq(0) | d.insufficient_support | d.paired_coverage_low | ambiguous],
        ["SUSPECT", "UNRESOLVED"],
        default="CONSISTENT_CANDIDATE",
    )
    d["candidate_consistency_training"] = d.status.eq("CONSISTENT_CANDIDATE")
    d["physical_truth_training_approved"] = False
    d["physical_truth_evaluation_eligible"] = False
    codes = pd.Series("", index=d.index, dtype="str")
    # Keep established v2 reason order so regression can compare exact strings.
    for c in review + [
        "insufficient_support",
        "paired_coverage_low",
        "registered_identity_inconsistent",
        NEAR_ZERO,
    ]:
        codes = codes.where(~d[c], codes + "|" + c)
    d["reason_codes"] = codes.str.lstrip("|")
    return d


def features(d):
    return pd.DataFrame(
        {
            "engine_rpm_abs": d.engine_rpm.abs(),
            "turbo": d.turbo,
            "fuel_load": d.fuel_load,
            "scavenge": d.scavenge,
            "stw": d.stw,
            "sog": d.sog,
            "turn_abs": d.turn.abs(),
        },
        index=d.index,
    )


def inference_scope(d, min_count=3):
    complete = pd.Series(True, index=d.index)
    for c in SOURCES:
        complete &= np.isfinite(d[c]) & d[c + "_count"].ge(min_count)
    active = d.engine_rpm.abs().gt(20) & d.turbo.gt(1000) & d.fuel_load.gt(5) & d.scavenge.gt(0)
    idle = (
        d.engine_rpm.abs().le(2)
        & d.turbo.abs().le(1000)
        & d.fuel_load.abs().le(1)
        & d.stw.abs().le(1)
        & d.sog.abs().le(1)
    )
    regime = pd.Series(
        np.select([active, idle], ["active", "idle"], default="transition"), index=d.index
    )
    return complete, regime


def exclusion_metadata(q, accepted, supported, complete):
    """Deterministic descriptive provenance; never changes measurement decisions.

    An exclusion or numerical contradiction is not a confirmed sensor fault.
    Keep established reason strings intact and supplement their evidence rather
    than allowing the broad near-zero diagnostic to hide a specific reason.
    """
    primary = pd.Series(
        np.select(
            [
                accepted,
                q.source_conflict,
                q.magnitude_identity_inconsistent,
                q.zero_with_engine_activity,
                q.mixed_power_sign,
                q.power_min.lt(0),
                q.identity_sign_convention_unresolved,
                q.rotation_direction_disagreement,
                ~supported,
                q.paired_coverage_low,
                q.rpm_disagreement,
                q.positive_with_idle_engine,
                q[NEAR_ZERO] & q.power_sign.eq("positive"),
                q.status.ne("CONSISTENT_CANDIDATE"),
            ],
            [
                "eligible_conditional_proxy",
                "source_conflict",
                "magnitude_identity_inconsistent",
                "active_engine_zero",
                "mixed_power_sign_unresolved",
                "negative_sign_unresolved",
                "sign_convention_unresolved",
                "rotation_direction_unresolved",
                "insufficient_samples",
                "paired_coverage_low",
                "rpm_disagreement",
                "positive_with_idle_engine",
                NEAR_ZERO,
                "measurement_review_or_unresolved",
            ],
            default="outside_supported_label_regimes",
        ),
        index=q.index,
    )
    detailed = q.reason_codes.copy()
    additions = [
        ("mechanical_zero_with_engine_activity", q.mechanical_zero_with_engine_activity),
        ("navigation_motion_at_zero", q.navigation_motion_at_zero),
        ("independent_inputs_incomplete", ~complete),
        ("insufficient_samples", ~supported),
    ]
    additions += [(str(reason), primary.eq(reason) & ~accepted) for reason in primary.unique()]
    for code, mask in additions:
        present = detailed.str.contains(r"(?:^|\|)" + code + r"(?:\||$)", regex=True)
        add = mask & ~present
        detailed.loc[add] = (detailed.loc[add] + "|" + code).str.strip("|")
    return primary, detailed


def label_eligibility(d, min_count=3, rpm_scale=1):
    q = assess_reliability(d, rpm_abs=RPM_ABS * rpm_scale)
    complete, regime = inference_scope(d, min_count)
    positive = d.power_min.gt(0) & regime.eq("active")
    zero = d.power_min.eq(0) & d.power_max.eq(0) & regime.eq("idle")
    for c in ["torque", "shaft_rpm"]:
        zero &= d[c + "_min"].eq(0) & d[c + "_max"].eq(0)
    supported = complete.copy()
    for c in ["power", "torque", "shaft_rpm", *SOURCES]:
        supported &= d[c + "_count"].ge(min_count)
    accepted = q.status.eq("CONSISTENT_CANDIDATE") & supported & (positive | zero)
    out = pd.DataFrame(
        {
            "train_label_eligible": accepted,
            "eval_label_eligible": accepted,
            "independent_complete": complete,
            "inference_regime": regime,
            "recomputed_status": q.status,
            "label_regime": np.select(
                [accepted & positive, accepted & zero],
                ["positive_active", "idle_zero"],
                default="uncertain",
            ),
        },
        index=d.index,
    )
    out["exclusion_reason"], out["diagnostic_reason_codes"] = exclusion_metadata(
        q, accepted, supported, complete
    )
    out["previous_reason_codes"] = q.reason_codes
    out["policy_version"] = POLICY_VERSION
    return out


def inference_gate(x, domain, complete, regime):
    supported = x.ge(pd.Series(domain["low"])).all(axis=1) & x.le(pd.Series(domain["high"])).all(
        axis=1
    )
    return complete & regime.isin(["active", "idle"]) & supported


def split_of(index):
    start = index - pd.Timedelta(minutes=1)
    return pd.Series(
        np.select(
            [
                start < pd.Timestamp("2024-06-01", tz="UTC"),
                start < pd.Timestamp("2024-08-01", tz="UTC"),
                start < pd.Timestamp("2024-09-01", tz="UTC"),
                start < pd.Timestamp("2025-01-01", tz="UTC"),
            ],
            ["fit", "select", "calibrate", "evaluate"],
            default="unlabeled",
        ),
        index=index,
    )
