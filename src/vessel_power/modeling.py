import hashlib
import numpy as np
import pandas as pd
from .configuration import PARAMETERS as CHOICES
from .reliability import POLICY_VERSION

METHODS = CHOICES["methods"]
FIRST = pd.Timestamp(CHOICES["first_update"])
LAST = pd.Timestamp(CHOICES["last_update"])
DELAY = pd.Timedelta(hours=CHOICES["shaft_label_delay_hours"])
HISTORY = pd.Timedelta(days=CHOICES["history_days"])


def coefficient(rpm, y, fallback=0.0):
    v = np.asarray(rpm, float) ** 3
    y = np.asarray(y, float)
    if not np.isfinite(v).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite fit inputs")
    den = float(v @ v)
    return max(0.0, float((v @ y) / den)) if den > 0 else fallback, den


def score(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Scored rows must have finite labels and predictions")
    if not len(y):
        return {"n": 0}
    e = p - y
    vy = np.mean((y - y.mean()) ** 2)
    vp = np.mean((p - p.mean()) ** 2)
    return {
        "n": len(y),
        "mae": float(abs(e).mean()),
        "rmse": float(np.sqrt((e * e).mean())),
        "bias": float(e.mean()),
        "r_squared": float(1 - np.mean(e * e) / vy) if vy > 0 else np.nan,
        "pearson_r": (
            float(np.mean((y - y.mean()) * (p - p.mean())) / np.sqrt(vy * vp))
            if vy > 0 and vp > 0
            else np.nan
        ),
    }


def historical_members(z, day):
    cut = day - DELAY
    return z.loc[
        (z.index > cut - HISTORY)
        & (z.index <= cut)
        & z.train_label_eligible
        & z.served
        & z.power.notna()
    ]


def updates(z, x, static_a, days=None):
    """One shared v2 membership set feeds both fixed adaptive methods."""
    days = pd.date_range(FIRST, LAST, freq="D") if days is None else days
    offset_prediction = pd.Series(np.nan, index=z.index)
    rolling_prediction = pd.Series(np.nan, index=z.index)
    ledger = []
    for day in days:
        h = historical_members(z, day)
        covered = (h.index - pd.Timedelta(minutes=1)).floor("D").nunique()
        enough = len(h) >= CHOICES["minimum_labels"] and covered >= CHOICES["minimum_covered_days"]
        a, den = coefficient(x.loc[h.index, "engine_rpm_abs"], h.power, static_a)
        updated = enough and den > 0
        if not updated:
            a = static_a
        offset = (
            float((h.power - static_a * x.loc[h.index, "engine_rpm_abs"] ** 3).median())
            if enough
            else 0.0
        )
        apply = (z.index > day) & (z.index <= day + pd.Timedelta(days=1)) & z.served
        base = static_a * x.loc[apply, "engine_rpm_abs"] ** 3
        offset_prediction.loc[apply] = np.maximum(0, base + offset)
        rolling_prediction.loc[apply] = a * x.loc[apply, "engine_rpm_abs"] ** 3
        ledger.append(
            {
                "update_at": day,
                "cutoff": day - DELAY,
                "history_first": h.index.min(),
                "history_last": h.index.max(),
                "labels": len(h),
                "covered_days": covered,
                "member_sha256": hashlib.sha256(h.index.asi8.tobytes()).hexdigest(),
                "active_labels": int(h.inference_regime.eq("active").sum()),
                "idle_labels": int(h.inference_regime.eq("idle").sum()),
                "offset_kw": offset,
                "offset_updated": enough,
                "coefficient": a,
                "rolling_updated": updated,
                "denominator": den,
                "policy_version": POLICY_VERSION,
                "assumed_arrival_violations": int(((h.index + DELAY) > day).sum()),
                "physical_trust_verified": False,
            }
        )
    return offset_prediction, rolling_prediction, pd.DataFrame(ledger)


def metric_table(z):
    rows = []
    for split, h in z[z.proxy_error_allowed].groupby("split"):
        groups = [("all", "all", h), *[("all", reg, g) for reg, g in h.groupby("inference_regime")]]
        for month, g in h.groupby("month"):
            groups += [
                (month, "all", g),
                *[(month, reg, v) for reg, v in g.groupby("inference_regime")],
            ]
        for month, reg, g in groups:
            for method in METHODS:
                rows.append(
                    dict(
                        split=split,
                        month=month,
                        regime=reg,
                        method=method,
                        policy_version=POLICY_VERSION,
                        **score(g.power, g[method])
                    )
                )
    result = pd.DataFrame(rows)
    base = result[result.method.eq(METHODS[0])][["split", "month", "regime", "mae"]].rename(
        columns={"mae": "static_mae"}
    )
    result = result.merge(base, on=["split", "month", "regime"])
    result["mae_improvement_pct"] = 100 * (
        1 - result.mae / result.static_mae.where(result.static_mae.gt(0))
    )
    return result
