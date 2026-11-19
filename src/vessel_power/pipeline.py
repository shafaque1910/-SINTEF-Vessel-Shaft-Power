from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from .configuration import ROOT, PARAMETERS, project_path
from .preprocessing import extract_measurements
from .reliability import (
    POLICY_VERSION,
    assess_reliability,
    label_eligibility,
    features,
    inference_gate,
    split_of,
)
from .modeling import METHODS, FIRST, DELAY, coefficient, updates, metric_table
from .storage import write_json, sha256, frame_fingerprint


def calculate(d):
    """Unchanged accepted scientific path, independent of repository provenance."""
    flags = assess_reliability(d)
    labels = label_eligibility(d)
    labels["split"] = split_of(d.index)
    labels["month"] = (d.index - pd.Timedelta(minutes=1)).strftime("%Y-%m")
    x = features(d)
    z = labels.copy()
    z["power"] = d.power
    z["served"] = inference_gate(
        x, PARAMETERS["domain"], labels.independent_complete, labels.inference_regime
    )
    z["comparison_served"] = z.served & (z.index > FIRST)
    train = z.train_label_eligible & z.split.eq("fit")
    a, den = coefficient(x.loc[train, "engine_rpm_abs"], z.loc[train, "power"])
    fit_last = z.index[train].max()
    selector_last = z.index[z.eval_label_eligible & z.served & z.split.eq("select")].max()
    assert fit_last + DELAY <= FIRST and selector_last + DELAY <= FIRST
    z[METHODS[0]] = (a * x.engine_rpm_abs**3).where(z.comparison_served)
    z[METHODS[1]], z[METHODS[2]], ledger = updates(z, x, a)
    z["proxy_error_allowed"] = z.eval_label_eligible & z.comparison_served
    z["previous_status"] = flags.status
    z["zero_with_engine_activity"] = flags.zero_with_engine_activity
    z["physical_error_available"] = False
    z["physical_interval_guaranteed"] = False
    z["interpretation"] = np.select(
        [~z.comparison_served, ~z.eval_label_eligible],
        [
            "abstention_or_before_initialization",
            "conditional estimate; unresolved or suspect recorded measurement; physical error unknown",
        ],
        default="conditional proxy-label score; independently calibrated physical accuracy unavailable",
    )
    assert z.loc[z.comparison_served, METHODS].notna().all().all()
    bands = []
    for reg in ["active", "idle"]:
        h = z[
            z.eval_label_eligible & z.served & z.split.eq("calibrate") & z.inference_regime.eq(reg)
        ]
        bands.append(
            dict(
                regime=reg,
                n=len(h),
                radius_kw=float(
                    (a * x.loc[h.index, "engine_rpm_abs"] ** 3 - h.power)
                    .abs()
                    .quantile(0.9, interpolation="higher")
                ),
                hypothetical_labels_available_at=h.index.max() + DELAY,
                physical_coverage_guaranteed=False,
            )
        )
    state = {
        "policy_version": POLICY_VERSION,
        "coefficient": a,
        "training_labels": int(train.sum()),
        "training_members_sha256": hashlib.sha256(z.index[train].asi8.tobytes()).hexdigest(),
        "training_first": z.index[train].min(),
        "training_last": fit_last,
        "selector_last": selector_last,
        "first_update": FIRST,
        "denominator": den,
        "model_families": ["rpm_cubic"],
        "new_selection": False,
        "tuning_performed": False,
        "fixed_methods": METHODS,
    }
    reliability = flags[
        [
            "status",
            "reason_codes",
            "power_sign",
            "near_zero_with_engine_activity",
            "zero_with_engine_activity",
        ]
    ]
    return labels, reliability, x, z, ledger, metric_table(z), pd.DataFrame(bands), state


def run(raw, output, processed):
    if (output / "metrics.csv").exists():
        raise FileExistsError("Existing results are protected; select a fresh --output")
    output.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)
    d, inventory = extract_measurements(raw)
    d.to_pickle(processed)
    inventory.to_csv(output / "input_inventory.csv", index=False)
    labels, reliability, x, z, ledger, metrics, bands, state = calculate(d)
    z.to_pickle(output / "estimates.pkl")
    ledger.to_pickle(output / "update_history.pkl")
    metrics.to_csv(output / "metrics.csv", index=False)
    bands.to_csv(output / "uncertainty.csv", index=False)
    z.groupby(["split", "month", "inference_regime"]).agg(
        minutes=("served", "size"),
        eligible=("eval_label_eligible", "sum"),
        served=("served", "sum"),
        comparison_served=("comparison_served", "sum"),
        scored=("proxy_error_allowed", "sum"),
    ).to_csv(output / "coverage.csv")
    labels.groupby("split")[["train_label_eligible", "eval_label_eligible"]].sum().to_csv(
        output / "label_counts.csv"
    )
    reliability.groupby(["power_sign", "status"]).size().rename("minutes").to_csv(
        output / "reliability_summary.csv"
    )
    write_json(state, output / "model.json")
    hashes = {
        name: frame_fingerprint(frame)
        for name, frame in [
            ("measurements", d),
            ("labels", labels),
            ("reliability", reliability),
            ("features", x),
            ("predictions", z),
            ("update_history", ledger),
        ]
    }
    write_json(
        {
            "policy_version": POLICY_VERSION,
            "metadata_revision": "primary_reason_priority_v1",
            "raw_rows": int(inventory.raw_rows.sum()),
            "minutes": len(d),
            "frame_fingerprints": hashes,
            "raw_input_folder": (
                raw.relative_to(ROOT).as_posix()
                if raw.is_relative_to(ROOT)
                else "authorized_external_input"
            ),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "physical_accuracy_validated": False,
        },
        output / "provenance.json",
    )
    from .plotting import create_figures

    create_figures(z, metrics, output / "figures")
    active = metrics[
        metrics.split.eq("evaluate") & metrics.month.eq("all") & metrics.regime.eq("active")
    ]
    print(
        active[["method", "n", "mae", "rmse", "bias", "r_squared"]].to_string(index=False),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", default="data/raw")
    parser.add_argument("--output", default="results")
    parser.add_argument("--processed", default="data/processed/measurements.pkl")
    args = parser.parse_args()
    run(project_path(args.raw_data), project_path(args.output), project_path(args.processed))


if __name__ == "__main__":
    main()
