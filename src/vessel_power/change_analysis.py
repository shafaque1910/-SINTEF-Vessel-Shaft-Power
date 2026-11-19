import argparse
import json
import numpy as np
import pandas as pd
from .configuration import ROOT, project_path
from .storage import write_json, sha256
from .preprocessing import QMAP, clean_records, minute_end


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum(
        1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    )
    return q


def permutation_test(before, after, seed=731, permutations=999):
    """Shuffle whole daily summaries, never minutes; retrospective screen only.

    Daily exchangeability is an approximation. Serial dependence is not removed.
    """
    a = np.asarray(before, float)
    b = np.asarray(after, float)
    z = np.r_[a, b]
    effect = np.mean(b) - np.mean(a)
    rng = np.random.default_rng(seed)
    n = 0
    for _ in range(permutations):
        zz = rng.permutation(z)
        n += abs(zz[len(a) :].mean() - zz[: len(a)].mean()) >= abs(effect) - 1e-12
    return float(effect), (n + 1) / (permutations + 1)


def block_permutation_test(daily, seed=732, permutations=999):
    """28 scheduled days, NaNs retained; permute contiguous three-day blocks.

    Sensitivity to short-range dependence, not a general stationarity guarantee.
    """
    z = np.asarray(daily, float)
    assert len(z) == 28
    effect = np.nanmean(z[14:]) - np.nanmean(z[:14])
    blocks = [z[i : i + 3] for i in range(0, 28, 3)]
    rng = np.random.default_rng(seed)
    n = 0
    valid = 0
    for _ in range(permutations):
        zz = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        if min(np.isfinite(zz[:14]).sum(), np.isfinite(zz[14:]).sum()) < 7:
            continue
        valid += 1
        n += abs(np.nanmean(zz[14:]) - np.nanmean(zz[:14])) >= abs(effect) - 1e-12
    return (n + 1) / (valid + 1)


def matched_scan(d):
    z = d[d.power_count.gt(0) & d[["engine_rpm", "fuel_load", "stw"]].notna().all(axis=1)].copy()
    # No sign-based selection for primary analysis.
    z["day"] = (z.index - pd.Timedelta(minutes=1)).floor("D")
    z["cell"] = (
        np.floor(z.engine_rpm / 5).astype(int).astype(str)
        + "/"
        + np.floor(z.fuel_load / 5).astype(int).astype(str)
        + "/"
        + np.floor(z.stw / 1).astype(int).astype(str)
    )
    z["power_positive"] = z.power.where(z.power_min.gt(0))
    responses = [
        "power",
        "power_positive",
        "torque",
        "shaft_rpm",
        "fuel_flow",
        "thrust",
        "scavenge",
        "turbo",
    ]
    # Means of raw samples -> minute, robust medians within day and operating cell.
    g = z.groupby(["day", "cell"])[responses].agg(["median", "count"])
    tables = {c: g[c]["median"].where(g[c]["count"].ge(10)).unstack("cell") for c in responses}
    limits = {
        "power": 300,
        "power_positive": 300,
        "torque": 30,
        "shaft_rpm": 3,
        "fuel_flow": 100,
        "thrust": 20,
        "scavenge": 0.2,
        "turbo": 500,
    }
    rows = []
    support = []
    for date in pd.date_range("2024-01-15", "2024-12-18", freq="7D", tz="UTC"):
        for response, table in tables.items():
            a = table[(table.index >= date - pd.Timedelta(days=14)) & (table.index < date)]
            b = table[(table.index >= date) & (table.index < date + pd.Timedelta(days=14))]
            cols = table.columns[a.count().ge(3) & b.count().ge(3)]
            r = dict(
                boundary=date,
                response=response,
                window_days=14,
                common_cells=len(cols),
                testable=False,
                n_before_days=0,
                n_after_days=0,
                delta=np.nan,
                p=np.nan,
                block_p=np.nan,
                practical_threshold=limits[response],
            )
            if len(cols):
                reference = pd.concat([a[cols], b[cols]]).median()
                aa = (a[cols] - reference).median(axis=1).dropna()
                bb = (b[cols] - reference).median(axis=1).dropna()
                r.update(n_before_days=len(aa), n_after_days=len(bb))
                if min(len(aa), len(bb)) >= 7:
                    delta, p = permutation_test(aa, bb)
                    schedule = pd.date_range(date - pd.Timedelta(days=14), periods=28, freq="D")
                    series = pd.concat([aa, bb]).reindex(schedule)
                    block_p = block_permutation_test(series)
                    r.update(testable=True, delta=delta, p=p, block_p=block_p)
                for c in cols:
                    support.append(
                        dict(
                            boundary=date,
                            response=response,
                            cell=c,
                            before_days=int(a[c].count()),
                            after_days=int(b[c].count()),
                            before_median=a[c].median(),
                            after_median=b[c].median(),
                        )
                    )
            rows.append(r)
    r = pd.DataFrame(rows)
    r["q"] = np.nan
    r["block_q"] = np.nan
    valid = r.testable
    r.loc[valid, "q"] = bh(r.loc[valid, "p"])
    r.loc[valid, "block_q"] = bh(r.loc[valid, "block_p"])
    r["daily_candidate_change"] = r.q.le(0.05) & r.delta.abs().ge(r.practical_threshold)
    r["candidate_change"] = r.daily_candidate_change & r.block_q.le(0.05)
    return r, pd.DataFrame(support)


SETTINGS = json.loads((ROOT / "config/investigation.json").read_text())
EVENTS = SETTINGS["events"]


def auxiliary_channels(raw_folder, index):
    """Read only missing thrust/fuel-flow channels from original sources."""
    names = {"thrust", "fuel_flow"}
    mapping = {q: n for q, n in QMAP.items() if n in names}
    parts = []
    inventory = []
    for month in [*[str(i) for i in range(1, 13)], "Jan25"]:
        path = raw_folder / f"{month}.csv"
        pieces = []
        rows = 0
        jan = month == "Jan25"
        for chunk in pd.read_csv(
            path,
            header=0 if jan else None,
            names=(
                ["timestamp", "qid", "value", "records"] if jan else ["timestamp", "qid", "value"]
            ),
            usecols=[0, 1, 2],
            dtype=str,
            chunksize=300000,
        ):
            rows += len(chunk)
            c = chunk[chunk.qid.isin(mapping)].copy()
            if not len(c):
                continue
            c["ts"] = pd.to_datetime(
                c.timestamp, format="%d/%m/%Y %H:%M:%S" if jan else "ISO8601", utc=True
            )
            c["channel"] = c.qid.map(mapping)
            c["value"] = pd.to_numeric(c.value)
            pieces.append(c[["ts", "channel", "value"]])
        if pieces:
            raw, conflicts = clean_records(pd.concat(pieces, ignore_index=True))
            if len(conflicts):
                raise ValueError("Review conflicting auxiliary observations")
            raw["end"] = minute_end(raw.ts)
            # Preserve original mean/count/min/max/std aggregation arithmetic.
            grouped = raw.groupby(["end", "channel"], observed=True).value.agg(
                ["mean", "count", "min", "max", "std"]
            )
            wide = grouped.unstack("channel").swaplevel(0, 1, axis=1).sort_index(axis=1)
            for n in names:
                if (n, "mean") not in wide:
                    wide[(n, "mean")] = np.nan
            parts.append(pd.DataFrame({n: wide[(n, "mean")] for n in sorted(names)}))
        inventory.append(dict(file=path.name, rows_scanned=rows, sha256=sha256(path)))
        print("Read change-investigation source", month, "rows", rows, flush=True)
    result = pd.concat(parts).sort_index()
    if result.index.duplicated().any():
        raise ValueError("Auxiliary source minute overlap")
    return result.reindex(index), pd.DataFrame(inventory)


def original_windows(raw_folder):
    """Exact original eight two-hour windows, original numeric parser and QIDs."""
    allraw = []
    read_counts = []
    for month in [6, 7, 8]:
        intervals = [
            (
                pd.Timestamp(t, tz="UTC") - pd.Timedelta(hours=1),
                pd.Timestamp(t, tz="UTC") + pd.Timedelta(hours=1),
            )
            for _, t, m, _ in EVENTS
            if m == month
        ]
        prefixes = set()
        for a, b in intervals:
            prefixes.update(
                pd.date_range(a.floor("min"), b.ceil("min"), freq="min").strftime("%Y-%m-%dT%H:%M")
            )
        chunks = []
        n = 0
        for c in pd.read_csv(
            raw_folder / f"{month}.csv",
            header=None,
            names=["raw_ts", "qid", "value"],
            dtype={"raw_ts": str, "qid": str, "value": float},
            chunksize=300000,
        ):
            n += len(c)
            c = c[c.qid.isin(QMAP) & c.raw_ts.str.slice(0, 16).isin(prefixes)].copy()
            if len(c):
                c["ts"] = pd.to_datetime(c.raw_ts, format="ISO8601", utc=True)
                c["channel"] = c.qid.map(QMAP)
                chunks.append(c)
        raw, conflicts = clean_records(pd.concat(chunks, ignore_index=True))
        if len(conflicts):
            raise ValueError("Conflicting original transition observations")
        allraw.append(raw)
        read_counts.append(dict(file=f"{month}.csv", scanned_rows=n, retained_window_rows=len(raw)))
        print("Original transition windows", month, n, len(raw), flush=True)
    return pd.concat(allraw).sort_values(["channel", "ts"]), pd.DataFrame(read_counts)


def transition_tables(raw):
    """Same recorded transitions, +/-1s same-minute pairing, and pre/post summaries."""
    summary = []
    evidence = []
    verification = []
    clocks = []
    independent = ["engine_rpm", "turbo", "fuel_load", "scavenge", "stw", "sog", "turn"]
    channels = ["power", "shaft_rpm", "torque", "thrust", "fuel_flow", *independent]
    for name, text, month, state in EVENTS:
        anchor = pd.Timestamp(text, tz="UTC")
        power = raw[
            raw.channel.eq("power")
            & raw.ts.between(anchor - pd.Timedelta(minutes=3), anchor + pd.Timedelta(minutes=3))
        ].sort_values("ts")
        if state not in ["anchor", "missing_boundary"]:
            mask = {
                "zero": power.value.eq(0),
                "positive": power.value.gt(0),
                "negative": power.value.lt(0),
            }[state]
            changed = mask & ~mask.shift(1, fill_value=False)
            candidates = power.loc[changed, "ts"]
            verified = (
                candidates.iloc[np.argmin((candidates - anchor).abs())]
                if len(candidates)
                else pd.NaT
            )
        else:
            verified = anchor
        if pd.isna(verified):
            raise ValueError("No raw transition supported near " + name)
        verification.append(
            dict(
                event=name,
                coarse_anchor=anchor,
                verified_raw_transition=(
                    verified if state not in ["anchor", "missing_boundary"] else pd.NaT
                ),
                analysis_anchor=verified,
                recorded_state_after=state,
                interpretation=(
                    "end of observed zero run followed by missing power, not verified recovery"
                    if state == "missing_boundary"
                    else (
                        "coarse weekly comparison anchor, no exact transition asserted"
                        if state == "anchor"
                        else "verified recorded-channel transition only; no claim of physical state or modification date"
                    )
                ),
            )
        )
        for c in channels:
            z = raw[raw.channel.eq(c)].sort_values("ts")
            a = z[
                z.ts.ge(verified - pd.Timedelta(minutes=10))
                & z.ts.lt(verified - pd.Timedelta(minutes=1))
            ].value
            b = z[
                z.ts.ge(verified + pd.Timedelta(minutes=1))
                & z.ts.lt(verified + pd.Timedelta(minutes=10))
            ].value
            evidence.append(
                dict(
                    event=name,
                    channel=c,
                    source="torquemeter" if c not in independent else "separately_registered",
                    pre_samples=len(a),
                    post_samples=len(b),
                    pre_median=a.median(),
                    post_median=b.median(),
                    delta=b.median() - a.median(),
                    pre_min=a.min(),
                    pre_max=a.max(),
                    post_min=b.min(),
                    post_max=b.max(),
                )
            )
        p = (
            raw[
                raw.channel.eq("power")
                & raw.ts.ge(verified - pd.Timedelta(minutes=10))
                & raw.ts.lt(verified + pd.Timedelta(minutes=10))
            ][["ts", "value"]]
            .rename(columns={"value": "power"})
            .sort_values("ts")
        )
        for c in independent:
            z = (
                raw[raw.channel.eq(c)][["ts", "value"]]
                .rename(columns={"ts": c + "_ts", "value": c})
                .sort_values(c + "_ts")
            )
            p = pd.merge_asof(
                p,
                z,
                left_on="ts",
                right_on=c + "_ts",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=1),
            )
            same = minute_end(p.ts).eq(minute_end(p[c + "_ts"]))
            p.loc[~same, c] = np.nan
            clocks.append(
                dict(
                    event=name,
                    channel=c,
                    power_ticks=len(p),
                    paired_ticks=int(p[c].notna().sum()),
                    max_absolute_offset_seconds=float(
                        (p[c + "_ts"] - p.ts).dt.total_seconds().abs().max()
                    ),
                )
            )
        p["engine_active"] = (
            p.engine_rpm.abs().gt(20) & p.turbo.gt(1000) & p.fuel_load.gt(5) & p.scavenge.gt(0)
        )
        after = p[p.ts.ge(verified)]
        summary.append(
            dict(
                event=name,
                raw_transition=verified,
                after_ticks=len(after),
                engine_active_after_ticks=int(after.engine_active.sum()),
                zero_power_after_ticks=int(after.power.eq(0).sum()),
                negative_power_after_ticks=int(after.power.lt(0).sum()),
                sog_after_median=after.sog.median(),
                stw_after_median=after.stw.median(),
            )
        )
    return (
        pd.DataFrame(verification),
        pd.DataFrame(evidence),
        pd.DataFrame(summary),
        pd.DataFrame(clocks),
    )


def compare_table(actual, expected):
    """Compare serialized numerical evidence; report tolerances and exact counts."""
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        return {"passed": False, "shape_or_columns_differ": True}
    differences = {}
    passed = True
    for c in actual.columns:
        a = actual[c]
        b = expected[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            aa = a.to_numpy(float)
            bb = b.to_numpy(float)
            good = np.allclose(aa, bb, rtol=1e-12, atol=1e-9, equal_nan=True)
            finite = np.isfinite(aa) & np.isfinite(bb)
            differences[c] = float(np.max(abs(aa[finite] - bb[finite]))) if finite.any() else 0.0
        else:
            good = a.fillna("<missing>").astype(str).equals(b.fillna("<missing>").astype(str))
        passed &= bool(good)
    return {
        "passed": passed,
        "rows": len(actual),
        "maximum_numeric_differences": differences,
        "rtol": 1e-12,
        "atol": 1e-9,
    }


def run(processed, raw_folder, out):
    if out.exists():
        raise FileExistsError("Investigation output exists; select a fresh --output")
    out.mkdir(parents=True)
    protected = {
        p.name: sha256(p)
        for p in [
            ROOT / "results/estimates.pkl",
            ROOT / "results/metrics.csv",
            ROOT / "results/model.json",
            ROOT / "results/update_history.pkl",
        ]
    }
    write_json(SETTINGS, out / "frozen_definition.json")
    d = pd.read_pickle(processed)
    from .storage import frame_fingerprint

    expected = json.loads((ROOT / "config/expected_results.json").read_text())
    assert frame_fingerprint(d) == expected["frame_fingerprints"]["measurements"]
    aux, inventory = auxiliary_channels(raw_folder, d.index)
    inventory.to_csv(out / "source_inventory.csv", index=False)
    assert dict(zip(inventory.file, inventory.sha256)) == expected["raw_inputs"]
    all_data = d.join(aux)
    scan, cells = matched_scan(all_data[(all_data.index - pd.Timedelta(minutes=1)).year == 2024])
    scan.to_csv(out / "relationship_tests.csv", index=False)
    cells.to_csv(out / "common_conditions.csv", index=False)
    scan[scan.candidate_change | scan.response.eq("power") & scan.daily_candidate_change].to_csv(
        out / "relationship_changes.csv", index=False
    )
    raw, window_inventory = original_windows(raw_folder)
    window_inventory.to_csv(out / "transition_sources.csv", index=False)
    transitions, evidence, concurrent, alignment = transition_tables(raw)
    tables = {
        "transitions": transitions,
        "transition_evidence": evidence,
        "concurrent": concurrent,
        "alignment": alignment,
    }
    for name, table in tables.items():
        table.to_csv(out / (name + ".csv"), index=False)
    reference = json.loads((ROOT / "config/investigation_reference.json").read_text())
    checks = {}
    checks["all_relationship_tests"] = compare_table(
        pd.read_csv(out / "relationship_tests.csv"), pd.DataFrame(reference["scan"])
    )
    for name in ["transitions", "concurrent", "alignment"]:
        checks[name] = compare_table(
            pd.read_csv(out / (name + ".csv")), pd.DataFrame(reference[name])
        )
    checks["retained_changes"] = compare_table(
        pd.read_csv(out / "relationship_changes.csv"),
        pd.read_csv(ROOT / "results/evidence/relationship_changes.csv"),
    )
    checks["transition_sensor_summaries"] = compare_table(
        pd.read_csv(out / "transition_evidence.csv"),
        pd.read_csv(ROOT / "results/evidence/transition_evidence.csv"),
    )
    assert protected == {name: sha256(ROOT / "results" / name) for name in protected}
    result = {
        "completed": True,
        "passed": all(c["passed"] for c in checks.values()),
        "parity": checks,
        "scheduled_tests": len(scan),
        "testable_tests": int(scan.testable.sum()),
        "dependence_sensitive_candidates": int(scan.candidate_change.sum()),
        "power_candidates_passing_both_screens": int(
            (scan.candidate_change & scan.response.eq("power")).sum()
        ),
        "source_rows_scanned": int(inventory.rows_scanned.sum()),
        "transition_source_rows_scanned": int(window_inventory.scanned_rows.sum()),
        "retained_transition_rows": len(raw),
        "final_artifacts_unchanged": True,
        "archive_runtime_access": "blocked",
        "interpretation_unchanged": SETTINGS["interpretation"],
        "label_policy": "Final v2 labels remain unchanged; measurement investigations intentionally include suspect/unresolved raw readings rather than selecting reliable labels.",
    }
    write_json(result, out / "verification.json")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError(
            "Investigation evidence differs: inspect saved verification; do not change thresholds"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed/measurements.pkl")
    parser.add_argument("--raw-data", default="data/raw")
    parser.add_argument("--output", default="results/change_analysis")
    args = parser.parse_args(argv)
    from .verification import deny_archive_access
    import sys

    sys.addaudithook(deny_archive_access)
    run(project_path(args.processed), project_path(args.raw_data), project_path(args.output))
