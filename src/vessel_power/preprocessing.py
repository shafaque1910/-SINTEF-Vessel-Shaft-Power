import numpy as np
import pandas as pd
from .configuration import CHANNELS
from .storage import sha256

QMAP = {q: n for q, n, *_ in CHANNELS}
NAMES = list(QMAP.values())


def minute_end(t):
    return t.dt.floor("min") + pd.Timedelta(minutes=1)


def identity_kw(torque, rpm, torque_factor=1):
    return torque * rpm * (2 * np.pi / 60000) * torque_factor


def exceeds(a, b, rel=0.05, absolute=100):
    return (a - b).abs() > np.maximum(absolute, rel * np.maximum(a.abs(), b.abs()))


def clean_records(d):
    """Distinct copies only; never choose one conflicting sensor value."""
    d = d.drop_duplicates(["ts", "channel", "value"]).copy()
    key = d.groupby(["ts", "channel"], observed=True).value.agg(["size", "nunique"])
    bad = key[key["size"] > 1].reset_index()
    if len(bad):
        keys = pd.MultiIndex.from_frame(bad[["ts", "channel"]])
        mask = pd.MultiIndex.from_frame(d[["ts", "channel"]]).isin(keys)
        conflicts = d[mask].copy()
        d = d[~mask]
    else:
        conflicts = d.iloc[:0].copy()
    return d, conflicts


def paired_raw(d, tolerance=1):
    """Near-simultaneous samples, with timestamps and same-minute restriction."""
    base = (
        d[d.channel.eq("power")][["ts", "value"]]
        .rename(columns={"value": "power"})
        .sort_values("ts")
    )
    base["end"] = minute_end(base.ts)
    for c in ["torque", "shaft_rpm", "energy", "engine_rpm"]:
        src = (
            d[d.channel.eq(c)][["ts", "value"]]
            .rename(columns={"ts": c + "_ts", "value": c})
            .sort_values(c + "_ts")
        )
        base = pd.merge_asof(
            base,
            src,
            left_on="ts",
            right_on=c + "_ts",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tolerance),
        )
        valid = minute_end(base[c + "_ts"]).eq(base.end)
        base.loc[~valid, [c, c + "_ts"]] = [np.nan, pd.NaT]
        base[c + "_offset_s"] = (base[c + "_ts"] - base.ts).dt.total_seconds()
    base["registered_kw"] = identity_kw(base.torque, base.shaft_rpm)
    base["scaled_kw"] = identity_kw(base.torque, base.shaft_rpm, 1000)
    base["paired"] = base.torque.notna() & base.shaft_rpm.notna()
    base["registered_bad"] = exceeds(base.power, base.registered_kw) & base.paired
    base["scaled_bad"] = exceeds(base.power, base.scaled_kw) & base.paired
    base["magnitude_bad"] = exceeds(base.power.abs(), base.scaled_kw.abs()) & base.paired
    base["ratio"] = base.power / base.registered_kw.where(base.registered_kw.abs().gt(0.1))
    # Inter-channel span can be up to 2*tolerance; document, do not claim exact simultaneity.
    stamps = pd.concat([base.ts, base.torque_ts, base.shaft_rpm_ts], axis=1)
    base["pair_span_s"] = (
        (stamps.max(axis=1) - stamps.min(axis=1)).dt.total_seconds().where(base.paired)
    )
    return base


def read_month(path):
    """Read one monthly source in chunks; retain original aggregation arithmetic."""
    jan25 = path.stem == "Jan25"
    pieces = []
    rows = 0
    malformed = 0
    reader = pd.read_csv(
        path,
        header=0 if jan25 else None,
        names=["timestamp", "qid", "value", "records"] if jan25 else ["timestamp", "qid", "value"],
        usecols=[0, 1, 2],
        dtype=str,
        chunksize=300000,
    )
    for chunk in reader:
        rows += len(chunk)
        c = chunk[chunk.qid.isin(QMAP)].copy()
        c["channel"] = c.qid.map(QMAP)
        c["ts"] = pd.to_datetime(
            c.timestamp,
            format="%d/%m/%Y %H:%M:%S" if jan25 else "ISO8601",
            errors="coerce",
            utc=True,
        )
        c["value"] = pd.to_numeric(c.value, errors="coerce")
        valid = c.ts.notna() & np.isfinite(c.value)
        malformed += int((~valid).sum())
        pieces.append(c.loc[valid, ["ts", "channel", "value"]])
    records = pd.concat(pieces, ignore_index=True)
    finite = len(records)
    distinct = len(records.drop_duplicates(["ts", "channel", "value"]))
    records, conflicts = clean_records(records)
    records["end"] = minute_end(records.ts)
    g = records.groupby(["end", "channel"], observed=True).value.agg(
        ["mean", "count", "min", "max", "std"]
    )
    wide = g.unstack("channel").swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns = [c if stat == "mean" else c + "_" + stat for c, stat in wide.columns]
    for c in NAMES:
        for suffix in ["", "_count", "_min", "_max", "_std"]:
            if c + suffix not in wide:
                wide[c + suffix] = np.nan
        wide[c + "_count"] = wide[c + "_count"].fillna(0)
    conflict_counts = (
        conflicts.assign(end=minute_end(conflicts.ts))
        .drop_duplicates(["ts", "channel"])
        .groupby("end")
        .size()
    )
    wide["conflicting_keys"] = conflict_counts.reindex(wide.index, fill_value=0)
    pairs = paired_raw(records)
    summary = pairs.groupby("end").agg(
        paired_count=("paired", "sum"),
        registered_bad=("registered_bad", "sum"),
        scaled_bad=("scaled_bad", "sum"),
        magnitude_bad=("magnitude_bad", "sum"),
    )
    wide = wide.join(summary)
    wide["source_file"] = path.name
    return primitive_measurements(wide), dict(
        file=path.name,
        raw_rows=rows,
        selected_finite_rows=finite,
        exact_duplicate_copies=finite - distinct,
        conflicting_keys=len(conflicts.drop_duplicates(["ts", "channel"])),
        malformed_selected_rows=malformed,
        sha256=sha256(path),
    )


def primitive_measurements(d):
    from .reliability import SOURCES

    cols = [
        c + s
        for c in ["power", "torque", "shaft_rpm", *SOURCES]
        for s in ["", "_count", "_min", "_max"]
    ]
    cols += [
        "registered_bad",
        "scaled_bad",
        "magnitude_bad",
        "paired_count",
        "conflicting_keys",
        "source_file",
    ]
    return d[cols].copy()


def extract_measurements(folder):
    """Rebuild all thirteen original sources; no historical cache is consumed."""
    parts = []
    inventory = []
    for name in [*[str(i) for i in range(1, 13)], "Jan25"]:
        print("Reading original source", name, flush=True)
        frame, entry = read_month(folder / f"{name}.csv")
        parts.append(frame)
        inventory.append(entry)
    return align_months(parts), pd.DataFrame(inventory)


def align_months(parts):
    """Expose missing completed minutes on an explicit UTC grid without filling values."""
    d = pd.concat(parts).sort_index()
    if d.index.duplicated().any():
        raise ValueError("Cross-file timestamp overlap requires explicit source review")
    d = d.reindex(
        pd.date_range(pd.Timestamp("2024-01-01 00:01", tz="UTC"), d.index.max(), freq="min")
    )
    d.index.name = "end"
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
        d[c + "_count"] = d[c + "_count"].fillna(0)
    d["conflicting_keys"] = d.conflicting_keys.fillna(0)
    return d
