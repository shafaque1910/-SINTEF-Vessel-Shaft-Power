import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .configuration import ROOT, PARAMETERS, project_path
from .reliability import features, label_eligibility, inference_gate, split_of
from .storage import write_json, sha256, frame_fingerprint

DEFINITIONS = json.loads((ROOT / "config/comparison.json").read_text())


class SimpleBaseline:
    """Original mean and nonnegative single-coefficient cubic definitions."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, x, y):
        if self.kind == "mean":
            self.value = float(np.mean(y))
        else:
            v = (
                np.maximum(
                    0, x["stw"] if self.kind == "speed_cubic" else x["engine_rpm_abs"]
                ).to_numpy()
                ** 3
            )
            self.value = float(max(0, np.dot(v, y) / np.dot(v, v))) if np.dot(v, v) > 0 else 0.0
        return self

    def predict(self, x):
        if self.kind == "mean":
            return np.full(len(x), self.value)
        return (
            self.value
            * np.maximum(
                0, x["stw"] if self.kind == "speed_cubic" else x["engine_rpm_abs"]
            ).to_numpy()
            ** 3
        )


def make_model(name):
    if name in ["mean", "speed_cubic", "rpm_cubic"]:
        return SimpleBaseline(name)
    if name.startswith("ridge"):
        return make_pipeline(StandardScaler(), Ridge(alpha=DEFINITIONS["models"][name]["alpha"]))
    return HistGradientBoostingRegressor(**DEFINITIONS["models"][name], **DEFINITIONS["hgb_common"])


def fit_state(model):
    if isinstance(model, SimpleBaseline):
        return {"kind": model.kind, "value": model.value}
    if hasattr(model, "named_steps"):
        scale = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        return {
            "alpha": ridge.alpha,
            "scaler_mean": scale.mean_.tolist(),
            "scaler_scale": scale.scale_.tolist(),
            "coefficients": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
        }
    h = hashlib.sha256()
    for stage in model._predictors:
        for tree in stage:
            h.update(tree.nodes.tobytes())
    return {
        "parameters": model.get_params(),
        "iterations": model.n_iter_,
        "initial_prediction": model._baseline_prediction.tolist(),
        "tree_sha256": h.hexdigest(),
    }


def selection_scores(z, names):
    """Equal-regime mean daily MAE in June-July; never uses later outcomes."""
    rows = []
    for name in names:
        v = z[z.eval_label_eligible & z.served & z.split.eq("select")].copy()
        v["ae"] = (v[name] - v.power).abs()
        v["day"] = (v.index - pd.Timedelta(minutes=1)).floor("D")
        daily = v.groupby(["inference_regime", "day"]).ae.mean()
        reg = daily.groupby("inference_regime").mean()
        if not {"active", "idle"}.issubset(reg.index):
            raise ValueError("Insufficient selection regimes")
        rows.append(
            dict(
                model=name,
                selection_score=float(reg.mean()),
                active_daily_mae=float(reg["active"]),
                idle_daily_mae=float(reg["idle"]),
                selection_days=len(daily),
            )
        )
    return pd.DataFrame(rows).sort_values(["selection_score", "model"])


def diagnostic_metrics(y, p):
    """Original independent error decomposition used by the retained tables."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if not len(y):
        return {"n": 0}
    e = p - y
    bias = e.mean()
    mse = np.mean(e * e)
    vy = np.mean((y - y.mean()) ** 2)
    vp = np.mean((p - p.mean()) ** 2)
    return dict(
        n=len(y),
        mae=float(np.mean(abs(e))),
        rmse=float(np.sqrt(mse)),
        bias=float(bias),
        pearson_r=(
            float(np.mean((y - y.mean()) * (p - p.mean())) / np.sqrt(vy * vp))
            if vy > 0 and vp > 0
            else np.nan
        ),
        r_squared=float(1 - mse / vy) if vy > 0 else np.nan,
        overprediction_fraction=float(np.mean(e > 0)),
        mean_recorded_kw=float(y.mean()),
        mean_prediction_kw=float(p.mean()),
        recorded_std_kw=float(np.sqrt(vy)),
        centered_error_rmse_kw=float(np.sqrt(np.mean((e - bias) ** 2))),
        mse_fraction_from_bias=float(bias * bias / mse) if mse > 0 else np.nan,
    )


def run(processed, out):
    if out.exists():
        raise FileExistsError("Comparison output exists; select a fresh --output")
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
    write_json(DEFINITIONS, out / "frozen_definition.json")
    d = pd.read_pickle(processed)
    expected = json.loads((ROOT / "config/expected_results.json").read_text())
    assert frame_fingerprint(d) == expected["frame_fingerprints"]["measurements"]
    z = label_eligibility(d)
    z["split"] = split_of(d.index)
    z["month"] = (d.index - pd.Timedelta(minutes=1)).strftime("%Y-%m")
    x = features(d)
    z["power"] = d.power
    z["served"] = inference_gate(
        x, PARAMETERS["domain"], z.independent_complete, z.inference_regime
    )
    train = z.train_label_eligible & z.split.eq("fit")
    names = ["mean", "speed_cubic", "rpm_cubic", "ridge_1", "ridge_100", "hgb_15", "hgb_31"]
    states = {}
    for name in names:
        print("Fit frozen candidate", name, "v2 labels", int(train.sum()), flush=True)
        model = make_model(name)
        with threadpool_limits(limits=DEFINITIONS["threads"]):
            model.fit(x[train], z.loc[train, "power"])
            z[name] = np.nan
            z.loc[z.independent_complete, name] = np.maximum(
                0, model.predict(x[z.independent_complete])
            )
        states[name] = fit_state(model)
    selection = selection_scores(z, names)
    selection.to_csv(out / "selection.csv", index=False)
    rows = []
    for split in ["select", "calibrate", "evaluate"]:
        h = z[z.eval_label_eligible & z.served & z.split.eq(split)]
        groups = [("all", "all", h), *[("all", reg, g) for reg, g in h.groupby("inference_regime")]]
        for month, g in h.groupby("month"):
            groups += [
                (month, "all", g),
                *[(month, reg, v) for reg, v in g.groupby("inference_regime")],
            ]
        for month, reg, g in groups:
            for name in names:
                rows.append(
                    dict(
                        split=split,
                        month=month,
                        regime=reg,
                        method=name,
                        **diagnostic_metrics(g.power, g[name])
                    )
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    write_json(states, out / "fit_states.json")
    reference = json.loads((ROOT / "config/comparison_reference.json").read_text())
    coefficients = []
    for name in names:
        old = reference["models"][name]
        new = states[name]
        if "value" in new:
            coefficients.append(
                dict(
                    model=name,
                    field="value",
                    before=old["value"],
                    after=new["value"],
                    difference=new["value"] - old["value"],
                )
            )
        elif "coefficients" in new:
            for field in ["coefficients", "scaler_mean", "scaler_scale", "intercept"]:
                a = np.asarray(old[field])
                b = np.asarray(new[field])
                coefficients.append(
                    dict(
                        model=name,
                        field=field,
                        maximum_absolute_difference=float(np.max(abs(a - b))),
                    )
                )
        else:
            assert new["parameters"] == old["parameters"]
            coefficients.append(
                dict(
                    model=name, field="tree_sha256", equal=old["tree_sha256"] == new["tree_sha256"]
                )
            )
    pd.DataFrame(coefficients).to_csv(out / "fit_state_comparison.csv", index=False)
    old_selection = pd.read_csv(ROOT / "results/evidence/method_selection.csv")
    merged = selection.merge(old_selection, on="model", suffixes=("_v2_refit", "_historical"))
    merged["score_difference"] = merged.selection_score_v2_refit - merged.selection_score_historical
    merged.to_csv(out / "selection_parity.csv", index=False)
    old_metrics = pd.read_csv(ROOT / "results/evidence/candidate_validation.csv")
    eval_current = metrics[
        metrics.split.eq("evaluate")
        & metrics.month.eq("all")
        & metrics.regime.isin(["all", "active"])
    ]
    compare = eval_current.merge(
        old_metrics,
        on=["split", "month", "regime", "method"],
        suffixes=("_v2_refit", "_historical"),
    )
    for field in ["mae", "rmse", "bias", "r_squared"]:
        compare[field + "_difference"] = (
            compare[field + "_v2_refit"] - compare[field + "_historical"]
        )
    compare.to_csv(out / "evaluation_parity.csv", index=False)
    assert protected == {name: sha256(ROOT / "results" / name) for name in protected}
    final_a = json.loads((ROOT / "results/model.json").read_text())["coefficient"]
    assert np.isclose(states["rpm_cubic"]["value"], final_a, rtol=1e-12, atol=1e-15)
    write_json(
        {
            "completed": True,
            "training_labels": int(train.sum()),
            "historical_training_labels": reference["source_fit_labels"],
            "unresolved_training_labels_not_reintroduced": reference["source_fit_labels"]
            - int(train.sum()),
            "diagnostic_ranking_first": selection.iloc[0].model,
            "final_selected_model": PARAMETERS["selected_family"],
            "final_model_reselected": False,
            "final_artifacts_unchanged": True,
            "historical_scores_exact": bool(
                np.allclose(merged.score_difference, 0, rtol=0, atol=1e-9)
            ),
            "non_parity_reason": DEFINITIONS["reference_meaning"],
            "archive_runtime_access": "blocked",
            "hyperparameters_unchanged": True,
            "rpm_coefficient_difference_vs_final": states["rpm_cubic"]["value"] - final_a,
            "coefficient_parity_rtol": 1e-12,
            "physical_accuracy_validated": False,
        },
        out / "verification.json",
    )
    print(selection[["model", "selection_score"]].to_string(index=False), flush=True)
    print(
        compare[compare.regime.eq("active")][
            ["method", "mae_historical", "mae_v2_refit", "mae_difference"]
        ].to_string(index=False),
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="data/processed/measurements.pkl")
    parser.add_argument("--output", default="results/model_comparison")
    args = parser.parse_args(argv)
    from .verification import deny_archive_access
    import sys

    sys.addaudithook(deny_archive_access)
    run(project_path(args.processed), project_path(args.output))
