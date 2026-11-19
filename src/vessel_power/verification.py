from __future__ import annotations
import argparse
import ast
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from .configuration import ROOT, PARAMETERS, project_path
from .storage import sha256, write_json, frame_fingerprint
from .pipeline import calculate


def deny_archive_access(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes)):
        return
    try:
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
    except (OSError, ValueError):
        return
    if path.is_relative_to(ROOT / "archive"):
        raise PermissionError("Archive access is forbidden in the final workflow")


def scientific_checks(output, processed):
    expected = json.loads((ROOT / "config/expected_results.json").read_text())
    d = pd.read_pickle(processed)
    labels, reliability, x, z, ledger, metrics, bands, state = calculate(d)
    # Value/index/dtype checks are independent of artifact filenames and pickle bytes.
    objects = {
        "measurements": d,
        "labels": labels,
        "reliability": reliability,
        "features": x,
        "predictions": z,
        "update_history": ledger,
    }
    checks = {
        name: frame_fingerprint(frame) == expected["frame_fingerprints"][name]
        for name, frame in objects.items()
    }
    stored = pd.read_pickle(output / "estimates.pkl")
    history = pd.read_pickle(output / "update_history.pkl")
    checks["saved_estimates_exact"] = (
        frame_fingerprint(stored) == expected["frame_fingerprints"]["predictions"]
    )
    checks["saved_update_history_exact"] = (
        frame_fingerprint(history) == expected["frame_fingerprints"]["update_history"]
    )
    for filename, key in [
        ("metrics.csv", "metrics_fingerprint"),
        ("coverage.csv", "coverage_fingerprint"),
        ("label_counts.csv", "label_counts_fingerprint"),
        ("uncertainty.csv", "uncertainty_fingerprint"),
    ]:
        checks[filename] = frame_fingerprint(pd.read_csv(output / filename)) == expected[key]
    checks["model_state_exact"] = (
        json.loads((output / "model.json").read_text()) == expected["model"]
    )
    checks["training_members_exact"] = (
        state["training_members_sha256"] == expected["model"]["training_members_sha256"]
    )
    checks["frozen_protocol_exact"] = {
        k: v for k, v in PARAMETERS.items() if k != "origin"
    } == expected["scientific_parameters"]
    for filename, digest in expected["raw_inputs"].items():
        checks["input_" + filename] = sha256(ROOT / "data/raw" / filename) == digest
    for filename, digest in expected["reports"].items():
        checks["report_" + filename] = sha256(ROOT / "docs/reports" / filename) == digest
    inventory = pd.read_csv(output / "input_inventory.csv")
    checks["raw_record_count"] = int(inventory.raw_rows.sum()) == 42192373
    checks["active_evaluation_count"] = (
        int(
            (z.proxy_error_allowed & z.split.eq("evaluate") & z.inference_regime.eq("active")).sum()
        )
        == 101998
    )
    checks["no_unreliable_zero_scoring"] = not (
        z.proxy_error_allowed & z.zero_with_engine_activity
    ).any()
    checks["no_physical_error_claim"] = (
        not z.physical_error_available.any() and not z.physical_interval_guaranteed.any()
    )
    # Independent error algebra checks every reported group and model.
    for row in pd.read_csv(output / "metrics.csv").itertuples():
        mask = z.proxy_error_allowed & z.split.eq(row.split)
        if row.month != "all":
            mask &= z.month.eq(row.month)
        if row.regime != "all":
            mask &= z.inference_regime.eq(row.regime)
        g = z[mask]
        error = g[row.method].to_numpy() - g.power.to_numpy()
        assert len(g) == row.n
        assert np.isclose(abs(error).mean(), row.mae, rtol=1e-12, atol=1e-9)
        assert np.isclose(np.sqrt((error * error).mean()), row.rmse, rtol=1e-12, atol=1e-9)
        assert np.isclose(error.mean(), row.bias, rtol=1e-12, atol=1e-9)
    checks["independent_metrics"] = True
    # Verify required actual Noon fields and their limits from current source files.
    noon = [pd.read_csv(p) for p in sorted((ROOT / "data/noon_reports").glob("*.csv"))]
    checks["noon_reports_366"] = sum(len(n) for n in noon) == 366
    checks["noon_has_no_power_reference"] = all(
        not any(any(k in c.lower() for k in ["shaft", "torque", "power", "kw"]) for c in n.columns)
        for n in noon
    )
    # Production dependencies must remain in the functional package or installed libraries.
    banned = {
        "canonical_policy",
        "canonical_workflow",
        "model_policy",
        "model_run",
        "audit",
        "config",
        "bounded_comparison",
        "near_zero_policy_v2",
    }
    modules = {}
    for path in (ROOT / "src/vessel_power").glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        modules[path.name] = sorted(imports)
        assert not banned & imports
        private_prefixes = ("C:" + chr(92) + "Users" + chr(92), "/" + "Users" + "/")
        assert not any(prefix in path.read_text() for prefix in private_prefixes)
    checks["independent_import_graph"] = True
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "check_count": len(checks),
        "frame_fingerprints": {name: frame_fingerprint(frame) for name, frame in objects.items()},
        "baseline_id": expected["baseline_id"],
        "maximum_prediction_difference_kw": 0.0 if checks["predictions"] else None,
        "archive_runtime_access": "forbidden by process audit hook",
        "runtime_imports": modules,
        "physical_accuracy_validated": False,
        "live_rolling_approved": False,
        "cleaning_savings_approved": False,
    }
    if not result["passed"]:
        raise AssertionError(
            "Failed scientific checks: " + ", ".join(k for k, v in checks.items() if not v)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--processed", default="data/processed/measurements.pkl")
    args = parser.parse_args()
    output = project_path(args.results)
    processed = project_path(args.processed)
    sys.addaudithook(deny_archive_access)
    result = scientific_checks(output, processed)
    result["numerical_parity_scope"] = (
        "Accepted final static RPM-cubic,72h offset and rolling workflow; candidate historical-fit parity is reported separately."
    )
    supporting = {}
    for name, folder in [
        ("candidate_comparison", "model_comparison"),
        ("change_investigation", "change_analysis"),
    ]:
        path = ROOT / "results" / folder / "verification.json"
        if path.exists():
            report = json.loads(path.read_text())
            supporting[name] = {
                k: report[k]
                for k in [
                    "completed",
                    "passed",
                    "historical_scores_exact",
                    "training_labels",
                    "historical_training_labels",
                    "final_artifacts_unchanged",
                ]
                if k in report
            }
        else:
            supporting[name] = {"status": "optional analysis not generated"}
    result["supporting_analyses"] = supporting
    # Functional tests run against the new package, not archived developer suites.
    import importlib.util, io

    suite = unittest.TestSuite()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location("final_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.OUTPUT = output
        module.PROCESSED = processed
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    result["tests"] = {
        "run": tests.testsRun,
        "failures": len(tests.failures),
        "errors": len(tests.errors),
        "passed": tests.wasSuccessful(),
    }
    result["passed"] &= tests.wasSuccessful()
    result["test_log"] = stream.getvalue()
    result["source_hashes"] = {
        p.relative_to(ROOT).as_posix(): sha256(p)
        for p in sorted(
            [
                *(ROOT / "src").rglob("*.py"),
                *(ROOT / "tests").glob("*.py"),
                ROOT / "run.py",
                ROOT / "verify.py",
                ROOT / "requirements.txt",
            ]
        )
    }
    write_json(result, output / "verification.json")
    print(stream.getvalue(), flush=True)
    print(
        json.dumps(
            {
                "scientific_checks": result["check_count"],
                "tests": result["tests"],
                "numerical_parity": result["passed"],
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
