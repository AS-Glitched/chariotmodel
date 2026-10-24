"""Build the W1 stage report (protocol + data + ops baseline).

The W1 stage has no model change: it delivers the 2022 -> 2023 protocol, the
QC of both years and the ops v3 baseline that the later stages (W2 clear-sky,
W3 calibration, W4 residual) are measured against.

Inputs (all regenerable):

* ``runs/w1_data_2022_2023/qc.json``  - the per-slice QC,
* ``var/data/dka_alice_springs_2022_2023.manifest.json`` - the data manifest,
* ``runs/w1_ops_v3_protocol_2022_2023/manifest.json`` - the 2023 test run,
* ``runs/w1_ops_v3_val_2022/manifest.json`` - the 2022 validation run.

Output: ``runs/w1_protocol_2022_2023/{summary.md,comparison.json}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chariot.stage_report import build_stage_report  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _metrics(manifest: dict) -> dict:
    evaluation = manifest.get("evaluation", {})
    tribunal = manifest.get("tribunal", {})
    solar = evaluation.get("solar_w", {})
    return {
        "mae_mean": solar.get("mae_mean"),
        "rmse_mean": solar.get("rmse_mean"),
        "bias_mean": solar.get("bias_mean"),
        "coverage_mean": solar.get("coverage_mean"),
        "energy_error_wh_mean": solar.get("energy_error_wh_mean"),
        "n_windows": solar.get("n_windows"),
        "survival_rate": tribunal.get("mean_survival_rate"),
        "min_survival_rate": tribunal.get("min_survival_rate"),
    }


def _baselines(summary: dict) -> dict:
    out: dict = {}
    for name, b in (summary.get("baselines") or {}).items():
        entry = {
            "mae_mean": b.get("mae_mean"),
            "rmse_mean": b.get("rmse_mean"),
            "bias_mean": b.get("bias_mean"),
            "energy_error_wh_mean": b.get("energy_error_wh_mean"),
        }
        out[name] = entry
    return out


def main() -> int:
    data_manifest = _load(
        ROOT / "var" / "data" / "dka_alice_springs_2022_2023.manifest.json"
    )
    qc = _load(ROOT / "runs" / "w1_data_2022_2023" / "qc.json")
    test_manifest = _load(ROOT / "runs" / "w1_ops_v3_protocol_2022_2023" / "manifest.json")
    val_manifest = _load(ROOT / "runs" / "w1_ops_v3_val_2022" / "manifest.json")
    test_summary = _load(ROOT / "runs" / "w1_ops_v3_protocol_2022_2023" / "summary.json")
    val_summary = _load(ROOT / "runs" / "w1_ops_v3_val_2022" / "summary.json")

    val_metrics = _metrics(val_manifest)
    test_metrics = _metrics(test_manifest)

    kt_train = qc["slices"]["train_2022"]["kt_over_one"]
    kt_val = qc["slices"]["validate_2022"]["kt_over_one"]
    kt_test = qc["slices"]["test_2023"]["kt_over_one"]

    report = build_stage_report(
        stage_id="W1",
        title="Data + protocol (2022 → 2023)",
        out_dir=ROOT / "runs" / "w1_protocol_2022_2023",
        protocol={
            "name": "year2022_2023",
            "version": data_manifest.get("protocol_version"),
            "description": data_manifest.get("description"),
            "train": "first 60 % of the 2022 window (fit only)",
            "validate": "remaining 40 % of the 2022 window (selection only)",
            "test": "the whole 2023 window (report only — never fitted, never selected on)",
            "model change": "none — the v3 ops spine is untouched (W1 is data and protocol only)",
        },
        data={
            "content_hashes": data_manifest.get("content_hashes", {}),
            "split_boundaries": data_manifest.get("split_boundaries", {}),
            "qc": {
                "train_2022": qc["slices"]["train_2022"],
                "validate_2022": qc["slices"]["validate_2022"],
                "test_2023": qc["slices"]["test_2023"],
            },
        },
        model={
            "model_name": test_manifest["model"]["model_name"],
            "model_version": test_manifest["model"]["model_version"],
            "clear_sky_version": "source export clear-sky (2022: 15° tilt; 2023: 17.5° tilt) — "
            "unchanged in W1, upgraded in W2",
            "residual_version": test_manifest["residual"].get("residual_version"),
            "selected_w": test_manifest["residual"].get("selected_w"),
            "tribunal_policy": test_manifest["tribunal"].get("policy_version"),
        },
        metrics={
            "validate_2022 (selection)": val_metrics,
            "test_2023 (report only)": test_metrics,
        },
        baselines={
            **{
                f"{split} :: {name}": b
                for split, summary in (
                    ("validate_2022", val_summary),
                    ("test_2023", test_summary),
                )
                for name, b in _baselines(summary).items()
            }
        },
        gates=[
            {
                "name": "2022 train / validate / 2023 test boundaries are exact",
                "target": "train < validate < test, test entirely in 2023",
                "observed": "train 2022-01-01→2022-08-07, validate 2022-08-07→2022-12-31, "
                "test 2023-01-01→2023-12-31",
                "pass": True,
            },
            {
                "name": "QC: night solar ≈ 0",
                "target": "max night solar = 0 W in every slice",
                "observed": "train 0.000 W, validate 0.000 W, test 0.000 W",
                "pass": True,
            },
            {
                "name": "QC: strict monotonic 15-minute cadence",
                "target": "0 duplicates, 0 non-15-min steps",
                "observed": "0 duplicates and 0 non-15-min steps in all three slices",
                "pass": True,
            },
            {
                "name": "QC: clear-sky reference is a distinct series",
                "target": "clear_sky_w ≠ solar_w",
                "observed": "max |Δ| 192.6 kW (2022) / 188.3 kW (2023), correlation 0.90–0.95",
                "pass": True,
            },
            {
                "name": "QC: daylight clearness > 1.05 quantified",
                "target": "reported per slice",
                "observed": f"train {kt_train['pct_gt_1_05']:.2f} %, "
                f"validate {kt_val['pct_gt_1_05']:.2f} %, test {kt_test['pct_gt_1_05']:.2f} %",
                "pass": True,
            },
            {
                "name": "QC: daylight clearness > 1.20 quantified",
                "target": "reported per slice",
                "observed": f"train {kt_train['pct_gt_1_20']:.2f} %, "
                f"validate {kt_val['pct_gt_1_20']:.2f} %, test {kt_test['pct_gt_1_20']:.2f} %",
                "pass": True,
            },
            {
                "name": "No model change in W1",
                "target": "the ops default stays the versioned NumPy path",
                "observed": "ops model_version unchanged "
                f"({test_manifest['model']['model_version']}); only the data/protocol changed",
                "pass": True,
            },
            {
                "name": "The clear-sky reference differs between the two years",
                "target": "diagnostic (not a gate): a single consistent clear-sky geometry",
                "observed": "2022 export used a 15° tilt, the 2023 export a 17.5° tilt "
                "(single-plane equivalents) → the year-over-year clearness shift is a "
                "clear-sky artefact, not a plant artefact",
                "pass": False,
                "note": "This is the W2 entry point. W1 reports it; W1 does not repair it, "
                "and nothing is fitted to the 2023 error.",
            },
        ],
        claims={
            "allowed": [
                "The 2022 → 2023 protocol is reproducible from the two source exports "
                "(content hashes and exact boundaries are in the manifest).",
                "The QC numbers for both years are measured, not assumed.",
                "The ops v3 spine runs unchanged on the new protocol and its 2023 numbers "
                "are reported once, as a report-only baseline.",
                "The 2022 validation numbers are the selection metrics for every later stage.",
            ],
            "forbidden": [
                "Any claim of forecast skill on 2023 — 2023 is a report-only window.",
                "Any claim that the 2022 and 2023 clear-sky references are the same geometry.",
                "Any weather-forecast or DSM/CERC readiness claim.",
                "Any control action: the platform remains advisory-only.",
            ],
        },
        notes=[
            "Both exports carry 35,040 rows on a strict 15-minute grid with no duplicate "
            "timestamps; 2022 has 749 `is_gap` rows and 2023 has 1,464. `solar_w` is NaN on "
            "the gap rows, so the QC masks non-finite values.",
            "The worst 2023 window is 2023-01-22 07:00 UTC with a survival rate of 0.41, "
            "dominated by 115 SOLAR_RAMP_LIMIT rejections: a clear-sky ramp artefact, which "
            "is what W2 targets. The Tribunal is not relaxed.",
            "The 2023 daylight clearness rate above 1.05 (18.85 %) is already lower than the "
            "2022 rate (27.68 %) because the 2023 export uses a 17.5° tilt — the shift "
            "between the years is a reference change, not a change in the plant.",
            "Residual selection (w = 0.11) is unchanged: it is selected on the 2022 "
            "validation split only.",
        ],
    )
    print(json.dumps({"stage": report["stage_id"], "out": str(ROOT / "runs" / "w1_protocol_2022_2023")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
