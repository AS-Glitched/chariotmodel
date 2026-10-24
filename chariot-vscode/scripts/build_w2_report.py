"""Build the W2 stage report (the clear-sky upgrade).

W2 replaced the exported clear-sky reference with one documented reference
that is identical for both years:

* the array layout is not in the export, so the geometry is *inferred* from
  the measured POA of the 2022 train slice (a 15 degree north-facing fixed
  plane; a single-axis tracker fits 4.4x worse on the same data, so no
  tracker mix is invented),
* the atmospheric model is Ineichen with the static Linke-turbidity
  climatology (no year-specific information),
* the DC conversion is the one the export used, so the two references are
  directly comparable.

Inputs:
  runs/w1_data_2022_2023/qc.json              - the exported reference QC
  runs/w2_data_cs_v2/qc.json                  - the rebuilt reference QC
  var/data/cs_v2/dka_alice_springs_2022_2023.manifest.json - the geometry
  runs/w1_ops_v3_val_2022/                    - the W1 baseline (2022 val)
  runs/ops_v3_cs_v2_val2022/                  - the W2 run (2022 val)
  runs/w1_ops_v3_protocol_2022_2023/          - the W1 baseline (2023 test)
  runs/ops_v3_cs_v2_2023test/                 - the W2 run (2023 test)

Output: runs/w2_clearsky_v2/{summary.md,comparison.json}
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


def _metrics(summary: dict) -> dict:
    solar = summary.get("evaluation", {}).get("solar_w", {})
    return {
        "mae_mean": solar.get("mae_mean"),
        "rmse_mean": solar.get("rmse_mean"),
        "bias_mean": solar.get("bias_mean"),
        "coverage_mean": solar.get("coverage_mean"),
        "energy_error_wh_mean": solar.get("energy_error_wh_mean"),
        "n_windows": solar.get("n_windows"),
        "survival_rate": summary.get("mean_survival_rate"),
        "min_survival_rate": summary.get("min_survival_rate"),
        "total_rejected": summary.get("total_rejected"),
        "constraint_counts": summary.get("constraint_counts"),
    }


def _baselines(summary: dict) -> dict:
    return {
        name: {
            "mae_mean": b.get("mae_mean"),
            "rmse_mean": b.get("rmse_mean"),
            "bias_mean": b.get("bias_mean"),
            "energy_error_wh_mean": b.get("energy_error_wh_mean"),
        }
        for name, b in (summary.get("baselines") or {}).items()
    }


def main() -> int:
    qc_old = _load(ROOT / "runs" / "w1_data_2022_2023" / "qc.json")
    qc_new = _load(ROOT / "runs" / "w2_data_cs_v2" / "qc.json")
    manifest = _load(ROOT / "var" / "data" / "cs_v2" / "dka_alice_springs_2022_2023.manifest.json")
    clear_sky = manifest["clear_sky"]

    w1_val = _load(ROOT / "runs" / "w1_ops_v3_val_2022" / "summary.json")
    w2_val = _load(ROOT / "runs" / "ops_v3_cs_v2_val2022" / "summary.json")
    w1_test = _load(ROOT / "runs" / "w1_ops_v3_protocol_2022_2023" / "summary.json")
    w2_test = _load(ROOT / "runs" / "ops_v3_cs_v2_2023test" / "summary.json")
    w2_test_manifest = _load(ROOT / "runs" / "ops_v3_cs_v2_2023test" / "manifest.json")

    kt_old = qc_old["slices"]["validate_2022"]["kt_over_one"]
    kt_new = qc_new["slices"]["validate_2022"]["kt_over_one"]
    kt_old_test = qc_old["slices"]["test_2023"]["kt_over_one"]
    kt_new_test = qc_new["slices"]["test_2023"]["kt_over_one"]

    m1, m2 = _metrics(w1_val), _metrics(w2_val)
    t1, t2 = _metrics(w1_test), _metrics(w2_test)

    report = build_stage_report(
        stage_id="W2",
        title="Clear-sky upgrade (one documented reference for both years)",
        out_dir=ROOT / "runs" / "w2_clearsky_v2",
        protocol={
            "name": "year2022_2023",
            "version": manifest.get("protocol_version"),
            "train": "first 60 % of the 2022 window (fit only)",
            "validate": "remaining 40 % of the 2022 window (selection only)",
            "test": "the whole 2023 window (report only — never fitted, never selected on)",
            "model change": (
                "none in the model — the inference code is untouched; the change is "
                "the *reference* the model reads (a data-side change, versioned as "
                f"{clear_sky['clear_sky_version']})"
            ),
        },
        data={
            "content_hashes": manifest.get("content_hashes", {}),
            "split_boundaries": manifest.get("split_boundaries", {}),
            "clear_sky": clear_sky,
            "qc_before": {
                name: slice_qc["kt_over_one"]
                for name, slice_qc in qc_old["slices"].items()
            },
            "qc_after": {
                name: slice_qc["kt_over_one"]
                for name, slice_qc in qc_new["slices"].items()
            },
        },
        model={
            "model_name": w2_test_manifest["model"]["model_name"],
            "model_version": w2_test_manifest["model"]["model_version"],
            "clear_sky_version": clear_sky["clear_sky_version"],
            "clear_sky_geometry": (
                f"fixed plane, tilt {clear_sky['geometry']['tilt_deg']}°, "
                f"azimuth {clear_sky['geometry']['azimuth_deg']}° "
                f"({clear_sky['geometry']['azimuth_convention']})"
            ),
            "clear_sky_model": (
                "Ineichen/Perez with the pvlib Linke-turbidity climatology, "
                "Hay-Davies transposition, DC conversion rated_pv_w / 1176"
            ),
            "geometry_evidence": (
                f"fixed plane RMSE {clear_sky['geometry_inference']['best']['rmse_w_m2']:.2f} W/m² "
                f"vs single-axis tracker RMSE {clear_sky['tracker_reference_rmse_w_m2']:.2f} W/m² "
                "on the 2022 train slice"
            ),
            "residual_version": w2_test_manifest["residual"].get("residual_version"),
            "selected_w": w2_test_manifest["residual"].get("selected_w"),
            "tribunal_policy": w2_test_manifest["tribunal"].get("policy_version"),
        },
        metrics={
            "validate_2022 (selection) — W1 baseline (exported reference)": m1,
            "validate_2022 (selection) — W2 (clear-sky v2)": m2,
            "test_2023 (report only) — W1 baseline (exported reference)": t1,
            "test_2023 (report only) — W2 (clear-sky v2)": t2,
        },
        baselines={
            **{
                f"validate_2022 :: {name}": b
                for name, b in _baselines(w1_val).items()
            },
            **{
                f"validate_2022 :: {name} (cs v2)": b
                for name, b in _baselines(w2_val).items()
            },
            **{
                f"test_2023 :: {name}": b
                for name, b in _baselines(w1_test).items()
            },
            **{
                f"test_2023 :: {name} (cs v2)": b
                for name, b in _baselines(w2_test).items()
            },
        },
        gates=[
            {
                "name": "GATE: daylight clearness > 1.05 rate down on 2022 validation",
                "target": "lower than the W1 baseline (22.45 %)",
                "observed": f"{kt_new['pct_gt_1_05']:.2f} % "
                f"(was {kt_old['pct_gt_1_05']:.2f} %)",
                "pass": kt_new["pct_gt_1_05"] < kt_old["pct_gt_1_05"],
                "note": "The exported reference was physically impossible on a quarter "
                "of the daylight rows (the plant beat its own clear sky, max ratio "
                "above 3). The rebuilt reference is above the measurement.",
            },
            {
                "name": "GATE (alt): clear-sky naive error down on 2022 validation",
                "target": "lower than the W1 baseline (22,757 W)",
                "observed": f"{_baselines(w2_val)['clear_sky_naive']['mae_mean']:,.0f} W "
                f"(was {_baselines(w1_val)['clear_sky_naive']['mae_mean']:,.0f} W) — higher",
                "pass": _baselines(w2_val)["clear_sky_naive"]["mae_mean"]
                < _baselines(w1_val)["clear_sky_naive"]["mae_mean"],
                "note": "This half of the gate FAILS, and it is the honest outcome: the "
                "old reference was rigged low, so the naive baseline looked better than "
                "it was. The model's margin over the naive baseline grows from "
                "1,614 W to 20,248 W on the same windows.",
            },
            {
                "name": "No regression: 2022 validation MAE",
                "target": "not materially worse than the W1 baseline (21,143 W)",
                "observed": f"{m2['mae_mean']:,.0f} W "
                f"({(m2['mae_mean'] / m1['mae_mean'] - 1) * 100:+.1f} %)",
                "pass": m2["mae_mean"] <= m1["mae_mean"] * 1.02,
            },
            {
                "name": "No regression: 2022 validation survival",
                "target": "not worse than the W1 baseline (0.944)",
                "observed": f"{m2['survival_rate']:.3f} "
                f"(min {m2['min_survival_rate']:.3f} vs {m1['min_survival_rate']:.3f})",
                "pass": m2["survival_rate"] >= m1["survival_rate"],
                "note": "The physically impossible SOLAR_OVERSCALE rejections fall from "
                f"{m1['constraint_counts']['SOLAR_OVERSCALE']} to "
                f"{m2['constraint_counts']['SOLAR_OVERSCALE']} on the validation windows.",
            },
            {
                "name": "No tracker mix invented",
                "target": "the array layout is not in the export, so no fixed/tracker "
                "mix is fabricated",
                "observed": "a fixed 15° north plane fits the measured POA at "
                f"{clear_sky['geometry_inference']['best']['rmse_w_m2']:.2f} W/m² RMSE vs "
                f"{clear_sky['tracker_reference_rmse_w_m2']:.2f} W/m² for a single-axis "
                "tracker — the fixed plane wins by 4.4x",
                "pass": clear_sky["geometry_inference"]["best"]["rmse_w_m2"]
                < clear_sky["tracker_reference_rmse_w_m2"],
            },
            {
                "name": "Nothing fitted on 2023",
                "target": "the geometry comes from 2022 train; the atmospheric model is "
                "a static climatology",
                "observed": "geometry inferred on the 2022 train slice; Linke turbidity "
                "is the static monthly climatology; the 2023 slice is only re-referenced",
                "pass": True,
            },
        ],
        claims={
            "allowed": [
                "The clear-sky reference is now identical for both years and is "
                "documented (geometry, atmospheric model, DC conversion, version).",
                "The daylight clearness rate above 1.05 fell from 22.45 % to 1.78 % on "
                "the 2022 validation window and from 18.85 % to 1.07 % on the 2023 window.",
                "The physically impossible overscale rejections fell from 101 to 21 "
                "(2022 validation) and from 225 to 34 (2023 test).",
                "The Tribunal was not touched: the survival rate went up because the "
                "scenarios stopped contradicting the reference, not because any "
                "constraint was relaxed.",
            ],
            "forbidden": [
                "Any claim that the 2023 numbers are a forecast-skill result — 2023 is "
                "report-only.",
                "Any claim that the reference is the plant's true array geometry: it is a "
                "documented single-plane equivalent inferred from the measured POA.",
                "Any claim that the clear-sky naive baseline improved — it did not; the "
                "old reference was rigged low.",
            ],
        },
        notes=[
            "The 2023 window is reported once, after the gate was decided on 2022.",
            "The residual weight was re-selected on the 2022 validation window only "
            f"(w = {w2_test_manifest['residual'].get('selected_w')}).",
            "The worst 2023 window is still 2023-01-22 07:00 UTC: a morning ramp "
            "rejected by SOLAR_RAMP_LIMIT. The reference now explains why the plant "
            "ramped faster than a fixed plane would; that is the W3/W4 territory, and "
            "the Tribunal stays as it is.",
            "The clear-sky reference is a data-preparation step (pvlib); the "
            "operational model path stays pure NumPy.",
        ],
    )
    print(json.dumps({"stage": report["stage_id"], "out": str(ROOT / "runs" / "w2_clearsky_v2")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
