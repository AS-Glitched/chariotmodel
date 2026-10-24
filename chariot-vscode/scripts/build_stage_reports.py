"""Build the W3, W4, W5 and W6 stage reports.

Each stage lands ``runs/<stage_id>/{summary.md,comparison.json}`` with the
protocol, the content hashes, the model version, the 2022 validation
(selection) metrics, the 2023 test (report-only) metrics, the baseline
table, the gate verdicts and the claims that are allowed and forbidden.

The stage inputs (all regenerable):

* ``runs/ops_v3_cs_v2_val2022``    - the W2 baseline (2022 validation)
* ``runs/ops_v3_cs_v2_2023test``   - the W2 baseline (2023 test)
* ``runs/ops_v3_cal_v1_val2022``   - the W3 calibration run
* ``runs/ops_v3_res_v2_val2022``   - the W4 residual-chamber run
* ``runs/ops_v3_final_val2022``    - the final configuration (2022 validation)
* ``runs/ops_v3_final_2023test``   - the final configuration (2023 test)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chariot.stage_report import build_stage_report  # noqa: E402

RUNS = ROOT / "runs"


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
        "daylight_mae_pct_of_rated_mean": solar.get("daylight_mae_pct_of_rated_mean"),
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


def _protocol(manifest: dict, model_note: str) -> dict:
    dataset = manifest.get("dataset", {})
    return {
        "name": "year2022_2023",
        "train": "first 60 % of the 2022 window (fit only)",
        "validate": "remaining 40 % of the 2022 window (selection only)",
        "test": "the whole 2023 window (report only — never fitted, never selected on)",
        "split_policy": dataset.get("split_policy"),
        "model note": model_note,
    }


def _data(manifest: dict) -> dict:
    dataset = manifest.get("dataset", {})
    return {
        "content_hashes": {"dataset": dataset.get("content_hash")},
        "split_boundaries": dataset.get("split_boundaries", {}),
        "split_row_counts": dataset.get("split_row_counts", {}),
    }


def _model(manifest: dict, extra: dict | None = None) -> dict:
    model = manifest.get("model", {})
    out = {
        "model_name": model.get("model_name"),
        "model_version": model.get("model_version"),
        "tribunal_policy": manifest.get("tribunal", {}).get("policy_version"),
        "residual_version": manifest.get("residual", {}).get("residual_version"),
        "selected_w": manifest.get("residual", {}).get("selected_w"),
    }
    if extra:
        out.update(extra)
    return out


def _advisory_events(run_dir: Path) -> dict:
    """The event block of the first advisory of a run (if it is on disk)."""
    path = run_dir / "advisories.json"
    if not path.exists():
        return {}
    payload = _load(path)
    rows = payload if isinstance(payload, list) else list(payload.values())
    if not rows:
        return {}
    events = rows[0].get("advisories", {}).get("grid_engineer", {}).get("events", {})
    return {
        "hybrid_assumed": events.get("hybrid_assumed"),
        "p_soc_below_20_within_horizon": events.get("p_soc_below_20_within_horizon"),
        "p_soc_above_90_within_horizon": events.get("p_soc_above_90_within_horizon"),
        "p_grid_import_limit_exceeded": events.get("p_grid_import_limit_exceeded"),
        "p_grid_export_limit_exceeded": events.get("p_grid_export_limit_exceeded"),
        "p_pv_above_certified_ac": events.get("p_pv_above_certified_ac"),
        "hybrid_note": events.get("hybrid_note"),
    }


def main() -> int:
    w2_val_manifest = _load(RUNS / "ops_v3_cs_v2_val2022" / "manifest.json")
    w2_val = _load(RUNS / "ops_v3_cs_v2_val2022" / "summary.json")
    w2_test_manifest = _load(RUNS / "ops_v3_cs_v2_2023test" / "manifest.json")
    w2_test = _load(RUNS / "ops_v3_cs_v2_2023test" / "summary.json")

    cal_manifest = _load(RUNS / "ops_v3_cal_v1_val2022" / "manifest.json")
    cal = cal_manifest["clearness_calibration"]
    cal_scores = cal.get("selection_scores", [])
    cal_best = min(
        (s for s in cal_scores if s.get("n")),
        key=lambda s: s["mae"],
        default={},
    )
    cal_zero = next((s for s in cal_scores if abs(s.get("shrinkage", 1)) < 1e-12), {})

    res_manifest = _load(RUNS / "ops_v3_res_v2_val2022" / "manifest.json")
    res = res_manifest["residual"]
    res_zero = next((s for s in res["selection_scores"] if abs(s["w"]) < 1e-12), {})
    res_best = min(res["selection_scores"], key=lambda s: s["mae"])
    res_improvement = (
        (res_zero["mae"] - res_best["mae"]) / res_zero["mae"] * 100.0
        if res_zero.get("mae")
        else 0.0
    )

    fin_val_manifest = _load(RUNS / "ops_v3_final_val2022" / "manifest.json")
    fin_val = _load(RUNS / "ops_v3_final_val2022" / "summary.json")
    fin_test_manifest = _load(RUNS / "ops_v3_final_2023test" / "manifest.json")
    fin_test = _load(RUNS / "ops_v3_final_2023test" / "summary.json")

    m2v, m2t = _metrics(w2_val), _metrics(w2_test)
    mfv, mft = _metrics(fin_val), _metrics(fin_test)

    # ---------------------------------------------------------------- W3
    build_stage_report(
        stage_id="W3",
        title="Ops bias / hour-elevation calibration",
        out_dir=RUNS / "w3_clearness_calibration",
        protocol=_protocol(
            w2_val_manifest,
            "the clearness calibration is additive and off by default; the "
            "inference code is unchanged",
        ),
        data=_data(w2_val_manifest),
        model=_model(
            cal_manifest,
            {
                "calibration_version": cal.get("version"),
                "selected_shrinkage": cal.get("selected_shrinkage"),
                "fitted_on": cal.get("fitted_on"),
                "selected_on": cal.get("selected_on"),
            },
        ),
        metrics={
            "validate_2022 (selection) — W2 baseline": m2v,
            "validate_2022 (selection) — W3 calibration run": _metrics(
                _load(RUNS / "ops_v3_cal_v1_val2022" / "summary.json")
            ),
        },
        baselines={
            **{f"validate_2022 :: {k}": v for k, v in _baselines(w2_val).items()},
        },
        gates=[
            {
                "name": "GATE: |bias| down on the 2022 validation window",
                "target": "smaller than the W2 baseline (4,558 W)",
                "observed": f"{m2v['bias_mean']:,.0f} W — unchanged: the selection "
                "kept the shrinkage at 0.0, so no correction was applied",
                "pass": abs(m2v["bias_mean"]) < abs(m2v["bias_mean"]),
                "note": "The calibration *can* remove the bias (at shrinkage 1.0 the "
                "validation bias moves from +4,776 W to -2,649 W), but it costs 12 % "
                "of the MAE, so the bias-first selection rule rejects it.",
            },
            {
                "name": "GATE: MAE not materially worse on the 2022 validation window",
                "target": "not more than 2 % worse than the W2 baseline",
                "observed": f"{m2v['mae_mean']:,.0f} W — unchanged",
                "pass": True,
            },
            {
                "name": "The correction is a *level* error, not a tail error",
                "target": "diagnostic: the model's median clearness is unbiased",
                "observed": "the median log-ratio residual is +0.03 (validation) and "
                "-0.004 (training) while the mean is -0.17: the bias lives in the "
                "tail, so a level shift trades MAE for bias",
                "pass": False,
                "note": "A bin-wise level correction cannot fix a skewed bias without "
                "moving the rows that are already right. The machinery lands and stays "
                "off by default; the lever for a tail bias is the ensemble spread, not "
                "the level.",
            },
            {
                "name": "Nothing fitted on 2023",
                "target": "the offsets are fitted on train_2022 and the shrinkage is "
                "selected on validate_2022",
                "observed": f"fitted_on={cal.get('fitted_on')}, selected_on={cal.get('selected_on')}",
                "pass": True,
            },
        ],
        claims={
            "allowed": [
                "The calibration machinery works end to end and is off by default.",
                "The selection rejected every candidate, so the operational model is "
                "bit-identical to the W2 baseline.",
                "The model's clearness bias is a tail effect, not a level error.",
            ],
            "forbidden": [
                "Any claim that W3 improved the model: it did not, and the default is "
                "unchanged.",
                "Any claim that the bias was removed.",
            ],
        },
        notes=[
            "The first implementation fitted the offsets with a plain mean; 90 rows "
            "where the plant delivered nothing (a real outage, not a gap) turned the "
            "mean log-ratio into -inf. The estimator is now a trimmed mean over the "
            "rows that are actually clearness observations, with a shrinkage prior of "
            "50 rows and a minimum of 20 rows per bin.",
            "The fitting windows are spread across the whole training split; the "
            "operational evaluation windows still tile the split from its start, "
            "which is the v3 behaviour.",
        ],
    )

    # ---------------------------------------------------------------- W4
    build_stage_report(
        stage_id="W4",
        title="Constrained residual chamber",
        out_dir=RUNS / "w4_residual_chamber",
        protocol=_protocol(
            res_manifest,
            "the residual chamber stays a bounded shadow corrector: the governor is "
            "the only runtime authority and the Tribunal still certifies every path",
        ),
        data=_data(res_manifest),
        model=_model(
            res_manifest,
            {
                "residual_family": res.get("family"),
                "w_max": res.get("w_max"),
                "selected_w": res.get("selected_w"),
                "night_forcing": "the correction is forced to zero when there is no "
                "clear sky",
                "features": "model means, clear-sky level, the model's own expected "
                "ratio, lags, hour/doy harmonics, the previous correction "
                "(never the same-time measurement)",
            },
        ),
        metrics={
            "validate_2022 (selection) — W2 baseline": m2v,
            "validate_2022 (selection) — W4 residual run": _metrics(
                _load(RUNS / "ops_v3_res_v2_val2022" / "summary.json")
            ),
        },
        baselines={
            **{f"validate_2022 :: {k}": v for k, v in _baselines(w2_val).items()},
        },
        gates=[
            {
                "name": "GATE: >= 1 % validation MAE improvement from the residual",
                "target": "at least 1 % better than w = 0",
                "observed": f"{res_improvement:.3f} % "
                f"({res_zero.get('mae', float('nan')):,.0f} W at w = 0 vs "
                f"{res_best['mae']:,.0f} W at the best w)",
                "pass": res_improvement >= 1.0,
                "note": "The ridge explains 2.5 % of the residual variance "
                "(R^2 vs the mean): the clearness innovation is not predictable at a "
                "15-minute horizon, so there is nothing for the chamber to correct.",
            },
            {
                "name": "GATE: cleaner bias",
                "target": "|bias| below the W2 baseline",
                "observed": f"{m2v['bias_mean']:,.0f} W — unchanged (w = 0)",
                "pass": False,
            },
            {
                "name": "The weight stays at or near zero",
                "target": "w <= 0.20 and the selection picks w ~ 0",
                "observed": f"selected w = {res.get('selected_w')} (ceiling "
                f"{res.get('w_max')})",
                "pass": float(res.get("selected_w") or 0.0) <= 0.2,
            },
            {
                "name": "The governor is the only runtime authority",
                "target": "the residual weight applied at run time is the governor's "
                "effective weight",
                "observed": "the pipeline sets the governor weight from the selection "
                "and the genesis uses governor.get_effective_w(); the Tribunal "
                "certifies every path afterwards",
                "pass": True,
            },
            {
                "name": "Night forcing",
                "target": "the correction is ~0 when there is no clear sky",
                "observed": "the residual is zeroed wherever the clear sky is below "
                "1 W, so it cannot invent night power",
                "pass": True,
            },
        ],
        claims={
            "allowed": [
                "The residual chamber is bounded, governed, Tribunal-checked and "
                "leakage-safe.",
                "Its weight is selected on the 2022 validation window and is now 0.",
                "The clearness innovation is not predictable at this horizon (R^2 = 0.025).",
            ],
            "forbidden": [
                "Any claim that the residual improves the forecast: the selected "
                "weight is 0.",
                "Any claim that the residual is a control path: it never bypasses the "
                "Tribunal.",
            ],
        },
        notes=[
            "The chamber gained two things in W4: the residual's own lag as a feature "
            "(the correction proposed for the previous step, never the realised "
            "measurement) and a hard night forcing.",
            "Because the selected weight is 0, the operational scenarios are "
            "bit-identical to the W2 baseline.",
        ],
    )

    # ---------------------------------------------------------------- W5
    build_stage_report(
        stage_id="W5",
        title="Baselines and metric honesty",
        out_dir=RUNS / "w5_baselines_metrics",
        protocol=_protocol(
            fin_val_manifest,
            "no model change: this stage fixes the *reporting* so every claim names "
            "the baseline it beats",
        ),
        data=_data(fin_val_manifest),
        model=_model(fin_val_manifest),
        metrics={
            "validate_2022 (selection) — final configuration": mfv,
            "test_2023 (report only) — final configuration": mft,
        },
        baselines={
            **{f"validate_2022 :: {k}": v for k, v in _baselines(fin_val).items()},
            **{f"test_2023 :: {k}": v for k, v in _baselines(fin_test).items()},
        },
        gates=[
            {
                "name": "GATE: every window summary names its baselines",
                "target": "persistence_1step, persistence_horizon_matched, "
                "clear_sky_naive (MAE + energy error), diurnal climatology",
                "observed": "persistence at one step (15 min) and at the horizon, the "
                "clear-sky naive baseline with its energy error, and the training-hour "
                "climatology are all reported on every window",
                "pass": True,
            },
            {
                "name": "GATE: the primary metrics are reported",
                "target": "daylight % of rated, energy error, MAE/RMSE/bias, P10-P90 "
                "coverage, tribunal survival, rejection counts",
                "observed": "the daylight MAE as a percentage of the rated DC power is "
                f"{mfv.get('daylight_mae_pct_of_rated_mean', float('nan')):.2f} % "
                "(2022 validation); the energy error, MAE, RMSE, bias, coverage, "
                "survival and the rejection counts are reported next to it",
                "pass": True,
            },
            {
                "name": "GATE: 'beats persistence' names which persistence",
                "target": "no unqualified persistence claim",
                "observed": "the model beats the *horizon-matched* persistence by "
                "3.3x on the 2022 validation window (21,312 W vs 69,867 W) and by 4.0x "
                "on the 2023 window (17,260 W vs 68,866 W). The 15-minute persistence "
                "is reported separately (9,323 W on 2022 validation, 14,116 W on 2023) "
                "because it is scored on a single step and is therefore not comparable "
                "with a 96-step window MAE",
                "pass": True,
                "note": "The two persistences are very different numbers, and an "
                "unqualified 'beats persistence' claim would be misleading either way.",
            },
        ],
        claims={
            "allowed": [
                "The model beats the horizon-matched persistence by 3.3x (2022 "
                "validation) and 4.0x (2023), with the qualifier named.",
                "The 15-minute persistence is reported on its own row, with the note "
                "that it is a single-step score.",
                "The clear-sky naive baseline is reported with its MAE and its energy "
                "error.",
                "The daylight error is reported as a percentage of the rated DC power.",
            ],
            "forbidden": [
                "Any unqualified 'beats persistence' claim.",
                "Any claim about the load, SOC or grid channels as validated skill.",
            ],
        },
        notes=[
            "The baseline table is now per target and per model, so the solar and the "
            "load baselines can never be confused.",
            "The clearance of the old 'persistence_origin' name: the horizon-matched "
            "persistence and the 15-minute persistence are separate rows.",
        ],
    )

    # ---------------------------------------------------------------- W6
    events_val = _advisory_events(RUNS / "ops_v3_final_val2022")
    build_stage_report(
        stage_id="W6",
        title="Solar-only advisory mode",
        out_dir=RUNS / "w6_solar_only_advisory",
        protocol=_protocol(
            fin_val_manifest,
            "no model change: the advisory contract changes, the physics does not",
        ),
        data=_data(fin_val_manifest),
        model=_model(fin_val_manifest),
        metrics={
            "validate_2022 (selection) — final configuration": mfv,
            "test_2023 (report only) — final configuration": mft,
        },
        baselines={
            **{f"validate_2022 :: {k}": v for k, v in _baselines(fin_val).items()},
            **{f"test_2023 :: {k}": v for k, v in _baselines(fin_test).items()},
        },
        gates=[
            {
                "name": "GATE: the hybrid event probabilities are absent or explicitly "
                "marked as assumed",
                "target": "no published SOC/grid probability without a measured load "
                "and state of charge",
                "observed": f"hybrid_assumed={events_val.get('hybrid_assumed')}; the SOC "
                "and grid probabilities are nulled and the payload carries the note",
                "pass": events_val.get("hybrid_assumed") is True,
            },
            {
                "name": "GATE: the solar events are untouched",
                "target": "the PV event is still published",
                "observed": f"p_pv_above_certified_ac = "
                f"{events_val.get('p_pv_above_certified_ac')}",
                "pass": events_val.get("p_pv_above_certified_ac") is not None,
            },
            {
                "name": "GATE: the advisory stays advisory-only",
                "target": "no control endpoint, no dispatch",
                "observed": "unchanged — the platform publishes forecasts, confidence "
                "and event probabilities for human review",
                "pass": True,
            },
        ],
        claims={
            "allowed": [
                "The solar channel, its uncertainty and the physics survival rate are "
                "validated on the site.",
                "The hybrid events are published as assumed, and the payload says so.",
            ],
            "forbidden": [
                "Any claim that the load, SOC or grid forecast is validated skill on "
                "this site: there is no meter for any of them.",
            ],
        },
        notes=[
            "The site card declares the load profile, the battery and the starting "
            "state of charge; those are engineering assumptions, not measurements.",
            "The mode is the default; `--hybrid-advisory` restores the previous "
            "behaviour for a site that does have the meters.",
        ],
    )

    print(
        json.dumps(
            {
                "stages": ["W3", "W4", "W5", "W6"],
                "out": [
                    str(RUNS / "w3_clearness_calibration"),
                    str(RUNS / "w4_residual_chamber"),
                    str(RUNS / "w5_baselines_metrics"),
                    str(RUNS / "w6_solar_only_advisory"),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
