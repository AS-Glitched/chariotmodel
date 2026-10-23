"""CHARIOT v3 end-to-end spine.

    Telemetry
      -> Site Master
      -> Admissibility Pre-Gate
      -> River Flow (slices + replay certificate)
      -> Calibration
      -> Immutable Capsule
      -> Training Extract (60/20/20 time-ordered split)
      -> CHARIOT model
      -> Residual Chamber (optional, capped)
      -> Scenario Genesis
      -> Reality Tribunal + Rejection Echo
      -> Physics-Survival Confidence
      -> Advisory (Research / Grid Engineer / Investor)

Every layer writes its artifacts and lineage into the store so that a full
replay is possible from the advisory id back to the raw telemetry.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chariot.admissibility import evaluate_frame, summarize
from chariot.advisory import build_all
from chariot.calibration import create_version, approve_version, resolve as resolve_calibration
from chariot.capsule import seal_capsule
from chariot.config import (
    ADMISSIBILITY_POLICY,
    CONFIDENCE_POLICY,
    HORIZONS,
    RESIDUAL_POLICY,
    TRIBUNAL_POLICY,
    policy_snapshot,
)
from chariot.confidence import evaluate as evaluate_confidence
from chariot.feedback import run_cortex
from chariot.genesis import generate_batch
from chariot.governor import ResidualStabilityGovernor
from chariot.hashing import hash_obj, short_hash
from chariot.metrics import aggregate_windows, baseline_scoreboard, score_window
from chariot.model import (
    clear_sky_series,
    filter_state_series,
    fit as fit_model,
    one_step_prediction,
    simulate,
    solar_series,
)
from chariot.residual import (
    build_features,
    fit_ridge,
    register as register_residual,
    select_weight,
)
from chariot.river_flow import (
    build_horizontal_slice,
    build_vertical_slice,
    issue_replay_certificate,
    verify_replay,
)
from chariot.site_master import resolve_context, upsert_site
from chariot.store import Store, new_id, utc_now
from chariot.training_extract import build_training_extract, load_dataset
from chariot.tribunal import validate_batch

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Site Master rating for the DKASC Alice Springs reference site
# --------------------------------------------------------------------------

#: Nominal coverage target of the published P10-P90 band.
COVERAGE_TARGET = 0.78

#: Bounds of the validation-fitted ensemble spread correction.
SPREAD_FACTOR_BOUNDS = (0.70, 1.30)

DEFAULT_WINDOW_START = "2022-01-01"
DEFAULT_WINDOW_END = "2023-01-01"

#: Declared load profile in UTC hours (W).  The site has **no** load meter:
#: this is an engineering declaration carried in the Site Master, and the
#: model treats the load as a declared profile with a declared uncertainty.
DECLARED_LOAD_PROFILE_W: dict[str, float] = {
    "0": 45000, "1": 46000, "2": 47000, "3": 46000, "4": 44000, "5": 42000,
    "6": 38000, "7": 34000, "8": 28000, "9": 22000, "10": 19000, "11": 17000,
    "12": 16000, "13": 16000, "14": 16000, "15": 16000, "16": 16000, "17": 17000,
    "18": 18000, "19": 19000, "20": 20000, "21": 22000, "22": 26000, "23": 32000,
}


def alice_springs_site() -> dict[str, Any]:
    """The Site Master rating used for the DKASC Alice Springs site."""
    return {
        "tenant_id": "dkasc",
        "site_id": "dka_alice_springs",
        "name": "DKASC Alice Springs (PV reference site)",
        "rated_pv_w": 220_000.0,
        "usable_battery_capacity_wh": 220_000.0,
        "max_charge_w": 110_000.0,
        "max_discharge_w": 110_000.0,
        "timezone": "Australia/Darwin",
        "latitude": -23.762,
        "longitude": 133.875,
        "fleet_id": "dkasc",
        "metadata": {
            "max_grid_import_w": 60_000.0,
            "max_grid_export_w": 250_000.0,
            "solar_ramp_w_per_step": 110_000.0,
            "soc_floor_pct": 5.0,
            "soc_ceiling_pct": 95.0,
            "charge_loss_factor": 0.03,
            "discharge_loss_factor": 0.03,
            "load_relative_sigma": 0.18,
            "initial_soc_percent": 50.0,
            "load_profile_w": DECLARED_LOAD_PROFILE_W,
            "load_source": "declared_profile",
            "load_observed": False,
            "soc_source": "declared_setpoint",
            "rating_note": (
                "PV rating from the DKASC site header. Battery, interconnection and "
                "load are engineering declarations for the CHARIOT reference site: "
                "the DKASC dataset itself is solar-only."
            ),
        },
    }


@dataclass
class PipelineConfig:
    """Configuration for one end-to-end CHARIOT run."""

    source_csv: str | Path
    out_dir: str | Path = "runs/chariot3_spring2022"
    store_path: str | Path = "var/chariot.db"
    tenant_id: str = "dkasc"
    site_id: str = "dka_alice_springs"
    start: str = DEFAULT_WINDOW_START
    end: str = DEFAULT_WINDOW_END
    n_scenarios: int = 500
    seed: int = 7
    horizon: str = "operational"
    train_frac: float = 0.60
    validate_frac: float = 0.20
    #: Calibrate the ensemble spread on the validation split.
    calibrate_spread: bool = True
    spread_windows: int = 6
    spread_scenarios: int = 120
    test_frac: float = 0.20
    use_residual: bool = True
    residual_alpha: float = 25.0
    max_windows: int | None = None
    keep_parquet: bool = False
    site: dict[str, Any] = field(default_factory=alice_springs_site)


def _load_telemetry_window(csv_path: str | Path, start: str, end: str) -> pd.DataFrame:
    """Load and slice the analysis window from telemetry.

    Accepts either the raw CSV export or the prepared Parquet slice; the
    window filter is idempotent either way.
    """
    path = Path(csv_path)
    if path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    mask = (frame["timestamp_utc"] >= pd.Timestamp(start, tz="UTC")) & (
        frame["timestamp_utc"] < pd.Timestamp(end, tz="UTC")
    )
    sliced = frame.loc[mask].sort_values("timestamp_utc").reset_index(drop=True)
    if sliced.empty:
        raise ValueError(f"no telemetry in {start}..{end} for {csv_path}")
    if "is_gap" in sliced.columns:
        sliced["is_gap"] = sliced["is_gap"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        sliced["is_gap"] = False
    return sliced


def run(config: PipelineConfig) -> dict[str, Any]:
    """Execute the full CHARIOT spine and return the run manifest."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(config.store_path)

    site = config.site
    upsert_site(store, site["tenant_id"], site["site_id"], **{k: v for k, v in site.items() if k not in ("tenant_id", "site_id")})
    ctx = resolve_context(store, config.tenant_id, config.site_id)

    # ---------------------------------------------------------------- data
    raw = _load_telemetry_window(config.source_csv, config.start, config.end)

    # ------------------------------------------------------- admissibility
    decisions = evaluate_frame(raw, ctx=ctx, policy=ADMISSIBILITY_POLICY, feature_set="solar_only_v1")
    decision_rows = decisions.to_dict(orient="records")
    for ts, row in zip(raw["timestamp_utc"].astype(str), decision_rows):
        store.insert(
            "admissibility_decisions",
            {
                "decision_id": new_id("adm"),
                "tenant_id": config.tenant_id,
                "site_id": config.site_id,
                "timestamp_utc": ts,
                "decision": row["decision"],
                "hard_failures": json.dumps(row["hard_failures"]),
                "soft_warnings": json.dumps(row["soft_warnings"]),
                "rule_set_version": row["rule_set_version"],
                "input_fingerprint": row["input_fingerprint"],
                "created_at_utc": utc_now(),
            },
        )
    admitted_mask = decisions["decision"].isin(["ADMISSIBLE", "ADMISSIBLE_WITH_WARNINGS"]).to_numpy()
    admitted = raw.loc[admitted_mask].copy().reset_index(drop=True)
    admitted["decision"] = decisions.loc[admitted_mask, "decision"].to_numpy()
    admitted["quality_flags"] = decisions.loc[admitted_mask, "soft_warnings"].apply(list).to_numpy()
    admitted["input_fingerprint"] = decisions.loc[admitted_mask, "input_fingerprint"].to_numpy()

    # ---------------------------------------------------------- river flow
    vertical = build_vertical_slice(
        store, admitted, tenant_id=config.tenant_id, site_id=config.site_id
    )
    horizontal = {
        param: build_horizontal_slice(
            store, admitted, param, tenant_id=config.tenant_id, site_id=config.site_id
        )
        for param in ("solar_w", "clear_sky_w")
    }
    slice_ids = [vertical["slice_id"]] + [h["slice_id"] for h in horizontal.values()]
    slice_hashes = [vertical["content_hash"]] + [h["content_hash"] for h in horizontal.values()]
    replay = issue_replay_certificate(
        store, slice_ids, tenant_id=config.tenant_id, site_id=config.site_id
    )
    replay_check = verify_replay(store, replay["replay_id"])

    # -------------------------------------------------------- calibration
    calib_version = create_version(
        store,
        scope_type="site",
        scope_id=config.site_id,
        capacity_factor=1.0,
        charge_loss_factor=float(ctx.metadata.get("charge_loss_factor", 0.0)),
        discharge_loss_factor=float(ctx.metadata.get("discharge_loss_factor", 0.0)),
        pv_correction_factor=1.0,
        method="engineering_declaration",
        notes="site-level operating losses and PV correction for the DKASC reference site",
        tenant_id=config.tenant_id,
        site_id=config.site_id,
    )
    approve_version(store, calib_version, approved_by="pipeline")
    factors = resolve_calibration(
        store, tenant_id=config.tenant_id, site_id=config.site_id, fleet_id=ctx.fleet_id
    )

    # ------------------------------------------------------------- capsule
    capsule = seal_capsule(
        store,
        admitted,
        tenant_id=config.tenant_id,
        site_id=config.site_id,
        calibration_version=factors.calibration_version,
        admissibility_versions=[ADMISSIBILITY_POLICY.rule_set_version],
        slice_ids=slice_ids,
        slice_hashes=slice_hashes,
        ctx=ctx,
        created_by="pipeline",
    )

    # ---------------------------------------------------- training extract
    manifest = build_training_extract(
        store,
        admitted,
        tenant_id=config.tenant_id,
        site_id=config.site_id,
        capsule_ids=[capsule["capsule_id"]],
        calibration_version=factors.calibration_version,
        calibration_factors=factors.to_dict(),
        admissibility_versions=[ADMISSIBILITY_POLICY.rule_set_version],
        slice_ids=slice_ids,
        ctx=ctx,
        train_frac=config.train_frac,
        validate_frac=config.validate_frac,
        test_frac=config.test_frac,
        created_by="pipeline",
    )
    dataset = load_dataset(store, manifest["dataset_version"])

    # ---------------------------------------------------------------- model
    train = dataset.loc[(dataset["split"] == "train") & dataset["is_trainable"]].reset_index(drop=True)
    validate = dataset.loc[(dataset["split"] == "validate") & dataset["is_trainable"]].reset_index(drop=True)
    test = dataset.loc[(dataset["split"] == "test") & dataset["is_trainable"]].reset_index(drop=True)

    model = fit_model(
        train,
        ctx,
        tenant_id=config.tenant_id,
        site_id=config.site_id,
        fleet_id=ctx.fleet_id or "fleet",
        dataset_version=manifest["dataset_version"],
        dataset_content_hash=manifest["content_hash"],
        validation=validate,
    )
    # ------------------------------------------- ensemble spread calibration
    spread_info: dict[str, Any] = {"calibrated": False}
    if config.calibrate_spread and len(validate):
        spread_info = _calibrate_spread(
            model,
            validate,
            ctx,
            config.site_id,
            horizon_steps=HORIZONS[config.horizon],
            target_coverage=COVERAGE_TARGET,
            n_windows=config.spread_windows,
            n_scenarios=config.spread_scenarios,
            seed=config.seed,
        )
        model.sites[config.site_id].solar.diagnostics["spread_calibration"] = spread_info

    store.upsert(
        "model_versions",
        {
            "model_version": model.model_version,
            "model_name": model.model_name,
            "dataset_version": manifest["dataset_version"],
            "dataset_content_hash": manifest["content_hash"],
            "params_json": json.dumps(model.to_dict(), sort_keys=True, default=str),
            "diagnostics_json": json.dumps(model.diagnostics, sort_keys=True, default=str),
            "status": "active",
            "created_at_utc": utc_now(),
        },
    )

    # --------------------------------------------------- residual chamber
    residual_info: dict[str, Any] = {"enabled": False}
    residual_artifact = None
    residual_weight = 0.0
    if config.use_residual:
        residual_artifact, residual_weight, residual_info = _fit_residual(
            store,
            model,
            train,
            validate,
            ctx,
            factors.to_dict(),
            dataset,
            config,
        )

    # ---------------------------------------------- governor (production)
    governor = ResidualStabilityGovernor(store=store, w_max=RESIDUAL_POLICY.w_max)
    governor.set_weight(float(residual_weight), actor="pipeline")

    # ------------------------------------------------------ rolling origins
    history = dataset.sort_values("timestamp_utc").reset_index(drop=True)
    windows = _plan_windows(test, HORIZONS[config.horizon], config.max_windows)

    results: list[dict[str, Any]] = []
    for idx, (origin_label, future, actual) in enumerate(windows):
        state = _origin_state(model, history, future, ctx)
        batch = generate_batch(
            model,
            config.site_id,
            future,
            tenant_id=config.tenant_id,
            n_scenarios=config.n_scenarios,
            seed=config.seed + idx,
            ctx=ctx,
            calibration=factors.to_dict(),
            residual_version=residual_artifact.residual_version if residual_artifact else None,
            residual_weight=governor.get_effective_w(),
            residual_artifact=residual_artifact,
            state=state,
            dt_h=0.25,
        )
        store.insert(
            "scenario_batches",
            {
                "batch_id": batch.batch_id,
                "tenant_id": batch.tenant_id,
                "site_id": batch.site_id,
                "origin_utc": batch.origin_utc,
                "horizon_steps": batch.horizon_steps,
                "dt_h": batch.dt_h,
                "n_scenarios": batch.n_scenarios,
                "model_version": batch.model_version,
                "residual_version": batch.residual_version,
                "residual_weight": batch.residual_weight,
                "dataset_version": batch.dataset_version,
                "content_hash": batch.content_hash,
                "created_at_utc": utc_now(),
            },
        )
        tribunal = validate_batch(
            batch, ctx=ctx, calibration=factors.to_dict(), policy=TRIBUNAL_POLICY, store=store
        )
        report = evaluate_confidence(
            batch,
            tribunal.retained_idx,
            survival_rate=tribunal.survival_rate,
            policy=CONFIDENCE_POLICY,
            residual_context={
                "residual_version": batch.residual_version,
                "residual_weight": batch.residual_weight,
                "governor_state": governor.state.state,
            },
            store=store,
        )
        lineage = _lineage(
            config=config,
            ctx=ctx,
            manifest=manifest,
            capsule=capsule,
            replay=replay,
            slice_ids=slice_ids,
            slice_hashes=slice_hashes,
            factors=factors,
            model=model,
            residual_artifact=residual_artifact,
            batch=batch,
            tribunal=tribunal,
            report=report,
            origin=origin_label,
        )
        baselines = baseline_scoreboard(
            actual,
            last_solar=state["last_solar"],
            last_load=state["last_load"],
            last_soc=state["soc_percent"],
        )
        advisories = build_all(
            batch,
            tribunal.retained_idx,
            report,
            lineage=lineage,
            ctx=ctx,
            baselines=baselines,
            rejection_sample=tribunal.rejections,
        )
        for role, payload in advisories.items():
            store.insert(
                "advisories",
                {
                    "advisory_id": new_id("adv"),
                    "tenant_id": config.tenant_id,
                    "site_id": config.site_id,
                    "role_mode": role,
                    "horizon": payload["horizon"],
                    "status": payload["status"],
                    "batch_id": batch.batch_id,
                    "report_id": report.report_id,
                    "payload_json": json.dumps(payload, sort_keys=True, default=str),
                    "created_at_utc": utc_now(),
                },
            )
        metrics = score_window(actual, advisories["grid_engineer"], dt_h=0.25)
        results.append(
            {
                "origin": origin_label,
                "batch_id": batch.batch_id,
                "survival_rate": tribunal.survival_rate,
                "confidence": report.confidence,
                "uncertainty": report.uncertainty,
                "status": report.status,
                "n_retained": tribunal.n_retained,
                "n_rejected": tribunal.n_rejected,
                "constraint_counts": tribunal.constraint_counts,
                "metrics": metrics,
                "baselines": baselines,
                "advisories": advisories,
                "lineage": lineage,
                "report_id": report.report_id,
            }
        )

    # ------------------------------------------------------ feedback cortex
    cortex = []
    for entry in results:
        batch_id = entry["batch_id"]
        cortex.append(
            run_cortex(
                store,
                batch_id=batch_id,
                governor=governor,
                survival_rate=entry["survival_rate"],
                residual_magnitude=None,
                target_magnitude=None,
            )
        )

    # ------------------------------------------------------------- outputs
    summary = _summarize(results)
    run_manifest = {
        "system_version": policy_snapshot()["system_version"],
        "created_at_utc": utc_now(),
        "data_source": {
            "path": str(config.source_csv),
            "window": [config.start, config.end],
            "rows_raw": int(len(raw)),
            "rows_admitted": int(len(admitted)),
            "window": [config.start, config.end],
        },
        "admissibility": {
            "rule_set_version": ADMISSIBILITY_POLICY.rule_set_version,
            **summarize(decision_rows),
        },
        "river_flow": {
            "slice_ids": slice_ids,
            "content_hashes": slice_hashes,
            "replay_id": replay["replay_id"],
            "root_hash": replay["root_hash"],
            "verified": replay_check["valid"],
        },
        "calibration": factors.to_dict(),
        "capsule_id": capsule["capsule_id"],
        "dataset": {
            "dataset_version": manifest["dataset_version"],
            "content_hash": manifest["content_hash"],
            "trainable_row_count": manifest["trainable_row_count"],
            "total_row_count": manifest["total_row_count"],
            "split_row_counts": manifest["split_row_counts"],
            "trainable_row_counts_by_split": manifest["trainable_row_counts_by_split"],
            "split_boundaries": manifest.get("split_boundaries", {}),
            "split_policy": manifest["split_policy"],
        },
        "model": {
            "model_name": model.model_name,
            "model_version": model.model_version,
            "diagnostics": model.diagnostics,
            "solar_channel": model.sites[config.site_id].solar.to_dict(),
            "load_channel": model.sites[config.site_id].load.to_dict(),
        },
        "residual": residual_info,
        "governor": governor.status(),
        "tribunal": {
            "policy_version": TRIBUNAL_POLICY.policy_version,
            "mean_survival_rate": summary["mean_survival_rate"],
            "min_survival_rate": summary["min_survival_rate"],
            "total_proposed": summary["total_proposed"],
            "total_rejected": summary["total_rejected"],
            "constraint_counts": summary["constraint_counts"],
        },
        "confidence": {
            "mean_confidence": summary["mean_confidence"],
            "mean_uncertainty": summary["mean_uncertainty"],
            "status_counts": summary["status_counts"],
            "policy_version": CONFIDENCE_POLICY.policy_version,
        },
        "evaluation": summary["evaluation"],
        "windows": len(results),
        "store_path": str(config.store_path),
    }

    (out_dir / "manifest.json").write_text(_dump_json(run_manifest), encoding="utf-8")
    (out_dir / "summary.json").write_text(_dump_json(summary), encoding="utf-8")
    (out_dir / "advisories.json").write_text(
        _dump_json(
            [
                {
                    "origin": r["origin"],
                    "batch_id": r["batch_id"],
                    "advisories": r["advisories"],
                }
                for r in results
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "windows.json").write_text(
        _dump_json(
            [
                {k: v for k, v in r.items() if k not in ("advisories", "lineage")}
                for r in results
            ]
        ),
        encoding="utf-8",
    )
    if config.keep_parquet:
        admitted.to_parquet(out_dir / "admitted_slice.parquet", index=False)
        dataset.to_parquet(out_dir / "training_dataset.parquet", index=False)
    store.close()
    return run_manifest


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _dump_json(payload: Any) -> str:
    """Serialise to *strict* JSON.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` for non-finite floats, which
    is not valid JSON and breaks strict parsers downstream.  Non-finite values
    become ``null`` so the run artefacts are portable.
    """
    return json.dumps(_json_safe(payload), indent=2, default=str, allow_nan=False)


def _json_safe(payload: Any) -> Any:
    """Recursively replace non-finite floats with ``None``."""
    if isinstance(payload, dict):
        return {k: _json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(v) for v in payload]
    if isinstance(payload, np.ndarray):
        return _json_safe(payload.tolist())
    if isinstance(payload, (np.floating, float)):
        value = float(payload)
        return value if math.isfinite(value) else None
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.bool_,)):
        return bool(payload)
    return payload


def _calibrate_spread(
    model: Any,
    validate: pd.DataFrame,
    ctx: Any,
    site_id: str,
    *,
    horizon_steps: int,
    target_coverage: float = 0.80,
    n_windows: int = 6,
    n_scenarios: int = 120,
    seed: int = 17,
) -> dict[str, Any]:
    """Calibrate the ensemble spread on the validation split.

    The published band is a nominal P10-P90 band; a raw ensemble is rarely
    calibrated.  This routine scales the process noise of the clearness
    channel until the *validation* P10-P90 coverage matches the nominal
    target.  The test split is never used for this decision.
    """
    channel = model.sites[site_id].solar
    if validate is None or len(validate) < horizon_steps * 2:
        return {"calibrated": False, "reason": "insufficient_validation_rows"}

    history = validate.sort_values("timestamp_utc").reset_index(drop=True)
    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    stride = max(1, len(history) // max(n_windows, 1))
    for i in range(n_windows):
        start = i * stride
        future = history.iloc[start : start + horizon_steps]
        if len(future) < horizon_steps:
            break
        windows.append((future.reset_index(drop=True), future.reset_index(drop=True)))
    if not windows:
        return {"calibrated": False, "reason": "no_validation_windows"}

    def coverage_for(factor: float) -> tuple[float, int]:
        channel.spread_factor = float(factor)
        hits = 0
        total = 0
        for idx, (future, actual) in enumerate(windows):
            state = _origin_state(model, history, future, ctx)
            sim = simulate(
                model,
                site_id,
                future,
                n_scenarios=n_scenarios,
                seed=seed + idx,
                state=state,
                dt_h=0.25,
            )
            solar = sim["solar_w"]
            p10 = np.percentile(solar, 10, axis=0)
            p90 = np.percentile(solar, 90, axis=0)
            truth = solar_series(actual)
            mask = np.isfinite(truth)
            if mask.any():
                hits += int(np.sum((truth[mask] >= p10[mask]) & (truth[mask] <= p90[mask])))
                total += int(mask.sum())
        return (hits / total if total else 0.0), total

    # The correction is bounded: the ensemble spread may be nudged towards
    # the nominal coverage, but it must not be inflated so far that the
    # physics envelope (which the Tribunal certifies) becomes unreachable.
    lo, hi = SPREAD_FACTOR_BOUNDS[0], SPREAD_FACTOR_BOUNDS[1]
    best_factor, best_cov = 1.0, 0.0
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        cov, n = coverage_for(mid)
        best_factor, best_cov = mid, cov
        if cov < target_coverage:
            lo = mid
        else:
            hi = mid
    channel.spread_factor = float(np.clip(best_factor, *SPREAD_FACTOR_BOUNDS))
    return {
        "calibrated": True,
        "target_coverage": target_coverage,
        "achieved_validation_coverage": round(float(best_cov), 4),
        "spread_factor": round(float(best_factor), 4),
        "n_windows": len(windows),
        "n_scenarios": n_scenarios,
    }



def _fit_residual(
    store: Store,
    model: Any,
    train: pd.DataFrame,
    validate: pd.DataFrame,
    ctx: Any,
    calibration: dict[str, float],
    dataset: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[Any, float, dict[str, Any]]:
    """Fit the residual chamber on train and pick ``w`` on validate."""
    from chariot.model import filter_state_series

    site_id = config.site_id
    history = train.sort_values("timestamp_utc").reset_index(drop=True)
    clear = clear_sky_series(history)
    mu_solar = one_step_prediction(model, site_id, history)
    y_solar = solar_series(history)

    expected_ratio = np.where(clear > 1.0, mu_solar / np.maximum(clear, 1.0), 0.0)
    X_train, names = build_features(
        history,
        mu_solar=mu_solar,
        mu_load=np.zeros(len(history)),
        clear_sky=clear,
        expected_ratio=expected_ratio,
        last_solar=float(y_solar[0]),
        calibration=calibration,
    )
    target = y_solar - mu_solar
    mask = np.isfinite(target)
    artifact = fit_ridge(
        X_train[mask],
        target[mask],
        alpha=config.residual_alpha,
        feature_names=names,
        target="solar_w",
        model_version=model.model_version,
        dataset_version=model.dataset_version,
        w_max=RESIDUAL_POLICY.w_max,
    )
    register_residual(store, artifact)

    # -- weight selection on the validation split -------------------------
    val_history = (
        dataset.loc[
            (dataset["split"].isin(["train", "validate"])) & dataset["is_trainable"]
        ]
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    val_clear = clear_sky_series(val_history)
    val_mu = one_step_prediction(model, site_id, val_history)
    val_y = solar_series(val_history)
    val_expected = np.where(val_clear > 1.0, val_mu / np.maximum(val_clear, 1.0), 0.0)
    X_val, _ = build_features(
        val_history,
        mu_solar=val_mu,
        mu_load=np.zeros(len(val_history)),
        clear_sky=val_clear,
        expected_ratio=val_expected,
        last_solar=float(val_y[0]),
        calibration=calibration,
    )
    val_mask = (
        np.isfinite(val_y)
        & (val_history["split"] == "validate").to_numpy()
        & (val_clear > 1.0)
    )
    selection = select_weight(
        artifact,
        X_val,
        y_true=val_y,
        mu=val_mu,
        mask=val_mask,
    )
    best_w = float(selection["best_w"])
    scores = selection["scores"]
    info = {
        "enabled": True,
        "residual_version": artifact.residual_version,
        "family": artifact.family,
        "target": artifact.target,
        "w_max": artifact.w_max,
        "selected_w": best_w,
        "selection_scores": scores,
        "train_diagnostics": artifact.diagnostics,
        "validation_mae_at_selected_w": min(s["mae"] for s in scores),
        "validation_mae_at_w0": next(
            (s["mae"] for s in scores if abs(s["w"]) < 1e-12), None
        ),
        "note": (
            "The load target has no residual corrector: the DKASC reference site "
            "has no load meter, so the load residual would be unidentifiable."
        ),
    }
    return artifact, best_w, info


def _origin_state(model: Any, history: pd.DataFrame, future: pd.DataFrame, ctx: Any) -> dict[str, float]:
    """State at the forecast origin, conditioned only on past observations."""
    from chariot.model import filter_state_series

    origin_ts = pd.Timestamp(future["timestamp_utc"].iloc[0])
    past = history.loc[history["timestamp_utc"] < origin_ts].sort_values("timestamp_utc")
    if past.empty:
        past = history.head(1)
    filt = filter_state_series(model, ctx.site_id, past)
    last_solar = float(solar_series(past)[-1])
    load_profile = (ctx.metadata or {}).get("load_profile_w", {})
    last_hour = int(pd.Timestamp(past["timestamp_utc"].iloc[-1]).hour)
    last_load = float(load_profile.get(str(last_hour), 0.0))
    return {
        "v_mean": float(filt["filtered"][-1]),
        "v_var": float(filt["filtered_var"][-1]),
        "last_solar": last_solar,
        "last_load": last_load,
        "soc_percent": float((ctx.metadata or {}).get("initial_soc_percent", 50.0)),
        "origin_utc": str(origin_ts),
    }


def _plan_windows(
    test: pd.DataFrame, horizon_steps: int, max_windows: int | None
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Tile the test split into rolling-origin windows."""
    windows: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    n = len(test)
    step = max(1, horizon_steps)
    start = 0
    while start < n:
        end = min(start + step, n)
        future = test.iloc[start:end].reset_index(drop=True)
        actual = test.iloc[start:end].reset_index(drop=True)
        label = str(future["timestamp_utc"].iloc[0])
        windows.append((label, future, actual))
        start = end
    if max_windows is not None:
        windows = windows[:max_windows]
    return windows


def _lineage(
    *,
    config: PipelineConfig,
    ctx: Any,
    manifest: dict[str, Any],
    capsule: dict[str, Any],
    replay: dict[str, Any],
    slice_ids: list[str],
    slice_hashes: list[str],
    factors: Any,
    model: Any,
    residual_artifact: Any,
    batch: Any,
    tribunal: Any,
    report: Any,
    origin: str,
) -> dict[str, Any]:
    """Unbroken lineage chain from raw telemetry to the advisory."""
    return {
        "telemetry": {
            "source": str(config.source_csv),
            "window": [config.start, config.end],
        },
        "admissibility_decision": {
            "rule_set_version": ADMISSIBILITY_POLICY.rule_set_version,
            "counts": "see manifest.admissibility",
        },
        "river_slice": {
            "slice_ids": slice_ids,
            "content_hashes": slice_hashes,
            "replay_id": replay["replay_id"],
            "root_hash": replay["root_hash"],
        },
        "calibration_version": factors.calibration_version,
        "capsule_id": capsule["capsule_id"],
        "dataset_version": manifest["dataset_version"],
        "dataset_content_hash": manifest["content_hash"],
        "chariot_model_version": model.model_version,
        "residual_version": residual_artifact.residual_version if residual_artifact else None,
        "scenario_batch": {
            "batch_id": batch.batch_id,
            "origin_utc": origin,
            "n_scenarios": batch.n_scenarios,
            "content_hash": batch.content_hash,
        },
        "tribunal_result": {
            "policy_version": tribunal.policy_version,
            "solver_version": tribunal.solver_version,
            "n_proposed": tribunal.n_proposed,
            "n_retained": tribunal.n_retained,
            "n_rejected": tribunal.n_rejected,
            "survival_rate": tribunal.survival_rate,
            "constraint_counts": tribunal.constraint_counts,
        },
        "rejection_ids": [r["scenario_id"] for r in tribunal.rejections][:50],
        "confidence_report": {
            "report_id": report.report_id,
            "status": report.status,
            "policy_version": report.policy_version,
            "confidence": report.confidence,
            "survival_rate": report.survival_rate,
            "uncertainty": report.uncertainty,
        },
        "site_context": ctx.to_dict(),
        "policies": {
            "admissibility": ADMISSIBILITY_POLICY.rule_set_version,
            "tribunal": TRIBUNAL_POLICY.policy_version,
            "confidence": CONFIDENCE_POLICY.policy_version,
            "residual": RESIDUAL_POLICY.policy_version,
        },
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the rolling-origin results."""
    if not results:
        return {}
    survival = [r["survival_rate"] for r in results]
    constraint_counts: dict[str, int] = {}
    for r in results:
        for code, count in (r.get("constraint_counts") or {}).items():
            constraint_counts[code] = constraint_counts.get(code, 0) + count
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    evaluation = aggregate_windows([r["metrics"] for r in results])
    baselines: dict[str, Any] = {}
    for r in results:
        for row in r.get("baselines", []):
            key = f"{row['model']}"
            baselines.setdefault(key, []).append(row.get("mae"))
    baseline_summary = {
        k: {"mae_mean": float(np.nanmean(v))} for k, v in baselines.items() if v
    }
    return {
        "n_windows": len(results),
        "mean_survival_rate": float(np.mean(survival)),
        "min_survival_rate": float(np.min(survival)),
        "max_survival_rate": float(np.max(survival)),
        "total_proposed": int(sum(r["n_retained"] + r["n_rejected"] for r in results)),
        "total_rejected": int(sum(r["n_rejected"] for r in results)),
        "constraint_counts": dict(sorted(constraint_counts.items(), key=lambda kv: -kv[1])),
        "status_counts": status_counts,
        "mean_confidence": float(np.mean([r["confidence"] for r in results])),
        "mean_uncertainty": float(np.mean([r["uncertainty"] for r in results])),
        "evaluation": evaluation,
        "baselines": baseline_summary,
    }
