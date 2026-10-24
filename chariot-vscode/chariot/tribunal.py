"""Reality Tribunal + Rejection Echo Memory — fail-closed certification (PDF §4.10).

The Tribunal is the *sole energy-accounting authority* for retained futures.
Every retained scenario must pass strict physics validation at **every**
modelled timestep.  A single timestep failure rejects the entire scenario.

Tribunal checks (every timestep):

* energy-balance residual <= ``energy_balance_tol_wh`` (default 0.1 Wh),
* ``SOC_min <= SOC <= SOC_max``,
* power / equipment limits from the Site Master + policy,
* finite values; optional ramp limits.

The Rejection Echo Memory is an append-only record of every rejection.  Echo
durability is required for batch success: if the echo write fails, the batch
fails closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chariot.config import TRIBUNAL_POLICY, TribunalPolicy
from chariot.genesis import ScenarioBatch
from chariot.hashing import hash_obj
from chariot.physics import balance_residual_wh
from chariot.store import Store, new_id, utc_now


@dataclass
class TribunalResult:
    """Outcome of validating one scenario batch."""

    batch_id: str
    n_proposed: int
    retained_idx: list[int]
    rejections: list[dict[str, Any]]
    survival_rate: float
    policy_version: str
    solver_version: str
    constraint_counts: dict[str, int] = field(default_factory=dict)
    batch_result: dict[str, Any] = field(default_factory=dict)

    @property
    def n_retained(self) -> int:
        return len(self.retained_idx)

    @property
    def n_rejected(self) -> int:
        return self.n_proposed - len(self.retained_idx)


def _certified_envelope(ctx: Any) -> dict[str, float | None]:
    meta = (getattr(ctx, "metadata", {}) or {}) if ctx is not None else {}
    return {
        "rated_pv_w": getattr(ctx, "rated_pv_w", None),
        "max_charge_w": getattr(ctx, "max_charge_w", None),
        "max_discharge_w": getattr(ctx, "max_discharge_w", None),
        "max_grid_import_w": meta.get("max_grid_import_w"),
        "max_grid_export_w": meta.get("max_grid_export_w"),
        "certified_ac_pv_w": meta.get("certified_ac_pv_w"),
        "solar_ramp_w_per_step": meta.get("solar_ramp_w_per_step"),
        "load_ramp_w_per_step": meta.get("load_ramp_w_per_step"),
        "usable_battery_capacity_wh": getattr(ctx, "usable_battery_capacity_wh", None),
    }


def validate_batch(
    batch: ScenarioBatch,
    *,
    ctx: Any,
    calibration: dict[str, float] | None = None,
    policy: TribunalPolicy = TRIBUNAL_POLICY,
    store: Store | None = None,
) -> TribunalResult:
    """Validate every scenario in the batch, fail-closed.

    When ``store`` is provided, every rejection is written to the Rejection
    Echo Memory and the batch result is persisted.  A failure to persist the
    echo raises — the system fails closed.
    """
    calibration = calibration or {}
    env = _certified_envelope(ctx)
    meta = (getattr(ctx, "metadata", {}) or {}) if ctx is not None else {}

    solar = np.asarray(batch.solar_w, dtype=float)
    load = np.asarray(batch.load_w, dtype=float)
    battery = np.asarray(batch.battery_w, dtype=float)
    grid = np.asarray(batch.grid_w, dtype=float)
    soc = np.asarray(batch.soc_percent, dtype=float)
    soc0 = np.asarray(batch.soc_initial_percent, dtype=float)

    n, t_steps = solar.shape
    dt_h = batch.dt_h

    capacity = float(env["usable_battery_capacity_wh"] or 0.0)
    capacity *= float(calibration.get("capacity_factor", 1.0))
    eta_c = float(np.clip(1.0 - float(calibration.get("charge_loss_factor", 0.0)), 0.5, 1.0))
    eta_d = float(np.clip(1.0 - float(calibration.get("discharge_loss_factor", 0.0)), 0.5, 1.0))

    # ---- per-check boolean masks (n, T) --------------------------------
    checks: dict[str, np.ndarray] = {}

    finite = (
        np.isfinite(solar) & np.isfinite(load) & np.isfinite(grid) & np.isfinite(soc)
    )
    checks["NON_FINITE"] = ~finite
    checks["SOLAR_NEGATIVE"] = solar < -1e-6
    checks["LOAD_NEGATIVE"] = load < -1e-6

    rated = env["rated_pv_w"]
    if rated is not None:
        checks["SOLAR_OVERSCALE"] = solar > policy.solar_scale_kappa * float(rated)
    ac_rating = env["certified_ac_pv_w"]
    if ac_rating is not None:
        checks["SOLAR_ABOVE_CERTIFIED_AC"] = solar > float(ac_rating)

    checks["SOC_BOUNDS"] = (soc < policy.soc_min - 1e-9) | (soc > policy.soc_max + 1e-9)

    if capacity > 0:
        soc_start = np.concatenate([soc0[:, None], soc[:, :-1]], axis=1)
        residual = balance_residual_wh(
            solar, load, grid, soc_start, soc, capacity, eta_c, eta_d, dt_h=dt_h
        )
        checks["ENERGY_BALANCE"] = residual > policy.energy_balance_tol_wh
        max_balance_error = residual.max(axis=1)
    else:
        residual = np.zeros_like(solar)
        max_balance_error = np.zeros(n)
        checks["ENERGY_BALANCE"] = np.zeros_like(solar, dtype=bool)

    max_charge = env["max_charge_w"]
    max_discharge = env["max_discharge_w"]
    if max_charge is not None:
        checks["BATTERY_CHARGE_LIMIT"] = battery < -policy.power_limit_tolerance * float(max_charge) - 1e-6
    if max_discharge is not None:
        checks["BATTERY_DISCHARGE_LIMIT"] = battery > policy.power_limit_tolerance * float(max_discharge) + 1e-6

    imp_limit = env["max_grid_import_w"]
    if imp_limit is not None:
        checks["GRID_IMPORT_LIMIT"] = grid > float(imp_limit) + 1e-6
    exp_limit = env["max_grid_export_w"]
    if exp_limit is not None:
        checks["GRID_EXPORT_LIMIT"] = grid < -float(exp_limit) - 1e-6

    if policy.solar_ramp_w_per_step is not None:
        d_solar = np.abs(np.diff(solar, axis=1, prepend=solar[:, :1]))
        checks["SOLAR_RAMP_LIMIT"] = d_solar > float(policy.solar_ramp_w_per_step) + 1e-6
    else:
        site_ramp = env["solar_ramp_w_per_step"]
        if site_ramp is not None:
            d_solar = np.abs(np.diff(solar, axis=1, prepend=solar[:, :1]))
            checks["SOLAR_RAMP_LIMIT"] = d_solar > float(site_ramp) + 1e-6
    if policy.load_ramp_w_per_step is not None:
        d_load = np.abs(np.diff(load, axis=1, prepend=load[:, :1]))
        checks["LOAD_RAMP_LIMIT"] = d_load > float(policy.load_ramp_w_per_step) + 1e-6

    # ---- fail-closed per-scenario judgement ----------------------------
    code_names = list(checks.keys())
    combined = np.zeros((n, t_steps), dtype=bool)
    for mask in checks.values():
        combined |= mask

    retained_idx: list[int] = []
    rejections: list[dict[str, Any]] = []
    constraint_counts: dict[str, int] = {}

    for i in range(n):
        failing = np.where(combined[i])[0]
        if failing.size == 0:
            retained_idx.append(i)
            continue
        failed_timestep = int(failing[0])
        codes = [name for name in code_names if bool(checks[name][i, failed_timestep])]
        for name in code_names:
            if bool(checks[name][i, failed_timestep]):
                constraint_counts[name] = constraint_counts.get(name, 0) + 1
        magnitude = _violation_magnitude(checks, i, failed_timestep, policy, residual)
        rejection = {
            "scenario_id": batch.scenario_ids[i],
            "scenario_index": i,
            "failed_timestep": failed_timestep,
            "constraint_codes": codes,
            "max_balance_error_wh": float(max_balance_error[i]),
            "violation_magnitude": float(magnitude),
            "scenario_content_hash": hash_obj(
                {
                    "solar": np.round(solar[i], 3).tolist(),
                    "load": np.round(load[i], 3).tolist(),
                    "grid": np.round(grid[i], 3).tolist(),
                    "soc": np.round(soc[i], 3).tolist(),
                }
            ),
        }
        rejections.append(rejection)

    survival = (len(retained_idx) / n) if n else 0.0
    result = TribunalResult(
        batch_id=batch.batch_id,
        n_proposed=n,
        retained_idx=retained_idx,
        rejections=rejections,
        survival_rate=survival,
        policy_version=policy.policy_version,
        solver_version=policy.solver_version,
        constraint_counts=constraint_counts,
    )

    if store is not None:
        _persist(store, batch, result, ctx)

    result.batch_result = {
        "batch_id": batch.batch_id,
        "n_proposed": n,
        "n_retained": result.n_retained,
        "n_rejected": result.n_rejected,
        "survival_rate": survival,
        "retained_scenario_ids": [batch.scenario_ids[i] for i in retained_idx],
        "tribunal_policy_version": policy.policy_version,
        "constraint_counts": constraint_counts,
    }
    return result


def _violation_magnitude(
    checks: dict[str, np.ndarray],
    i: int,
    t: int,
    policy: TribunalPolicy,
    residual: np.ndarray,
) -> float:
    """How badly the scenario violated the constraint (worst margin)."""
    worst = 0.0
    if bool(checks["ENERGY_BALANCE"][i, t]):
        worst = max(worst, float(residual[i, t]))
    if bool(checks["SOC_BOUNDS"][i, t]):
        worst = max(worst, 1.0)
    for name in (
        "SOLAR_OVERSCALE",
        "SOLAR_ABOVE_CERTIFIED_AC",
        "BATTERY_CHARGE_LIMIT",
        "BATTERY_DISCHARGE_LIMIT",
        "GRID_IMPORT_LIMIT",
        "GRID_EXPORT_LIMIT",
        "SOLAR_RAMP_LIMIT",
        "LOAD_RAMP_LIMIT",
    ):
        if name in checks and bool(checks[name][i, t]):
            worst = max(worst, 1.0)
    return worst


def _persist(store: Store, batch: ScenarioBatch, result: TribunalResult, ctx: Any) -> None:
    """Write the batch result and every rejection into the Echo Memory."""
    try:
        for rejection in result.rejections:
            row = {
                "rejection_id": new_id("rej"),
                "batch_id": batch.batch_id,
                "scenario_id": rejection["scenario_id"],
                "tenant_id": batch.tenant_id,
                "site_id": batch.site_id,
                "chariot_model_version": batch.model_version,
                "residual_version": batch.residual_version,
                "residual_weight_w": float(batch.residual_weight),
                "solver_version": result.solver_version,
                "tribunal_policy_version": result.policy_version,
                "failed_timestep": rejection["failed_timestep"],
                "constraint_codes": json.dumps(rejection["constraint_codes"]),
                "max_balance_error_wh": rejection["max_balance_error_wh"],
                "violation_magnitude": rejection["violation_magnitude"],
                "exception_fingerprint": hash_obj(
                    {
                        "codes": rejection["constraint_codes"],
                        "timestep": rejection["failed_timestep"],
                    }
                ),
                "lineage_json": json.dumps(batch.lineage, sort_keys=True, default=str),
                "scenario_content_hash": rejection["scenario_content_hash"],
                "created_at_utc": utc_now(),
            }
            store.insert("tribunal_rejections", row)
        store.upsert(
            "tribunal_batches",
            {
                "batch_id": batch.batch_id,
                "n_proposed": result.n_proposed,
                "n_retained": result.n_retained,
                "n_rejected": result.n_rejected,
                "survival_rate": result.survival_rate,
                "retained_scenario_ids": json.dumps(
                    [batch.scenario_ids[i] for i in result.retained_idx]
                ),
                "rejection_ids": json.dumps(
                    [r["scenario_id"] for r in result.rejections]
                ),
                "tribunal_policy_version": result.policy_version,
                "created_at_utc": utc_now(),
            },
        )
    except Exception as exc:  # pragma: no cover - durability guard
        raise RuntimeError(
            "Rejection Echo Memory write failed; the batch fails closed"
        ) from exc
