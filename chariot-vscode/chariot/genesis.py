"""Scenario Genesis Engine — trajectory fan for the Tribunal (PDF §4.9).

Modes:

* ``simple_monte_carlo``    — latent-state simulation with fixed parameters,
* ``hybrid_hierarchical``    — hierarchical parameter draws (default),
* ``reference_replay``       — replay of a stored posterior reference path.

For ``s = 1..S`` (default 500-1000):

    draw the base path from the CHARIOT model
    if the residual chamber is active:   tau = base + w * r_hat
    else:                                 tau = base
    attach scenario_id + lineage

The **entire** batch is sent to the Tribunal — Genesis never pre-drops an
"ugly" scenario.  That is the whole point: the Tribunal is the only judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from chariot.hashing import hash_obj, short_hash
from chariot.model import ChariotModel, clear_sky_series, simulate
from chariot.physics import BatterySpec, solve_flow

GENESIS_MODES = ("simple_monte_carlo", "hybrid_hierarchical", "reference_replay")


@dataclass
class ScenarioBatch:
    """A generated scenario ensemble awaiting certification."""

    batch_id: str
    tenant_id: str
    site_id: str
    origin_utc: str
    horizon_steps: int
    dt_h: float
    n_scenarios: int
    model_version: str
    residual_version: str | None
    residual_weight: float
    dataset_version: str
    solar_w: np.ndarray
    load_w: np.ndarray
    battery_w: np.ndarray
    grid_w: np.ndarray
    soc_percent: np.ndarray
    soc_initial_percent: np.ndarray
    timestamps: list[str]
    scenario_ids: list[str]
    lineage: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "tenant_id": self.tenant_id,
            "site_id": self.site_id,
            "origin_utc": self.origin_utc,
            "horizon_steps": self.horizon_steps,
            "dt_h": self.dt_h,
            "n_scenarios": self.n_scenarios,
            "model_version": self.model_version,
            "residual_version": self.residual_version,
            "residual_weight": self.residual_weight,
            "dataset_version": self.dataset_version,
            "content_hash": self.content_hash,
            "lineage": self.lineage,
        }


def _battery_from_context(ctx: Any, calibration: dict[str, float] | None = None) -> BatterySpec:
    """Build the certified battery spec from the site context + calibration.

    The *calibration version* is the authority for the loss factors: if the
    resolved calibration carries them, they override the site declaration so
    that the generator and the Tribunal solve the same energy balance.
    """
    meta = getattr(ctx, "metadata", {}) or {}
    calibration = calibration or {}
    capacity = getattr(ctx, "usable_battery_capacity_wh", None)
    if capacity is None:
        capacity = float(meta.get("usable_battery_capacity_wh", 0.0) or 0.0)
    eta_c = 1.0 - float(
        calibration.get("charge_loss_factor", meta.get("charge_loss_factor", 0.05)) or 0.0
    )
    eta_d = 1.0 - float(
        calibration.get("discharge_loss_factor", meta.get("discharge_loss_factor", 0.05)) or 0.0
    )
    max_charge = getattr(ctx, "max_charge_w", None)
    max_discharge = getattr(ctx, "max_discharge_w", None)
    if max_charge is None:
        max_charge = float(meta.get("max_charge_w", capacity) or capacity)
    if max_discharge is None:
        max_discharge = float(meta.get("max_discharge_w", capacity) or capacity)
    return BatterySpec(
        usable_capacity_wh=float(capacity),
        max_charge_w=float(max_charge),
        max_discharge_w=float(max_discharge),
        charge_efficiency=float(np.clip(eta_c, 0.5, 1.0)),
        discharge_efficiency=float(np.clip(eta_d, 0.5, 1.0)),
        soc_floor_pct=float(meta.get("soc_floor_pct", 5.0)),
        soc_ceiling_pct=float(meta.get("soc_ceiling_pct", 95.0)),
    )


def generate_batch(
    model: ChariotModel,
    site_id: str,
    future: pd.DataFrame,
    *,
    tenant_id: str,
    n_scenarios: int = 500,
    seed: int = 7,
    ctx: Any,
    calibration: dict[str, float] | None = None,
    residual_version: str | None = None,
    residual_weight: float = 0.0,
    residual_artifact: Any = None,
    state: dict[str, float] | None = None,
    mode: str = "hybrid_hierarchical",
    dt_h: float = 0.25,
) -> ScenarioBatch:
    """Generate a scenario batch and solve the physics for every path."""
    if mode not in GENESIS_MODES:
        raise ValueError(f"unknown genesis mode {mode!r}; known: {GENESIS_MODES}")

    frame = future.sort_values("timestamp_utc").reset_index(drop=True)
    battery = _battery_from_context(ctx, calibration)
    meta = getattr(ctx, "metadata", {}) or {}
    declared_load_sigma = float(meta.get("load_relative_sigma", 0.0) or 0.0)

    sim = simulate(
        model,
        site_id,
        frame,
        n_scenarios=n_scenarios,
        seed=seed,
        state=state,
        load_profile=None,
        declared_load_sigma=declared_load_sigma,
        dt_h=dt_h,
    )
    solar = sim["solar_w"]
    load = sim["load_w"]

    # -- residual augmentation (bounded, never bypasses the Tribunal) ------
    if residual_artifact is not None and residual_weight > 0.0:
        from chariot.residual import build_features, propose_paths

        clear = clear_sky_series(frame)
        clear = np.nan_to_num(clear, nan=0.0)
        expected_ratio = solar / np.maximum(clear, 1.0)[None, :]
        ctx_meta = (getattr(ctx, "metadata", {}) or {}) if ctx is not None else {}
        last_solar = float((state or {}).get("last_solar", 0.0))
        last_load = float((state or {}).get("last_load", 0.0))
        features, _ = build_features(
            frame,
            mu_solar=solar,
            mu_load=load,
            clear_sky=clear,
            expected_ratio=expected_ratio,
            last_solar=last_solar,
            last_load=last_load,
            calibration=calibration or {},
        )
        corrections = np.asarray(propose_paths(residual_artifact, features), dtype=float)
        corrections = corrections.reshape(solar.shape)
        solar = np.maximum(solar + float(residual_weight) * corrections, 0.0)

    # -- deterministic physics for every path ------------------------------
    n, t_steps = solar.shape
    battery_w = np.zeros_like(solar)
    grid_w = np.zeros_like(solar)
    soc = np.zeros_like(solar)
    soc0 = np.zeros(n, dtype=float)
    for i in range(n):
        flow = solve_flow(
            solar[i],
            load[i],
            soc_initial_pct=float(state.get("soc_percent", 50.0)) if state else 50.0,
            battery=battery,
            dt_h=dt_h,
        )
        battery_w[i] = flow["battery_w"]
        grid_w[i] = flow["grid_w"]
        soc[i] = flow["soc_percent"]
        soc0[i] = float(flow["soc_initial_percent"][0])

    timestamps = [str(ts) for ts in frame["timestamp_utc"]]
    lineage = {
        "model_version": model.model_version,
        "model_name": model.model_name,
        "dataset_version": model.dataset_version,
        "dataset_content_hash": model.dataset_content_hash,
        "residual_version": residual_version,
        "residual_weight": float(residual_weight),
        "calibration": calibration or {},
        "genesis_mode": mode,
        "site_context": ctx.to_dict() if hasattr(ctx, "to_dict") else {},
        "battery_spec": battery.to_dict(),
        "seed": seed,
    }
    content_hash = hash_obj(
        {
            "solar": np.round(solar, 3).tolist(),
            "load": np.round(load, 3).tolist(),
            "grid": np.round(grid_w, 3).tolist(),
            "soc": np.round(soc, 3).tolist(),
            "lineage": lineage,
        }
    )
    batch_id = f"batch_{short_hash(content_hash, 20)}"
    scenario_ids = [f"{batch_id}_s{i:04d}" for i in range(n)]

    return ScenarioBatch(
        batch_id=batch_id,
        tenant_id=tenant_id,
        site_id=site_id,
        origin_utc=timestamps[0] if timestamps else "",
        horizon_steps=int(t_steps),
        dt_h=dt_h,
        n_scenarios=int(n),
        model_version=model.model_version,
        residual_version=residual_version,
        residual_weight=float(residual_weight),
        dataset_version=model.dataset_version,
        solar_w=solar,
        load_w=load,
        battery_w=battery_w,
        grid_w=grid_w,
        soc_percent=soc,
        soc_initial_percent=soc0,
        timestamps=timestamps,
        scenario_ids=scenario_ids,
        lineage=lineage,
        content_hash=content_hash,
        extra={"sim": sim},
    )
