"""Advisory API — multi-horizon, multi-role, advisory-only publication (PDF §4.14).

Role modes:

* ``research``       — short horizon, full transparency, lineage, residual context,
* ``grid_engineer``  — operational horizon, risk events, actionable confidence,
* ``investor``       — long horizon, planning aggregates, site comparison context.

Every response carries ``advisory_only: true``, the confidence status, and the
full lineage (batch, model, residual, dataset, capsules, policies).

Abstain behaviour: empty series, explicit reasons, HTTP 200 with an abstained
payload.  A P50 is **never** fabricated.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from chariot.config import (
    HORIZON_HOURS,
    HORIZONS,
    ROLE_DEFAULT_HORIZON,
    ROLE_MODES,
)
from chariot.confidence import ConfidenceReport
from chariot.genesis import ScenarioBatch

ADVISORY_ONLY = True


def _bands(values: np.ndarray) -> dict[str, list[float]]:
    """P10/P50/P90 bands from the RETAINED scenarios only."""
    if values.size == 0:
        return {"p10": [], "p50": [], "p90": []}
    return {
        "p10": [round(float(v), 3) for v in np.quantile(values, 0.10, axis=0)],
        "p50": [round(float(v), 3) for v in np.quantile(values, 0.50, axis=0)],
        "p90": [round(float(v), 3) for v in np.quantile(values, 0.90, axis=0)],
    }


def _event_probability(values: np.ndarray, predicate) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(predicate(values)))


#: The event keys that describe the *hybrid* side of the plant.  They are
#: only meaningful when the load and the state of charge are measured.
HYBRID_EVENT_KEYS: tuple[str, ...] = (
    "p_soc_below_20_within_horizon",
    "p_soc_above_90_within_horizon",
    "p_grid_import_limit_exceeded",
    "p_grid_export_limit_exceeded",
)


def hybrid_channels_measured(ctx: Any) -> bool:
    """Whether the site has *measured* load / state of charge.

    A declared site card (a load profile, a battery size, a starting SOC)
    is an engineering assumption, not a measurement.  On a solar-only site
    the hybrid event probabilities would be numbers derived from assumptions,
    so they must not be published as if they were observations.
    """
    meta = (getattr(ctx, "metadata", {}) or {}) if ctx is not None else {}
    return bool(meta.get("load_observed")) and bool(meta.get("soc_observed"))


def build_events(
    batch: ScenarioBatch,
    retained_idx: Sequence[int],
    *,
    ctx: Any = None,
    horizon_steps: int | None = None,
    solar_only: bool = True,
) -> dict[str, Any]:
    """Empirical event probabilities computed on the retained set.

    In solar-only mode the hybrid (SOC / grid) event probabilities are
    nulled and the payload is marked ``hybrid_assumed``: the numbers exist
    in the simulation, but they come from a declared site card, not from a
    measurement, so they are not published as skill.
    """
    idx = np.asarray(list(retained_idx), dtype=int)
    if idx.size == 0:
        return {
            "p_soc_below_20_within_horizon": 0.0,
            "p_soc_above_90_within_horizon": 0.0,
            "p_grid_import_limit_exceeded": 0.0,
            "p_grid_export_limit_exceeded": 0.0,
            "p_pv_above_certified_ac": 0.0,
            "n_retained": 0,
        }
    steps = horizon_steps or batch.horizon_steps
    soc = np.asarray(batch.soc_percent[idx], dtype=float)[:, :steps]
    grid = np.asarray(batch.grid_w[idx], dtype=float)[:, :steps]
    solar = np.asarray(batch.solar_w[idx], dtype=float)[:, :steps]

    meta = (getattr(ctx, "metadata", {}) or {}) if ctx is not None else {}
    imp_limit = meta.get("max_grid_import_w")
    exp_limit = meta.get("max_grid_export_w")
    ac_rating = meta.get("certified_ac_pv_w")

    events: dict[str, Any] = {
        "p_soc_below_20_within_horizon": _event_probability(soc, lambda v: np.min(v, axis=1) < 20.0),
        "p_soc_above_90_within_horizon": _event_probability(soc, lambda v: np.max(v, axis=1) > 90.0),
        "n_retained": int(idx.size),
        "horizon_steps": int(steps),
    }
    if imp_limit is not None:
        events["p_grid_import_limit_exceeded"] = _event_probability(
            grid, lambda v: np.max(v, axis=1) > float(imp_limit)
        )
        events["grid_import_limit_w"] = float(imp_limit)
    if exp_limit is not None:
        events["p_grid_export_limit_exceeded"] = _event_probability(
            grid, lambda v: np.min(v, axis=1) < -float(exp_limit)
        )
        events["grid_export_limit_w"] = float(exp_limit)
    if ac_rating is not None:
        events["p_pv_above_certified_ac"] = _event_probability(
            solar, lambda v: np.max(v, axis=1) > float(ac_rating)
        )
        events["certified_ac_pv_w"] = float(ac_rating)

    if solar_only and not hybrid_channels_measured(ctx):
        # The hybrid probabilities are assumption-driven on this site: they
        # are removed from the payload and the payload says so.
        for key in HYBRID_EVENT_KEYS:
            if key in events:
                events[key] = None
        events["hybrid_assumed"] = True
        events["hybrid_note"] = (
            "SOC and grid event probabilities are nulled: this site has no "
            "measured load or state of charge, so the declared site card is an "
            "assumption and not an observation."
        )
    return events


def build_forecast(
    batch: ScenarioBatch,
    retained_idx: Sequence[int],
    report: ConfidenceReport,
    *,
    role_mode: str,
    horizon: str | None = None,
    lineage: dict[str, Any] | None = None,
    ctx: Any = None,
    horizon_steps: int | None = None,
    baselines: list[dict[str, Any]] | None = None,
    rejection_sample: list[dict[str, Any]] | None = None,
    solar_only: bool = True,
) -> dict[str, Any]:
    """Build one role-mode advisory payload."""
    if role_mode not in ROLE_MODES:
        raise ValueError(f"unknown role mode {role_mode!r}; known: {list(ROLE_MODES)}")

    horizon = horizon or ROLE_DEFAULT_HORIZON[role_mode]
    idx = np.asarray(list(retained_idx), dtype=int)
    certified_steps = batch.horizon_steps
    requested = horizon_steps or HORIZONS.get(horizon, certified_steps)
    if requested > certified_steps:
        requested = certified_steps
    steps = int(min(requested, certified_steps))

    common: dict[str, Any] = {
        "advisory_only": ADVISORY_ONLY,
        "not_a_controller": True,
        "role_mode": role_mode,
        "horizon": horizon,
        "horizon_hours": HORIZON_HOURS.get(horizon),
        "horizon_steps": steps,
        "certified_trajectory_steps": certified_steps,
        "status": report.status,
        "confidence": {
            "C": round(float(report.confidence), 4),
            "s_survival": round(float(report.survival_rate), 4),
            "u_uncertainty": round(float(report.uncertainty), 4),
            "policy_version": report.policy_version,
            "reasons": list(report.reasons),
        },
        "survival_rate": round(float(report.survival_rate), 4),
        "n_proposed": report.n_proposed,
        "n_retained": report.n_retained,
        "n_rejected": report.n_rejected,
        "lineage": lineage or {},
        "timestamps": batch.timestamps[:steps],
    }

    if report.status == "abstained" or idx.size == 0:
        abstain = {
            **common,
            "abstained": True,
            "message": (
                "Advisory abstained: physics survival too low. "
                "No certified quantiles are published."
            ),
            "solar_w": {"p10": [], "p50": [], "p90": []},
            "load_w": {"p10": [], "p50": [], "p90": []},
            "soc_percent": {"p10": [], "p50": [], "p90": []},
            "grid_w": {"p10": [], "p50": [], "p90": []},
            "events": {},
        }
        if role_mode == "research":
            abstain["rejection_sample"] = (rejection_sample or [])[:12]
            abstain["baselines"] = baselines or []
        return abstain

    solar = np.asarray(batch.solar_w[idx], dtype=float)[:, :steps]
    load = np.asarray(batch.load_w[idx], dtype=float)[:, :steps]
    soc = np.asarray(batch.soc_percent[idx], dtype=float)[:, :steps]
    grid = np.asarray(batch.grid_w[idx], dtype=float)[:, :steps]

    events = build_events(
        batch, idx, ctx=ctx, horizon_steps=steps, solar_only=solar_only
    )

    payload: dict[str, Any] = {
        **common,
        "abstained": False,
        "solar_w": _bands(solar),
        "load_w": _bands(load),
        "soc_percent": _bands(soc),
        "grid_w": _bands(grid),
        "events": events,
    }

    if role_mode == "grid_engineer":
        import_proxy = np.maximum(0.0, load - solar)
        payload.update(
            {
                "risk_summary": {
                    "min_soc_p10_pct": round(
                        float(np.min(np.quantile(soc, 0.10, axis=0))), 2
                    ),
                    "mean_import_proxy_w_p50": round(
                        float(np.median(import_proxy.mean(axis=1))), 2
                    ),
                    "max_import_w_p90": round(
                        float(np.quantile(np.max(grid, axis=1), 0.90)), 2
                    ),
                    "max_export_w_p90": round(
                        float(np.quantile(-np.min(grid, axis=1), 0.90)), 2
                    ),
                    "note": "grid stress is advisory; no dispatch setpoints are issued",
                }
            }
        )
    elif role_mode == "research":
        payload.update(
            {
                "diagnostics": {
                    "regime": report.regime,
                    "signal_widths": report.signal_widths,
                    "residual_context": report.residual_context,
                },
                "baselines": baselines or [],
                "rejection_sample": (rejection_sample or [])[:12],
                "notes": [
                    "Quantiles are computed from Tribunal-retained scenarios only.",
                    "The split is time-ordered (60/20/20); no future data enters training.",
                    "Survival dominates sharpness: low survival lowers confidence.",
                ],
            }
        )
    else:  # investor / government
        yield_kwh = float(np.mean(solar.sum(axis=1)) * batch.dt_h / 1000.0)
        p10_energy = float(np.quantile(solar.sum(axis=1), 0.10) * batch.dt_h / 1000.0)
        p90_energy = float(np.quantile(solar.sum(axis=1), 0.90) * batch.dt_h / 1000.0)
        payload.update(
            {
                "planning_aggregates": {
                    "solar_yield_p10_kwh": round(p10_energy, 3),
                    "solar_yield_p50_kwh": round(yield_kwh, 3),
                    "solar_yield_p90_kwh": round(p90_energy, 3),
                    "yield_band_note": "certified scenario band, not a financial guarantee",
                },
                "site_confidence": report.status,
                "not_a_financial_guarantee": True,
            }
        )
    return payload


def build_all(
    batch: ScenarioBatch,
    retained_idx: Sequence[int],
    report: ConfidenceReport,
    *,
    lineage: dict[str, Any] | None = None,
    ctx: Any = None,
    horizon_steps: int | None = None,
    baselines: list[dict[str, Any]] | None = None,
    rejection_sample: list[dict[str, Any]] | None = None,
    solar_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build all three role-mode advisories from one certified batch."""
    return {
        role: build_forecast(
            batch,
            retained_idx,
            report,
            role_mode=role,
            lineage=lineage,
            ctx=ctx,
            horizon_steps=horizon_steps,
            baselines=baselines,
            rejection_sample=rejection_sample,
            solar_only=solar_only,
        )
        for role in ROLE_MODES
    }
