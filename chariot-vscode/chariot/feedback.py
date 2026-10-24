"""Dual Feedback Cortex — structured learning signals from the Echo memory.

PDF §4.11.

Path A — CHARIOT model:

* sample re-weighting near recurrent failures,
* physics-informed penalty terms,
* dataset window proposals,
* lands as new train jobs -> new ``model_version`` (no silent in-place hack).

Path B — Residual:

* quarantine requests to the Governor,
* healthy -> maintain ``w``,
* breach -> attenuate ``w <- beta * w``,
* severe / persistent instability -> quarantine (``w = 0``),
* recovery -> gradual restore under policy,
* human -> audited force-set / quarantine / release.

Promotion discipline: new versions are only deployed if the held-out survival
does not degrade, the residual ``w`` stays within the ceiling, all diagnostics
pass and the lineage remains complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chariot.governor import ResidualStabilityGovernor, governor_signals
from chariot.store import Store, new_id, utc_now


@dataclass
class FeedbackJob:
    """A structured learning signal produced by the cortex."""

    job_id: str
    batch_id: str
    path: str  # "model" | "residual"
    payload: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "batch_id": self.batch_id,
            "path": self.path,
            "payload": self.payload,
            "created_at_utc": self.created_at_utc,
        }


def analyse_rejections(
    store: Store, batch_id: str, limit: int = 500
) -> dict[str, Any]:
    """Summarise the Rejection Echo Memory for one batch."""
    rows = store.rejections_for_batch(batch_id)[:limit]
    code_counts: dict[str, int] = {}
    timestep_counts: dict[str, int] = {}
    for row in rows:
        import json

        for code in json.loads(row["constraint_codes"] or "[]"):
            code_counts[code] = code_counts.get(code, 0) + 1
        key = str(row["failed_timestep"])
        timestep_counts[key] = timestep_counts.get(key, 0) + 1
    return {
        "n_rejections": len(rows),
        "constraint_code_counts": dict(sorted(code_counts.items(), key=lambda kv: -kv[1])),
        "failed_timestep_counts": dict(
            sorted(timestep_counts.items(), key=lambda kv: -kv[1])[:20]
        ),
        "max_balance_error_wh": max((row["max_balance_error_wh"] for row in rows), default=0.0),
    }


def run_cortex(
    store: Store,
    *,
    batch_id: str,
    governor: ResidualStabilityGovernor,
    survival_rate: float,
    survival_delta: float = 0.0,
    residual_magnitude: list[float] | None = None,
    target_magnitude: list[float] | None = None,
    reject_indicator: list[float] | None = None,
) -> dict[str, Any]:
    """Run the Dual Feedback Cortex over one batch.

    Produces a *model* learning job and a *residual* learning job, and lets
    the Governor act on the residual signals.
    """
    rejection_summary = analyse_rejections(store, batch_id)

    # -- Path A: CHARIOT model learning signals ----------------------------
    model_job = FeedbackJob(
        job_id=new_id("fbjob"),
        batch_id=batch_id,
        path="model",
        payload={
            "recommendation": "retrain_with_reweighting",
            "survival_rate": survival_rate,
            "rejection_summary": rejection_summary,
            "sample_reweighting": _reweighting(rejection_summary),
            "penalty_terms": _penalties(rejection_summary),
            "dataset_window_proposal": _window_proposal(rejection_summary),
            "promotion_discipline": {
                "held_out_survival_must_not_degrade": True,
                "lineage_must_be_complete": True,
                "diagnostics_must_pass": True,
            },
        },
        created_at_utc=utc_now(),
    )
    store.insert(
        "feedback_jobs",
        {
            "job_id": model_job.job_id,
            "batch_id": batch_id,
            "path": model_job.path,
            "payload_json": _dumps(model_job.payload),
            "created_at_utc": model_job.created_at_utc,
        },
    )

    # -- Path B: residual learning signals + Governor action ----------------
    if residual_magnitude and target_magnitude:
        signals = governor_signals(
            np.asarray(residual_magnitude, dtype=float),
            np.asarray(target_magnitude, dtype=float),
            np.asarray(reject_indicator, dtype=float) if reject_indicator else None,
            survival_rate=survival_rate,
            survival_delta=survival_delta,
            regime_reject_rates=_regime_rates(rejection_summary),
        )
        governor_result = governor.evaluate(signals, batch_id=batch_id)
    else:
        signals = {}
        governor_result = governor.status()

    residual_job = FeedbackJob(
        job_id=new_id("fbjob"),
        batch_id=batch_id,
        path="residual",
        payload={
            "signals": signals,
            "governor": governor_result,
            "action_taken": governor_result.get("evaluation", {}).get("action", "hold"),
            "recommendation": _residual_recommendation(governor_result),
        },
        created_at_utc=utc_now(),
    )
    store.insert(
        "feedback_jobs",
        {
            "job_id": residual_job.job_id,
            "batch_id": batch_id,
            "path": residual_job.path,
            "payload_json": _dumps(residual_job.payload),
            "created_at_utc": residual_job.created_at_utc,
        },
    )
    store.audit(
        "feedback.cortex.run",
        "batch",
        batch_id,
        payload={"jobs": [model_job.job_id, residual_job.job_id]},
    )
    return {
        "batch_id": batch_id,
        "model_job": model_job.to_dict(),
        "residual_job": residual_job.to_dict(),
        "governor": governor_result,
        "rejection_summary": rejection_summary,
    }


def _reweighting(summary: dict[str, Any]) -> dict[str, float]:
    """Sample re-weighting factors derived from recurrent failure modes."""
    total = max(summary.get("n_rejections", 0), 1)
    counts = summary.get("constraint_code_counts", {})
    weights = {}
    for code, count in counts.items():
        weights[code] = float(1.0 + count / total)
    return weights


def _penalties(summary: dict[str, Any]) -> list[str]:
    """Physics-informed penalty terms to add to the model objective."""
    penalties: list[str] = []
    counts = summary.get("constraint_code_counts", {})
    if counts.get("ENERGY_BALANCE"):
        penalties.append("penalise_energy_balance_residual")
    if counts.get("SOC_BOUNDS"):
        penalties.append("penalise_soc_envelope_excursion")
    if counts.get("SOLAR_OVERSCALE") or counts.get("SOLAR_ABOVE_CERTIFIED_AC"):
        penalties.append("penalise_pv_scale_overrun")
    if counts.get("GRID_IMPORT_LIMIT") or counts.get("GRID_EXPORT_LIMIT"):
        penalties.append("penalise_interconnection_overrun")
    if counts.get("SOLAR_RAMP_LIMIT"):
        penalties.append("penalise_ramp_overrun")
    return penalties


def _window_proposal(summary: dict[str, Any]) -> dict[str, Any]:
    """Propose a dataset window that emphasises the failure region."""
    counts = summary.get("failed_timestep_counts", {})
    if not counts:
        return {"action": "keep_window"}
    worst = max(counts, key=lambda k: counts[k])
    return {
        "action": "propose_window",
        "hotspot_index": int(worst),
        "rejections_at_hotspot": counts[worst],
        "rationale": "concentrate training samples near the recurrent failure timestep",
    }


def _regime_rates(summary: dict[str, Any]) -> dict[str, float]:
    counts = summary.get("constraint_code_counts", {})
    total = max(summary.get("n_rejections", 0), 1)
    return {code: float(count / total) for code, count in counts.items()}


def _residual_recommendation(governor_result: dict[str, Any]) -> str:
    action = governor_result.get("evaluation", {}).get("action", "hold")
    return {
        "attenuate": "residual_attenuated_by_governor",
        "quarantine": "residual_quarantined_by_governor",
        "restore": "residual_weight_restored",
        "healthy": "residual_weight_maintained",
        "hold_quarantined": "residual_stays_quarantined",
        "hold": "residual_weight_maintained",
    }.get(action, "residual_weight_maintained")


def _dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, sort_keys=True, default=str)
