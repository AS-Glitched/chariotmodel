"""Physics-Survival Confidence Engine (PDF §4.13).

    s = N_retained / N_proposed
    u = aggregated normalized P90-P10 width (RETAINED scenarios only)
    C = g(1 - u) * s            (default g = identity)

Status policy (versioned thresholds):

* ``s < s_min``                        -> ``abstained``
* ``C < C_low``                        -> ``reduced_confidence``
* ``u`` high with ``s < 1``            -> ``reduced_confidence``
* otherwise                            -> ``eligible_for_human_review``

Survival dominates sharpness: the residual may not inflate confidence while
survival ``s`` drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chariot.config import CONFIDENCE_POLICY, ConfidencePolicy
from chariot.genesis import ScenarioBatch
from chariot.store import Store, new_id, utc_now

STATUS_ELIGIBLE = "eligible_for_human_review"
STATUS_REDUCED = "reduced_confidence"
STATUS_ABSTAINED = "abstained"


@dataclass
class ConfidenceReport:
    """Joint statistical-uncertainty / physics-survival confidence report."""

    report_id: str
    batch_id: str
    status: str
    survival_rate: float
    uncertainty: float
    confidence: float
    n_proposed: int
    n_retained: int
    n_rejected: int
    policy_version: str
    reasons: list[str] = field(default_factory=list)
    regime: dict[str, Any] = field(default_factory=dict)
    residual_context: dict[str, Any] = field(default_factory=dict)
    signal_widths: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "batch_id": self.batch_id,
            "status": self.status,
            "survival_rate": self.survival_rate,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "n_proposed": self.n_proposed,
            "n_retained": self.n_retained,
            "n_rejected": self.n_rejected,
            "policy_version": self.policy_version,
            "reasons": list(self.reasons),
            "regime": dict(self.regime),
            "residual_context": dict(self.residual_context),
            "signal_widths": dict(self.signal_widths),
        }


def normalized_width(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray, floor: float) -> float:
    """Aggregate the P90-P10 band, normalized by the signal magnitude."""
    p10 = np.asarray(p10, dtype=float)
    p50 = np.asarray(p50, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    if p10.size == 0:
        return 1.0
    width = float(np.mean(p90 - p10))
    scale = float(np.mean(np.abs(p50))) + floor
    return float(width / max(scale, floor))


def evaluate(
    batch: ScenarioBatch,
    retained_idx: list[int],
    *,
    survival_rate: float,
    policy: ConfidencePolicy = CONFIDENCE_POLICY,
    residual_context: dict[str, Any] | None = None,
    store: Store | None = None,
) -> ConfidenceReport:
    """Evaluate the joint confidence metric for a certified batch."""
    idx = np.asarray(retained_idx, dtype=int)
    reasons: list[str] = []
    widths: dict[str, float] = {}

    if idx.size == 0:
        u = 1.0
        widths = {"solar_w": 1.0, "load_w": 1.0, "soc_percent": 1.0}
    else:
        u_parts: list[float] = []
        for name, arr, floor in (
            ("solar_w", batch.solar_w, 1000.0),
            ("load_w", batch.load_w, 500.0),
            ("soc_percent", batch.soc_percent, 1.0),
        ):
            values = np.asarray(arr[idx], dtype=float)
            p10 = np.quantile(values, 0.10, axis=0)
            p50 = np.quantile(values, 0.50, axis=0)
            p90 = np.quantile(values, 0.90, axis=0)
            width = normalized_width(p10, p50, p90, floor)
            widths[name] = float(width)
            u_parts.append(width)
        u = float(np.mean(u_parts))

    u = float(np.clip(u, 0.0, 10.0))
    if policy.sharpness == "identity":
        c = (1.0 - u) * survival_rate
    else:
        c = survival_rate
    c = float(np.clip(c, 0.0, 1.0))

    if survival_rate < policy.s_min:
        status = STATUS_ABSTAINED
        reasons.append(
            f"survival_rate {survival_rate:.3f} < s_min {policy.s_min:.2f}: physics survival too low"
        )
    elif c < policy.c_low:
        status = STATUS_REDUCED
        reasons.append(f"confidence {c:.3f} < C_low {policy.c_low:.2f}")
    elif u > policy.u_high and survival_rate < 1.0:
        status = STATUS_REDUCED
        reasons.append(
            f"uncertainty {u:.3f} > u_high {policy.u_high:.2f} with survival {survival_rate:.3f} < 1"
        )
    else:
        status = STATUS_ELIGIBLE

    report = ConfidenceReport(
        report_id=new_id("conf"),
        batch_id=batch.batch_id,
        status=status,
        survival_rate=float(survival_rate),
        uncertainty=float(u),
        confidence=float(c),
        n_proposed=batch.n_scenarios,
        n_retained=int(idx.size),
        n_rejected=int(batch.n_scenarios - idx.size),
        policy_version=policy.policy_version,
        reasons=reasons,
        regime=_regime_report(batch, idx),
        residual_context=residual_context or {},
        signal_widths=widths,
    )
    if store is not None:
        store.insert(
            "confidence_reports",
            {
                "report_id": report.report_id,
                "batch_id": batch.batch_id,
                "status": report.status,
                "survival_rate": report.survival_rate,
                "uncertainty": report.uncertainty,
                "confidence": report.confidence,
                "policy_version": report.policy_version,
                "report_json": _dumps(report.to_dict()),
                "created_at_utc": utc_now(),
            },
        )
    return report


def _regime_report(batch: ScenarioBatch, idx: np.ndarray) -> dict[str, Any]:
    """Regime-wise survival and confidence (daylight / night)."""
    if idx.size == 0:
        return {}
    clear = batch.extra.get("clear_sky") if batch.extra else None
    out: dict[str, Any] = {}
    if clear is None:
        return out
    daylight = np.asarray(clear, dtype=float) > 1.0
    for name, mask in (("daylight", daylight), ("night", ~daylight)):
        if not mask.any():
            continue
        out[name] = {
            "n_steps": int(mask.sum()),
            "retained_fraction": float(np.mean(idx < len(idx))),
        }
    return out


def _dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, sort_keys=True, default=str)
