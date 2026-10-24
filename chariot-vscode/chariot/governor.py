"""Residual Stability Governor — the sole runtime authority for ``w`` (PDF §4.12).

Signals monitored:

    c   = E[|r_hat|] / E[|y_hat|]
    rho = corr(|r_hat|, reject_indicator)
    per-regime reject rates
    residual-on vs. residual-off survival delta

State machine: ``active`` <-> ``attenuated`` <-> ``quarantined``.

Genesis always reads ``get_effective_w()``.  The Dual Feedback Cortex cannot
bypass the Governor to manipulate production weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chariot.config import GOVERNOR_POLICY, RESIDUAL_POLICY, GovernorPolicy
from chariot.store import Store, new_id, utc_now

STATE_ACTIVE = "active"
STATE_ATTENUATED = "attenuated"
STATE_QUARANTINED = "quarantined"


@dataclass
class GovernorState:
    """Current runtime state of the Residual Stability Governor."""

    state: str = STATE_ACTIVE
    w: float = 0.0
    w_max: float = RESIDUAL_POLICY.w_max
    breaches: int = 0
    healthy_streak: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "w": self.w,
            "w_max": self.w_max,
            "breaches": self.breaches,
            "healthy_streak": self.healthy_streak,
            "policy_version": GOVERNOR_POLICY.policy_version,
            "history": list(self.history[-20:]),
        }


class ResidualStabilityGovernor:
    """Runtime authority for the effective residual weight."""

    def __init__(
        self,
        store: Store | None = None,
        policy: GovernorPolicy = GOVERNOR_POLICY,
        w_max: float | None = None,
    ) -> None:
        self.store = store
        self.policy = policy
        self.state = GovernorState(w_max=w_max or policy.w_max)
        if store is not None:
            self._load_events()

    # -- persistence --------------------------------------------------------
    def _load_events(self) -> None:
        rows = self.store.query(
            "SELECT * FROM governor_events ORDER BY created_at_utc DESC LIMIT 50"
        )
        if not rows:
            return
        latest = rows[0]
        self.state.state = latest["state"]
        self.state.w = float(latest["w"])

    def _record(self, batch_id: str, action: str, signals: dict[str, Any]) -> None:
        event = {
            "event_id": new_id("gov"),
            "batch_id": batch_id,
            "previous_state": self.state.state,
            "state": self.state.state,
            "w": float(self.state.w),
            "c": float(signals.get("c", 0.0)),
            "rho": float(signals.get("rho", 0.0)),
            "signals_json": _dumps(signals),
            "action": action,
            "created_at_utc": utc_now(),
        }
        self.state.history.append(event)
        if self.store is not None:
            self.store.insert("governor_events", event)

    # -- API ----------------------------------------------------------------
    def get_effective_w(self) -> float:
        """The weight Genesis is allowed to use.  Quarantine forces ``w = 0``."""
        if self.state.state == STATE_QUARANTINED:
            return 0.0
        return float(np.clip(self.state.w, 0.0, self.state.w_max))

    def status(self) -> dict[str, Any]:
        return {
            **self.state.to_dict(),
            "effective_w": self.get_effective_w(),
        }

    def set_weight(self, w: float, actor: str = "system") -> dict[str, Any]:
        """Audited force-set of the residual weight."""
        w = float(np.clip(w, 0.0, self.state.w_max))
        self.state.w = w
        if w > 0 and self.state.state == STATE_QUARANTINED:
            self.state.state = STATE_ACTIVE
        self.state.healthy_streak = 0
        self._record("manual", "force_set_weight", {"w": w, "actor": actor})
        return self.status()

    def quarantine(self, actor: str = "system", reason: str = "manual") -> dict[str, Any]:
        """Quarantine the residual chamber (``w = 0``)."""
        previous = self.state.state
        self.state.state = STATE_QUARANTINED
        self.state.w = 0.0
        self.state.breaches = self.policy.breach_threshold
        self._record("manual", "quarantine", {"actor": actor, "reason": reason, "previous": previous})
        return self.status()

    def release(self, actor: str = "system", reason: str = "manual") -> dict[str, Any]:
        """Release from quarantine back to an attenuated state."""
        previous = self.state.state
        self.state.state = STATE_ATTENUATED
        self.state.healthy_streak = 0
        self._record("manual", "release", {"actor": actor, "reason": reason, "previous": previous})
        return self.status()

    def evaluate(
        self,
        signals: dict[str, Any],
        *,
        batch_id: str = "manual",
    ) -> dict[str, Any]:
        """Evaluate the governor signals and move the state machine."""
        c = float(signals.get("c", 0.0))
        rho = float(signals.get("rho", 0.0))
        survival_delta = float(signals.get("survival_delta", 0.0))
        survival = float(signals.get("survival_rate", 1.0))
        regime_rejects = signals.get("regime_reject_rates", {}) or {}

        breach_reasons: list[str] = []
        if c > self.policy.c_max:
            breach_reasons.append(f"contribution ratio c={c:.3f} > c_max {self.policy.c_max:.2f}")
        if rho > self.policy.rho_max:
            breach_reasons.append(f"reject correlation rho={rho:.3f} > rho_max {self.policy.rho_max:.2f}")
        if survival < self.policy.survival_floor:
            breach_reasons.append(
                f"survival {survival:.3f} < floor {self.policy.survival_floor:.2f}"
            )
        if survival_delta < self.policy.survival_delta_min:
            breach_reasons.append(
                f"survival delta {survival_delta:.3f} < {self.policy.survival_delta_min:.2f}"
            )
        if regime_rejects and max(regime_rejects.values()) > 0.25:
            worst = max(regime_rejects, key=lambda k: regime_rejects[k])
            breach_reasons.append(f"regime {worst} reject rate {regime_rejects[worst]:.3f} > 0.25")

        previous = self.state.state
        action = "hold"

        if breach_reasons:
            self.state.breaches += 1
            self.state.healthy_streak = 0
            if self.state.breaches >= self.policy.breach_threshold:
                self.state.state = STATE_QUARANTINED
                self.state.w = 0.0
                action = "quarantine"
            else:
                self.state.state = STATE_ATTENUATED
                self.state.w = float(np.clip(self.state.w * self.policy.beta, 0.0, self.state.w_max))
                action = "attenuate"
        else:
            self.state.healthy_streak += 1
            self.state.breaches = 0
            if self.state.state == STATE_QUARANTINED:
                action = "hold_quarantined"
            elif self.state.healthy_streak >= self.policy.recovery_threshold:
                self.state.state = STATE_ACTIVE
                self.state.w = float(np.clip(self.state.w, 0.0, self.state.w_max))
                action = "restore"
            else:
                action = "healthy"

        payload = {
            "c": c,
            "rho": rho,
            "survival_rate": survival,
            "survival_delta": survival_delta,
            "regime_reject_rates": regime_rejects,
            "breach_reasons": breach_reasons,
            "previous_state": previous,
            "action": action,
        }
        self._record(batch_id, action, payload)
        return {**self.status(), "evaluation": payload}


def governor_signals(
    residual_magnitude: np.ndarray,
    target_magnitude: np.ndarray,
    reject_indicator: np.ndarray | None = None,
    *,
    survival_rate: float = 1.0,
    survival_delta: float = 0.0,
    regime_reject_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute the governor's monitored signals."""
    residual_magnitude = np.asarray(residual_magnitude, dtype=float)
    target_magnitude = np.asarray(target_magnitude, dtype=float)
    c = float(
        np.mean(np.abs(residual_magnitude))
        / max(np.mean(np.abs(target_magnitude)), 1e-9)
    )
    rho = 0.0
    if reject_indicator is not None and len(reject_indicator) == len(residual_magnitude):
        r = np.asarray(reject_indicator, dtype=float)
        if np.std(r) > 0 and np.std(residual_magnitude) > 0:
            rho = float(np.corrcoef(np.abs(residual_magnitude), r)[0, 1])
    return {
        "c": c,
        "rho": rho,
        "survival_rate": float(survival_rate),
        "survival_delta": float(survival_delta),
        "regime_reject_rates": regime_reject_rates or {},
    }


def _dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, sort_keys=True, default=str)
