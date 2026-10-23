"""Admissibility Pre-Gate — the first physics admission controller (PDF §4.2).

The gate classifies every incoming record as one of

* ``ADMISSIBLE``
* ``ADMISSIBLE_WITH_WARNINGS``
* ``REJECTED``
* ``INSUFFICIENT_DATA``

Only the first two continue into the River Flow fabric.  Every evaluation is
logged immutably with an ``input_fingerprint`` and the ``rule_set_version`` so
a decision can be replayed byte-for-byte later.

Hard rules (unconditional):

``MANDATORY_FIELDS``
    Required fields present and non-null.
``SOLAR_NON_NEGATIVE``
    ``solar_w >= 0``.
``SOC_BOUNDS``
    ``0 <= soc_percent <= 100``.
``POWER_FINITE``
    All power channels finite.
``SOLAR_SCALE_CEILING``
    ``solar_w <= rated_pv_w * kappa`` when the rating is known.

Soft rules (merged into ``quality_flags``):

``ENERGY_BALANCE``  ``NIGHT_SOLAR``  ``SOC_RATE_LIMIT``
``CLEAR_SKY_MISSING``  ``WEATHER_CADENCE``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from chariot.config import ADMISSIBILITY_POLICY, AdmissibilityPolicy, SiteContext
from chariot.hashing import hash_obj

#: Feature-set specific mandatory field sets.
MANDATORY_FIELDS: dict[str, tuple[str, ...]] = {
    "solar_only_v1": ("timestamp_utc", "solar_w", "clear_sky_w"),
    "physics_rich_v1": (
        "timestamp_utc",
        "solar_w",
        "clear_sky_w",
        "load_w",
        "soc_percent",
    ),
}


@dataclass
class Decision:
    """One immutable admissibility decision."""

    decision: str
    hard_failures: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    rule_set_version: str = ADMISSIBILITY_POLICY.rule_set_version
    input_fingerprint: str = ""

    @property
    def admitted(self) -> bool:
        return self.decision in ("ADMISSIBLE", "ADMISSIBLE_WITH_WARNINGS")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "hard_failures": list(self.hard_failures),
            "soft_warnings": list(self.soft_warnings),
            "rule_set_version": self.rule_set_version,
            "input_fingerprint": self.input_fingerprint,
        }


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def evaluate_record(
    record: dict[str, Any],
    ctx: SiteContext | None = None,
    policy: AdmissibilityPolicy = ADMISSIBILITY_POLICY,
    feature_set: str = "solar_only_v1",
    night_mask: bool | None = None,
) -> Decision:
    """Evaluate a single record against the hard/soft rule families.

    ``night_mask`` may be supplied by a caller that knows the local solar
    geometry; when omitted the policy's UTC night window is used.
    """
    hard: list[str] = []
    soft: list[str] = []

    required = MANDATORY_FIELDS.get(feature_set, MANDATORY_FIELDS["solar_only_v1"])
    missing = [f for f in required if _is_null(record.get(f))]
    if missing:
        hard.append("MANDATORY_FIELDS")

    solar = record.get("solar_w")
    if not _is_null(solar):
        if not _finite(solar):
            hard.append("POWER_FINITE")
        elif float(solar) < 0.0:
            hard.append("SOLAR_NON_NEGATIVE")

    for col in ("load_w", "grid_w", "battery_w"):
        value = record.get(col)
        if not _is_null(value) and not _finite(value):
            hard.append("POWER_FINITE")
            break

    soc = record.get("soc_percent")
    if not _is_null(soc):
        if not _finite(soc):
            hard.append("POWER_FINITE")
        elif not (policy.soc_min <= float(soc) <= policy.soc_max):
            hard.append("SOC_BOUNDS")

    # Scale ceiling — skipped (and reported) when the rating is unknown.
    rated = ctx.rated_pv_w if ctx is not None else None
    if rated is not None and not _is_null(solar) and _finite(solar):
        if float(solar) > policy.solar_scale_kappa * float(rated):
            hard.append("SOLAR_SCALE_CEILING")

    # -- soft rules --------------------------------------------------------
    if night_mask is None:
        ts = record.get("timestamp_utc")
        hour = None
        try:
            hour = pd.Timestamp(ts).hour
        except (TypeError, ValueError):
            hour = None
        if hour is not None:
            night_mask = hour < policy.night_start_hour or hour >= policy.night_end_hour
    if night_mask and rated is not None and not _is_null(solar) and _finite(solar):
        threshold = max(policy.night_solar_floor_w, policy.night_solar_fraction * float(rated))
        if float(solar) > threshold:
            soft.append("NIGHT_SOLAR")

    if _is_null(record.get("clear_sky_w")):
        soft.append("CLEAR_SKY_MISSING")

    native_dt = record.get("weather_native_dt_min")
    if not _is_null(native_dt) and int(native_dt) not in (5, 15):
        soft.append("WEATHER_CADENCE")

    fingerprint = hash_obj(
        {
            "k": sorted(str(k) for k in record.keys()),
            "v": {k: record.get(k) for k in sorted(record, key=str)},
            "rated": rated,
            "rule_set": policy.rule_set_version,
        }
    )

    if "MANDATORY_FIELDS" in hard and bool(record.get("is_gap", False)):
        decision = "INSUFFICIENT_DATA"
    elif hard:
        decision = "REJECTED"
    elif soft:
        decision = "ADMISSIBLE_WITH_WARNINGS"
    else:
        decision = "ADMISSIBLE"
    return Decision(
        decision=decision,
        hard_failures=hard,
        soft_warnings=soft,
        rule_set_version=policy.rule_set_version,
        input_fingerprint=fingerprint,
    )


def energy_balance_residual(
    frame: pd.DataFrame, ctx: SiteContext, dt_h: float = 0.25
) -> pd.Series:
    """Per-step energy-balance residual in Wh (soft rule input).

    ``residual_t = (dsoc_t / 100) * E_usable  -  (solar + grid - load) * dt_h``
    """
    if ctx.usable_battery_capacity_wh is None:
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    cap = float(ctx.usable_battery_capacity_wh)
    soc = pd.to_numeric(frame.get("soc_percent"), errors="coerce")
    solar = pd.to_numeric(frame.get("solar_w"), errors="coerce")
    load = pd.to_numeric(frame.get("load_w"), errors="coerce")
    grid = pd.to_numeric(frame.get("grid_w"), errors="coerce")
    grid = grid.fillna(0.0)
    dsoc = soc.diff() / 100.0 * cap
    net = (solar + grid - load) * dt_h
    return (dsoc - net).abs()


def evaluate_frame(
    frame: pd.DataFrame,
    ctx: SiteContext | None = None,
    policy: AdmissibilityPolicy = ADMISSIBILITY_POLICY,
    feature_set: str = "solar_only_v1",
) -> pd.DataFrame:
    """Evaluate every row of ``frame`` and return a decisions frame.

    The returned frame is index-aligned with ``frame`` and carries
    ``decision``, ``hard_failures``, ``soft_warnings``, ``rule_set_version``
    and ``input_fingerprint`` columns.
    """
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        decision = evaluate_record(record, ctx=ctx, policy=policy, feature_set=feature_set)
        rows.append(decision.to_dict())

    decisions = pd.DataFrame(rows, index=frame.index)

    # Soft rule: scale-aware energy balance (sequential, needs a sequence).
    if (
        ctx is not None
        and ctx.usable_battery_capacity_wh is not None
        and {"soc_percent", "load_w"}.issubset(frame.columns)
    ):
        residual = energy_balance_residual(frame, ctx)
        tol = policy.energy_balance_tol_wh
        flagged = residual > tol
        for idx in frame.index[flagged]:
            warnings = decisions.at[idx, "soft_warnings"]
            if "ENERGY_BALANCE" not in warnings:
                decisions.at[idx, "soft_warnings"] = list(warnings) + ["ENERGY_BALANCE"]
        decisions["energy_balance_residual_wh"] = residual.round(6)
        decisions.loc[flagged, "decision"] = np.where(
            decisions.loc[flagged, "decision"] == "ADMISSIBLE",
            "ADMISSIBLE_WITH_WARNINGS",
            decisions.loc[flagged, "decision"],
        )
    return decisions


def apply_gate(
    frame: pd.DataFrame,
    ctx: SiteContext | None = None,
    policy: AdmissibilityPolicy = ADMISSIBILITY_POLICY,
    feature_set: str = "solar_only_v1",
) -> pd.DataFrame:
    """Apply the gate and return only the admitted rows.

    The returned frame keeps every original column plus ``decision``,
    ``quality_flags`` (merged soft codes) and ``input_fingerprint``.
    """
    decisions = evaluate_frame(frame, ctx=ctx, policy=policy, feature_set=feature_set)
    out = frame.copy()
    out["decision"] = decisions["decision"]
    out["quality_flags"] = decisions["soft_warnings"].apply(list)
    out["input_fingerprint"] = decisions["input_fingerprint"]
    out["rule_set_version"] = decisions["rule_set_version"]
    keep = out["decision"].isin(["ADMISSIBLE", "ADMISSIBLE_WITH_WARNINGS"])
    return out.loc[keep].reset_index(drop=True)


def summarize(decisions: Sequence[Decision] | Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate decision counts and code frequencies."""
    items = [d.to_dict() if isinstance(d, Decision) else d for d in decisions]
    counts: dict[str, int] = {}
    hard: dict[str, int] = {}
    soft: dict[str, int] = {}
    for item in items:
        counts[item["decision"]] = counts.get(item["decision"], 0) + 1
        for code in item.get("hard_failures", []):
            hard[code] = hard.get(code, 0) + 1
        for code in item.get("soft_warnings", []):
            soft[code] = soft.get(code, 0) + 1
    return {
        "n_evaluated": len(items),
        "decision_counts": counts,
        "hard_failure_counts": hard,
        "soft_warning_counts": soft,
    }
