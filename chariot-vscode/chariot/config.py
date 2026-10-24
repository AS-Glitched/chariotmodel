"""CHARIOT v3 — central configuration, versions, and versioned policies.

Every policy in this module is *versioned*. Policies are never mutated in place:
a change produces a new version string, and the version string is stamped onto
every artifact (capsules, datasets, model versions, batches, tribunal results,
confidence reports, advisories) so that lineage can be replayed exactly.

This module is the single source of truth for the numeric guardrails referenced
by the CHARIOT-3 master reference (Design Freeze Index).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# System identity
# --------------------------------------------------------------------------

SYSTEM_NAME = "CHARIOT"
SYSTEM_VERSION = "3.0.0"
SYSTEM_TAGLINE = (
    "Physics-governed hierarchical Bayesian energy intelligence platform."
)

# --------------------------------------------------------------------------
# Versioned component identifiers (stamped into lineage)
# --------------------------------------------------------------------------

ADMISSIBILITY_RULE_SET_VERSION = "admissibility_v3.0"
CALIBRATION_POLICY_VERSION = "calibration_v3.0"
RIVER_VERTICAL_SCHEMA_VERSION = "river_vertical_v1"
RIVER_HORIZONTAL_SCHEMA_VERSION = "river_horizontal_v1"
CAPSULE_SCHEMA_VERSION = "capsule_v1"
TRAINING_EXTRACT_SCHEMA_VERSION = "training_extract_v1"
FEATURE_SET_ID = "physics_rich_v1"
TRIBUNCAL_POLICY_VERSION = "tribunal_v3.0"
SOLVER_VERSION = "solver_coulomb_v1"
CONFIDENCE_POLICY_VERSION = "confidence_v3.0"
RESIDUAL_POLICY_VERSION = "residual_v3.0"
FEEDBACK_POLICY_VERSION = "feedback_v3.0"
GOVERNOR_POLICY_VERSION = "governor_v3.0"
ADVISORY_POLICY_VERSION = "advisory_v3.0"

# --------------------------------------------------------------------------
# Horizons and roles
# --------------------------------------------------------------------------

#: Horizon lengths in 15-minute steps (dataset cadence).
HORIZONS: dict[str, int] = {
    "short": 16,       # 4 h
    "operational": 96,  # 24 h
    "long": 288,        # 72 h
}

HORIZON_HOURS: dict[str, float] = {
    "short": 4.0,
    "operational": 24.0,
    "long": 72.0,
}

ROLE_MODES: tuple[str, ...] = ("research", "grid_engineer", "investor")

#: Default horizon per role mode (PDF §4.14).
ROLE_DEFAULT_HORIZON: dict[str, str] = {
    "research": "short",
    "grid_engineer": "operational",
    "investor": "long",
}

# --------------------------------------------------------------------------
# Dataclasses (policies)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissibilityPolicy:
    """Physics admission controller policy (PDF §4.2)."""

    rule_set_version: str = ADMISSIBILITY_RULE_SET_VERSION
    #: solar_w may not exceed rated_pv_w * kappa when rated_pv_w is known.
    solar_scale_kappa: float = 1.2
    #: SOC hard bounds (percent).
    soc_min: float = 0.0
    soc_max: float = 100.0
    #: Minimum fraction of the mandatory field set that must be present for a
    #: record to be *admissible* rather than *insufficient*.
    min_present_fraction: float = 0.6
    #: Scale-aware energy-balance tolerance (Wh per interval) used by the soft
    #: ENERGY_BALANCE rule.  Scaled by interval length and site ratings.
    energy_balance_tol_wh: float = 50.0
    #: Night-solar soft threshold as a fraction of rated PV.
    night_solar_fraction: float = 0.001
    #: Absolute night-solar floor (W) below which night solar is not flagged.
    night_solar_floor_w: float = 10.0
    #: SOC rate limit in percent-per-interval (soft rule).
    soc_rate_limit_pct: float = 25.0
    #: UTC hours considered "night" for the NIGHT_SOLAR soft rule.
    night_start_hour: int = 20
    night_end_hour: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TribunalPolicy:
    """Reality Tribunal policy (PDF §4.10).  Fail-closed, per timestep."""

    policy_version: str = TRIBUNCAL_POLICY_VERSION
    solver_version: str = SOLVER_VERSION
    #: Maximum absolute energy-balance residual per timestep (Wh).
    energy_balance_tol_wh: float = 0.1
    #: SOC hard bounds (percent).
    soc_min: float = 0.0
    soc_max: float = 100.0
    #: Solar scale ceiling factor applied by the Tribunal (kappa).
    solar_scale_kappa: float = 1.2
    #: Battery charge/discharge power ceiling tolerance factor.
    power_limit_tolerance: float = 1.0
    #: Optional ramp limit on PV output (W per step).  ``None`` disables.
    solar_ramp_w_per_step: float | None = None
    #: Optional ramp limit on load (W per step).  ``None`` disables.
    load_ramp_w_per_step: float | None = None
    #: Minimum retained fraction required for a batch to be *certified*.
    min_survival_for_certified: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidencePolicy:
    """Physics-Survival Confidence Engine policy (PDF §4.13)."""

    policy_version: str = CONFIDENCE_POLICY_VERSION
    #: Below this survival rate the engine abstains.
    s_min: float = 0.30
    #: Below this confidence the engine reports reduced confidence.
    c_low: float = 0.40
    #: Above this normalized uncertainty the engine reports reduced confidence.
    u_high: float = 0.80
    #: Sharpness function g.  ``identity`` -> C = (1 - u) * s
    sharpness: str = "identity"
    #: Normalisation reference width for the P90-P10 band (relative to the
    #: magnitude of the retained median signal).
    uncertainty_reference: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualPolicy:
    """Residual Proposal Chamber policy (PDF §4.8)."""

    policy_version: str = RESIDUAL_POLICY_VERSION
    #: Hard contribution ceiling.  y = mu + w * r_hat with 0 <= w <= w_max.
    w_max: float = 0.20
    #: Soft correction floor (below this the correction is ignored).
    min_abs_correction_w: float = 0.0
    #: SOC residual correction is policy-gated and OFF by default.
    soc_residual_enabled: bool = False
    #: Targets the chamber is allowed to correct.
    allowed_targets: tuple[str, ...] = ("solar_w", "load_w")
    #: Candidate families, in preference order.
    families: tuple[str, ...] = ("ridge",)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GovernorPolicy:
    """Residual Stability Governor policy (PDF §4.12)."""

    policy_version: str = GOVERNOR_POLICY_VERSION
    #: Attenuation factor applied on a breach.
    beta: float = 0.5
    #: Minimum relative contribution ratio c = E[|r|]/E[|y|] tolerated.
    c_max: float = 0.20
    #: Maximum tolerated correlation between |r_hat| and the reject indicator.
    rho_max: float = 0.50
    #: Minimum acceptable survival rate with the residual active.
    survival_floor: float = 0.50
    #: Survival delta (residual-on minus residual-off) tolerated.
    survival_delta_min: float = -0.05
    #: Consecutive breaches required to move to quarantined.
    breach_threshold: int = 2
    #: Consecutive healthy evaluations required to restore the weight.
    recovery_threshold: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationDefaults:
    """Identity defaults for the four effective calibration factors."""

    policy_version: str = CALIBRATION_POLICY_VERSION
    capacity_factor: float = 1.0
    charge_loss_factor: float = 0.0
    discharge_loss_factor: float = 0.0
    pv_correction_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SiteContext:
    """Resolved site scale context handed from Site Master to every layer."""

    tenant_id: str
    site_id: str
    name: str = ""
    status: str = "active"
    rated_pv_w: float | None = None
    usable_battery_capacity_wh: float | None = None
    max_charge_w: float | None = None
    max_discharge_w: float | None = None
    timezone: str = "UTC"
    latitude: float | None = None
    longitude: float | None = None
    fleet_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "missing"
    missing_rating: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Singletons
# --------------------------------------------------------------------------

ADMISSIBILITY_POLICY = AdmissibilityPolicy()
TRIBUNAL_POLICY = TribunalPolicy()
CONFIDENCE_POLICY = ConfidencePolicy()
RESIDUAL_POLICY = ResidualPolicy()
GOVERNOR_POLICY = GovernorPolicy()
CALIBRATION_DEFAULTS = CalibrationDefaults()


def horizon_steps(horizon: str) -> int:
    """Return the number of 15-minute steps for a named horizon."""
    try:
        return HORIZONS[horizon]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown horizon {horizon!r}; known horizons: {sorted(HORIZONS)}"
        ) from exc


def next_policy_version(current: str, policy_type: str) -> str:
    """Mint the next versioned identifier for a policy.

    Policies are versioned, never edited in place.  ``tribunal_v3.0`` becomes
    ``tribunal_v3.0.1``, then ``tribunal_v3.0.2`` and so on.
    """
    base = current or policy_type
    parts = base.split(".")
    # A trailing pure counter (``tribunal_v3.0.1``) is incremented; otherwise
    # the first revision counter is appended (``tribunal_v3.0`` -> ``.1``).
    if len(parts) >= 3 and parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    return f"{base}.1"


def policy_snapshot() -> dict[str, Any]:
    """Return every active policy for manifest / lineage stamping."""
    return {
        "system_version": SYSTEM_VERSION,
        "admissibility": ADMISSIBILITY_POLICY.to_dict(),
        "tribunal": TRIBUNAL_POLICY.to_dict(),
        "confidence": CONFIDENCE_POLICY.to_dict(),
        "residual": RESIDUAL_POLICY.to_dict(),
        "governor": GOVERNOR_POLICY.to_dict(),
        "calibration_defaults": CALIBRATION_DEFAULTS.to_dict(),
        "feature_set_id": FEATURE_SET_ID,
        "horizons": dict(HORIZONS),
        "roles": list(ROLE_MODES),
    }
