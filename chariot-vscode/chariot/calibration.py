"""Calibration Engine — versioned, hierarchical effective physical parameters.

PDF §4.4.  The four primary *effective* parameters are mathematically
effective abstractions, not universal constants and not directly visible
sensor readings:

======================  ========  ==========================================
Parameter               Symbol    Relation
======================  ========  ==========================================
Capacity factor         c_f       E_usable = c_f * E_nominal
Charge loss factor      l_c       eta_c   = 1 - l_c
Discharge loss factor   l_d       eta_d   = 1 - l_d
PV correction factor    gamma_pv  P_eff   = gamma_pv * P_meas
======================  ========  ==========================================

Scope priority (most specific wins): equipment -> site -> fleet -> identity
defaults.  Lifecycle is ``draft -> approved -> retired``; every transition is
audited and approved versions are immutable (retire + create a new version
rather than editing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chariot.config import CALIBRATION_DEFAULTS, CalibrationDefaults
from chariot.store import Store, utc_now

SCOPE_ORDER = ("equipment", "site", "fleet")


@dataclass(frozen=True)
class CalibrationFactors:
    """The four effective parameters plus their effective physical meaning."""

    capacity_factor: float
    charge_loss_factor: float
    discharge_loss_factor: float
    pv_correction_factor: float
    calibration_version: str
    scope_type: str
    scope_id: str

    @property
    def usable_capacity_wh(self) -> float | None:
        return None  # resolved against a site rating at apply time

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity_factor": self.capacity_factor,
            "charge_loss_factor": self.charge_loss_factor,
            "discharge_loss_factor": self.discharge_loss_factor,
            "pv_correction_factor": self.pv_correction_factor,
            "calibration_version": self.calibration_version,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "effective_charge_efficiency": 1.0 - self.charge_loss_factor,
            "effective_discharge_efficiency": 1.0 - self.discharge_loss_factor,
        }


def identity_factors(scope_type: str = "identity", scope_id: str = "default") -> CalibrationFactors:
    """Identity defaults: (1, 0, 0, 1).  Resolution always succeeds."""
    d = CALIBRATION_DEFAULTS
    return CalibrationFactors(
        capacity_factor=d.capacity_factor,
        charge_loss_factor=d.charge_loss_factor,
        discharge_loss_factor=d.discharge_loss_factor,
        pv_correction_factor=d.pv_correction_factor,
        calibration_version="calibration_identity_v1",
        scope_type=scope_type,
        scope_id=scope_id,
    )


def create_version(
    store: Store,
    *,
    scope_type: str,
    scope_id: str,
    capacity_factor: float = 1.0,
    charge_loss_factor: float = 0.0,
    discharge_loss_factor: float = 0.0,
    pv_correction_factor: float = 1.0,
    method: str = "engineering_declaration",
    notes: str = "",
    tenant_id: str | None = None,
    site_id: str | None = None,
    created_by: str = "system",
) -> str:
    """Create a *draft* calibration version.  Approve privilege is separate."""
    if scope_type not in SCOPE_ORDER:
        raise ValueError(f"scope_type must be one of {SCOPE_ORDER}, got {scope_type!r}")
    if not (0.0 < capacity_factor <= 1.2):
        raise ValueError("capacity_factor must be in (0, 1.2]")
    if not (0.0 <= charge_loss_factor < 0.5):
        raise ValueError("charge_loss_factor must be in [0, 0.5)")
    if not (0.0 <= discharge_loss_factor < 0.5):
        raise ValueError("discharge_loss_factor must be in [0, 0.5)")
    if not (0.0 < pv_correction_factor <= 1.2):
        raise ValueError("pv_correction_factor must be in (0, 1.2]")

    version = f"calib_{scope_type}_{scope_id}_{utc_now().replace(':', '').replace('-', '')}"
    store.insert(
        "calibration_versions",
        {
            "calibration_version": version,
            "scope_type": scope_type,
            "scope_id": scope_id,
            "tenant_id": tenant_id,
            "site_id": site_id,
            "capacity_factor": capacity_factor,
            "charge_loss_factor": charge_loss_factor,
            "discharge_loss_factor": discharge_loss_factor,
            "pv_correction_factor": pv_correction_factor,
            "status": "draft",
            "method": method,
            "notes": notes,
            "created_at_utc": utc_now(),
            "approved_at_utc": None,
            "approved_by": None,
            "retired_at_utc": None,
        },
    )
    store.audit(
        "calibration.create",
        "calibration_version",
        version,
        tenant_id=tenant_id,
        actor=created_by,
        payload={"scope_type": scope_type, "scope_id": scope_id, "method": method},
    )
    return version


def approve_version(
    store: Store, calibration_version: str, approved_by: str = "system"
) -> None:
    """Approve a draft version.  Approval is a privileged, audited action."""
    row = store.query_one(
        "SELECT * FROM calibration_versions WHERE calibration_version = ?",
        (calibration_version,),
    )
    if row is None:
        raise KeyError(f"unknown calibration version {calibration_version}")
    if row["status"] != "draft":
        raise ValueError(f"only draft versions can be approved (got {row['status']})")
    store.execute(
        "UPDATE calibration_versions SET status='approved', approved_at_utc=?, approved_by=? "
        "WHERE calibration_version=?",
        (utc_now(), approved_by, calibration_version),
    )
    store.audit(
        "calibration.approve",
        "calibration_version",
        calibration_version,
        actor=approved_by,
        payload={"calibration_version": calibration_version},
    )


def retire_version(
    store: Store, calibration_version: str, actor: str = "system"
) -> None:
    """Retire an approved version (it stays queryable for lineage)."""
    store.execute(
        "UPDATE calibration_versions SET status='retired', retired_at_utc=? "
        "WHERE calibration_version=?",
        (utc_now(), calibration_version),
    )
    store.audit(
        "calibration.retire",
        "calibration_version",
        calibration_version,
        actor=actor,
        payload={"calibration_version": calibration_version},
    )


def resolve(
    store: Store,
    *,
    tenant_id: str | None = None,
    site_id: str | None = None,
    fleet_id: str | None = None,
    equipment_id: str | None = None,
    at: str | None = None,
) -> CalibrationFactors:
    """Resolve the effective calibration factors for a scope.

    Most specific approved scope wins: equipment -> site -> fleet -> identity.
    """
    candidates: list[tuple[str, str]] = []
    if equipment_id:
        candidates.append(("equipment", equipment_id))
    if site_id:
        candidates.append(("site", site_id))
    if fleet_id:
        candidates.append(("fleet", fleet_id))

    for scope_type, scope_id in candidates:
        row = store.query_one(
            "SELECT * FROM calibration_versions "
            "WHERE scope_type=? AND scope_id=? AND status='approved' "
            "ORDER BY approved_at_utc DESC LIMIT 1",
            (scope_type, scope_id),
        )
        if row is not None:
            return CalibrationFactors(
                capacity_factor=float(row["capacity_factor"]),
                charge_loss_factor=float(row["charge_loss_factor"]),
                discharge_loss_factor=float(row["discharge_loss_factor"]),
                pv_correction_factor=float(row["pv_correction_factor"]),
                calibration_version=row["calibration_version"],
                scope_type=scope_type,
                scope_id=scope_id,
            )
    return identity_factors()


def effective_capacity_wh(
    factors: CalibrationFactors, nominal_capacity_wh: float | None
) -> float | None:
    """E_usable = c_f * E_nominal."""
    if nominal_capacity_wh is None:
        return None
    return float(factors.capacity_factor) * float(nominal_capacity_wh)


def stamp_application(
    store: Store,
    calibration_version: str,
    subject_type: str,
    subject_id: str,
    actor: str = "system",
) -> None:
    """Record that a calibration version was applied to an artifact."""
    store.audit(
        "calibration.apply",
        subject_type,
        subject_id,
        actor=actor,
        payload={"calibration_version": calibration_version},
    )
