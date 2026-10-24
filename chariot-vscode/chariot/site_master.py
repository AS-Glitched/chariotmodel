"""Site Master — authoritative registry of physical site ratings (PDF §4.1).

The Site Master is the only place where *scale* is declared.  Every downstream
rule (admissibility scale ceiling, tribunal power limits, calibration scope,
capsule lineage) resolves its numeric context from here, so the same physics
rule family scales from a 5 kW rooftop to a 5 MW plant without a code change.

Resolution order for a site context:

1. explicit request override (used by tests and ad-hoc analyses),
2. the active Site Master row,
3. missing -> ``rated_pv_w is None``; scale hard rules are skipped and logged.
"""

from __future__ import annotations

import json

from typing import Any

from chariot.config import SiteContext
from chariot.store import Store, new_id, utc_now

SITE_FIELDS = (
    "name",
    "status",
    "rated_pv_w",
    "usable_battery_capacity_wh",
    "max_charge_w",
    "max_discharge_w",
    "timezone",
    "latitude",
    "longitude",
    "fleet_id",
    "metadata",
)


def _as_mapping(value: Any) -> dict[str, Any]:
    """Return a mapping from a stored metadata value (JSON text or dict)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _row_to_context(row: dict[str, Any]) -> SiteContext:
    rated = row.get("rated_pv_w")
    return SiteContext(
        tenant_id=row["tenant_id"],
        site_id=row["site_id"],
        name=row.get("name") or "",
        status=row.get("status") or "active",
        rated_pv_w=None if rated is None else float(rated),
        usable_battery_capacity_wh=(
            None
            if row.get("usable_battery_capacity_wh") is None
            else float(row["usable_battery_capacity_wh"])
        ),
        max_charge_w=(
            None if row.get("max_charge_w") is None else float(row["max_charge_w"])
        ),
        max_discharge_w=(
            None
            if row.get("max_discharge_w") is None
            else float(row["max_discharge_w"])
        ),
        timezone=row.get("timezone") or "UTC",
        latitude=None if row.get("latitude") is None else float(row["latitude"]),
        longitude=None if row.get("longitude") is None else float(row["longitude"]),
        fleet_id=row.get("fleet_id"),
        metadata=_as_mapping(row.get("metadata_json")),
        source="site_master",
        missing_rating=rated is None,
    )


def upsert_site(
    store: Store,
    tenant_id: str,
    site_id: str,
    *,
    name: str = "",
    status: str = "active",
    rated_pv_w: float | None = None,
    usable_battery_capacity_wh: float | None = None,
    max_charge_w: float | None = None,
    max_discharge_w: float | None = None,
    timezone: str = "UTC",
    latitude: float | None = None,
    longitude: float | None = None,
    fleet_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    updated_by: str = "system",
) -> dict[str, Any]:
    """Create or update a Site Master row (PATCH semantics)."""
    now = utc_now()
    existing = store.get_site(tenant_id, site_id)
    row = {
        "tenant_id": tenant_id,
        "site_id": site_id,
        "name": name,
        "status": status,
        "rated_pv_w": rated_pv_w,
        "usable_battery_capacity_wh": usable_battery_capacity_wh,
        "max_charge_w": max_charge_w,
        "max_discharge_w": max_discharge_w,
        "timezone": timezone,
        "latitude": latitude,
        "longitude": longitude,
        "fleet_id": fleet_id,
        "metadata_json": __import__("json").dumps(metadata or {}, sort_keys=True),
        "created_at_utc": existing["created_at_utc"] if existing else now,
        "updated_at_utc": now,
        "updated_by": updated_by,
    }
    store.upsert("sites", row, conflict="tenant_id, site_id")
    store.audit(
        "site.upsert",
        "site",
        site_id,
        tenant_id=tenant_id,
        actor=updated_by,
        payload={"site_id": site_id, "status": status},
    )
    return row


def retire_site(store: Store, tenant_id: str, site_id: str, updated_by: str = "system") -> None:
    """Retire a site so it stops resolving an active context."""
    site = store.get_site(tenant_id, site_id)
    if site is None:
        raise KeyError(f"unknown site {tenant_id}/{site_id}")
    store.execute(
        "UPDATE sites SET status='retired', updated_at_utc=?, updated_by=? "
        "WHERE tenant_id=? AND site_id=?",
        (utc_now(), updated_by, tenant_id, site_id),
    )
    store.audit(
        "site.retire",
        "site",
        site_id,
        tenant_id=tenant_id,
        actor=updated_by,
        payload={"site_id": site_id},
    )


def resolve_context(
    store: Store,
    tenant_id: str,
    site_id: str,
    *,
    override: dict[str, Any] | None = None,
    at: str | None = None,
) -> SiteContext:
    """Resolve the effective site context.

    ``override`` is a request-level override (used by tests).  ``at`` is
    accepted for API symmetry; the Site Master keeps a single active row per
    site, so no historical lookup is performed.
    """
    override = override or {}
    row = store.get_site(tenant_id, site_id)
    if row is None:
        ctx = SiteContext(
            tenant_id=tenant_id,
            site_id=site_id,
            source="missing",
            missing_rating=True,
        )
    else:
        ctx = _row_to_context(row)
        if ctx.status != "active":
            ctx.source = "missing"
            ctx.missing_rating = True

    if override:
        for field in SITE_FIELDS:
            if field in override and override[field] is not None:
                setattr(ctx, field, override[field])
        ctx.source = "request_override"
        ctx.missing_rating = ctx.rated_pv_w is None
    return ctx


def context_payload(ctx: SiteContext) -> dict[str, Any]:
    """JSON-safe representation of a site context."""
    return ctx.to_dict()
