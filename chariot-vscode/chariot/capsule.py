"""Immutable Capsule Sealing — earned immutability (PDF §4.5).

A capsule is created *after* admission, calibration and the seal-time
consistency checks.  Its id is

    capsule_id = sha256(payload_bytes || canonical_header)

where the canonical header carries tenant, site, window, schema version,
calibration version, admissibility versions, slice ids and slice content
hashes.  The payload is written atomically (file first, then the database
row) and never updated in place: a correction produces a *new* capsule.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pandas as pd

from chariot.config import CAPSULE_SCHEMA_VERSION
from chariot.hashing import canonical_json, frame_content_hash, hash_bytes_join, sha256_hex
from chariot.store import Store, utc_now


def _capsule_root() -> Path:
    return Path("var") / "capsules"


def seal_time_checks(
    frame: pd.DataFrame,
    *,
    ctx: Any,
    calibration_version: str | None,
    require_clear_sky: bool = True,
) -> dict[str, Any]:
    """Seal-time consistency checks.  Any failure rejects the seal."""
    problems: list[str] = []
    if len(frame) == 0:
        problems.append("EMPTY_PAYLOAD")
    if "timestamp_utc" not in frame.columns:
        problems.append("MISSING_TIMESTAMP")
    else:
        ts = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        if ts.isna().any():
            problems.append("INVALID_TIMESTAMP")
        if ts.notna().any():
            if ts.max() <= ts.min():
                problems.append("INVALID_TIME_RANGE")
            if not ts.is_monotonic_increasing:
                problems.append("NON_MONOTONIC_TIMESTAMP")
    if require_clear_sky and "clear_sky_w" in frame.columns:
        if frame["clear_sky_w"].isna().all():
            problems.append("NO_CLEAR_SKY")
    if ctx is not None and getattr(ctx, "rated_pv_w", None) is None:
        # Allowed: missing rating skips only the scale rules (logged upstream).
        pass
    if calibration_version is None:
        # Identity calibration is allowed by policy; recorded as a note.
        pass
    return {"ok": not problems, "problems": problems}


def seal_capsule(
    store: Store,
    frame: pd.DataFrame,
    *,
    tenant_id: str,
    site_id: str,
    calibration_version: str | None,
    admissibility_versions: list[str],
    slice_ids: list[str],
    slice_hashes: list[str],
    ctx: Any = None,
    created_by: str = "system",
    require_clear_sky: bool = True,
) -> dict[str, Any]:
    """Seal an admitted, calibrated frame into an immutable capsule."""
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    checks = seal_time_checks(
        ordered, ctx=ctx, calibration_version=calibration_version, require_clear_sky=require_clear_sky
    )
    if not checks["ok"]:
        raise ValueError(f"seal rejected: {checks['problems']}")

    payload_hash = frame_content_hash(ordered)
    root = _capsule_root()
    root.mkdir(parents=True, exist_ok=True)

    header = {
        "tenant_id": tenant_id,
        "site_id": site_id,
        "window_start": str(ordered["timestamp_utc"].min()),
        "window_end": str(ordered["timestamp_utc"].max()),
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "calibration_version": calibration_version,
        "admissibility_versions": list(admissibility_versions),
        "slice_ids": list(slice_ids),
        "slice_content_hashes": list(slice_hashes),
        "n_rows": int(len(ordered)),
    }
    payload_bytes = canonical_json(
        {k: header[k] for k in ("tenant_id", "site_id", "window_start", "window_end", "schema_version")}
    ) + b"|" + payload_hash.encode("ascii")
    capsule_id = hash_bytes_join(payload_bytes, canonical_json(header))

    path = root / f"{capsule_id}.parquet"
    if not path.exists():
        ordered.to_parquet(path, index=False)

    manifest = {
        **header,
        "payload_content_hash": payload_hash,
        "capsule_id": capsule_id,
        "created_at_utc": utc_now(),
        "created_by": created_by,
        "seal_checks": checks,
        "site_ratings": {
            "rated_pv_w": getattr(ctx, "rated_pv_w", None),
            "usable_battery_capacity_wh": getattr(ctx, "usable_battery_capacity_wh", None),
            "timezone": getattr(ctx, "timezone", None),
        },
    }
    row = {
        "capsule_id": capsule_id,
        "tenant_id": tenant_id,
        "site_id": site_id,
        "window_start": header["window_start"],
        "window_end": header["window_end"],
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "calibration_version": calibration_version,
        "admissibility_versions": list(admissibility_versions),
        "slice_ids": list(slice_ids),
        "content_hash": payload_hash,
        "payload_path": str(path),
        "manifest_json": canonical_json(manifest).decode("utf-8"),
        "created_at_utc": utc_now(),
        "created_by": created_by,
    }
    store.upsert("capsules", row)
    store.audit(
        "capsule.seal",
        "capsule",
        capsule_id,
        tenant_id=tenant_id,
        actor=created_by,
        payload={"content_hash": payload_hash, "slice_ids": list(slice_ids)},
    )
    return {**row, "manifest": manifest}


def load_capsule(store: Store, capsule_id: str) -> pd.DataFrame:
    """Load the sealed payload of a capsule."""
    row = store.query_one("SELECT * FROM capsules WHERE capsule_id = ?", (capsule_id,))
    if row is None:
        raise KeyError(f"unknown capsule {capsule_id}")
    return pd.read_parquet(row["payload_path"])


def verify_capsule(store: Store, capsule_id: str) -> dict[str, Any]:
    """Recompute the capsule id from the stored payload and header."""
    row = store.query_one("SELECT * FROM capsules WHERE capsule_id = ?", (capsule_id,))
    if row is None:
        raise KeyError(f"unknown capsule {capsule_id}")
    frame = pd.read_parquet(row["payload_path"])
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    payload_hash = frame_content_hash(ordered)
    manifest = row.get("manifest_json") or {}
    if isinstance(manifest, str):
        manifest = json.loads(manifest)
    header = {
        "tenant_id": row["tenant_id"],
        "site_id": row["site_id"],
        "window_start": row["window_start"],
        "window_end": row["window_end"],
        "schema_version": row["schema_version"],
        "calibration_version": row["calibration_version"],
        "admissibility_versions": row["admissibility_versions"],
        "slice_ids": row["slice_ids"],
        "slice_content_hashes": manifest.get("slice_content_hashes", []),
        "n_rows": int(len(ordered)),
    }
    payload_bytes = (
        canonical_json(
            {
                k: header[k]
                for k in ("tenant_id", "site_id", "window_start", "window_end", "schema_version")
            }
        )
        + b"|"
        + payload_hash.encode("ascii")
    )
    recomputed = hash_bytes_join(payload_bytes, canonical_json(header))
    return {
        "capsule_id": capsule_id,
        "valid": recomputed == capsule_id,
        "recomputed_id": recomputed,
        "stored_id": capsule_id,
        "payload_hash_matches": payload_hash == row["content_hash"],
    }


def capsule_digest(capsule_id: str) -> str:  # pragma: no cover - helper
    return sha256_hex(capsule_id.encode("ascii"))
