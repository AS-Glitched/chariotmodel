"""River Flow Extraction Fabric — dual-axis, replay-verifiable extraction (PDF §4.3).

Two artifact families:

* **Vertical slice** — the full system state at every timestep of a window.
* **Horizontal slice** — a single parameter trajectory over time.

Guarantees:

* content-addressed Parquet payloads (SHA-256 over the canonical frame),
* idempotent builds (same input -> same ``slice_id``),
* quality flags and gaps preserved verbatim (the fabric never re-judges
  physics; upstream Admissibility owns rejection),
* a **Replay Certificate** that binds slice ids, content hashes and a root
  hash together so the extraction can be verified end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from chariot.config import (
    RIVER_HORIZONTAL_SCHEMA_VERSION,
    RIVER_VERTICAL_SCHEMA_VERSION,
)
from chariot.hashing import frame_content_hash, hash_obj, short_hash
from chariot.store import Store, new_id, utc_now


def _payload_root() -> Path:
    return Path("var") / "river_flow"


def slice_id_for(kind: str, tenant_id: str, site_id: str, content_hash: str) -> str:
    """Deterministic slice id: identical content -> identical id."""
    return f"slice_{kind}_{short_hash(hash_obj([kind, tenant_id, site_id, content_hash]), 16)}"


def write_payload(frame: pd.DataFrame, content_hash: str) -> str:
    """Write the Parquet payload for a slice and return its relative path."""
    root = _payload_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{content_hash}.parquet"
    if not path.exists():
        frame.to_parquet(path, index=False)
    return str(path)


def build_vertical_slice(
    store: Store,
    frame: pd.DataFrame,
    *,
    tenant_id: str,
    site_id: str,
) -> dict[str, Any]:
    """Build (idempotently) a vertical slice: full state over a time window."""
    if "timestamp_utc" not in frame.columns:
        raise ValueError("vertical slice requires a timestamp_utc column")
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    content_hash = frame_content_hash(ordered)
    slice_id = slice_id_for("vertical", tenant_id, site_id, content_hash)
    path = write_payload(ordered, content_hash)
    row = {
        "slice_id": slice_id,
        "tenant_id": tenant_id,
        "site_id": site_id,
        "slice_type": "vertical",
        "parameter": None,
        "time_start": str(ordered["timestamp_utc"].min()),
        "time_end": str(ordered["timestamp_utc"].max()),
        "n_rows": int(len(ordered)),
        "content_hash": content_hash,
        "path": path,
        "replay_id": None,
        "created_at_utc": utc_now(),
    }
    store.upsert("river_slices", row)
    return {**row, "schema_version": RIVER_VERTICAL_SCHEMA_VERSION}


def build_horizontal_slice(
    store: Store,
    frame: pd.DataFrame,
    parameter: str,
    *,
    tenant_id: str,
    site_id: str,
) -> dict[str, Any]:
    """Build (idempotently) a horizontal slice: one parameter vs. time."""
    if parameter not in frame.columns:
        raise KeyError(f"parameter {parameter!r} not present in frame")
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    content_hash = frame_content_hash(ordered)
    slice_id = slice_id_for("horizontal", tenant_id, site_id, f"{parameter}:{content_hash}")
    path = write_payload(ordered, content_hash)
    row = {
        "slice_id": slice_id,
        "tenant_id": tenant_id,
        "site_id": site_id,
        "slice_type": "horizontal",
        "parameter": parameter,
        "time_start": str(ordered["timestamp_utc"].min()),
        "time_end": str(ordered["timestamp_utc"].max()),
        "n_rows": int(len(ordered)),
        "content_hash": content_hash,
        "path": path,
        "replay_id": None,
        "created_at_utc": utc_now(),
    }
    store.upsert("river_slices", row)
    return {**row, "schema_version": RIVER_HORIZONTAL_SCHEMA_VERSION}


def issue_replay_certificate(
    store: Store, slice_ids: Sequence[str], *, tenant_id: str, site_id: str
) -> dict[str, Any]:
    """Bind a set of slices into a Replay Certificate with a root hash."""
    if not slice_ids:
        raise ValueError("a replay certificate needs at least one slice")
    rows = []
    for sid in slice_ids:
        row = store.query_one("SELECT * FROM river_slices WHERE slice_id = ?", (sid,))
        if row is None:
            raise KeyError(f"unknown slice {sid}")
        rows.append(row)
    content_hashes = [r["content_hash"] for r in rows]
    root_hash = hash_obj(
        {
            "slice_ids": list(slice_ids),
            "content_hashes": content_hashes,
            "schema_version": RIVER_VERTICAL_SCHEMA_VERSION,
        }
    )
    replay_id = f"replay_{short_hash(root_hash, 20)}"
    record = {
        "replay_id": replay_id,
        "tenant_id": tenant_id,
        "site_id": site_id,
        "slice_ids": list(slice_ids),
        "content_hashes": content_hashes,
        "root_hash": root_hash,
        "schema_version": RIVER_VERTICAL_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
    }
    store.upsert("replay_certificates", record)
    for row in rows:
        store.execute(
            "UPDATE river_slices SET replay_id = ? WHERE slice_id = ?", (replay_id, row["slice_id"])
        )
    store.audit(
        "river_flow.replay",
        "replay_certificate",
        replay_id,
        tenant_id=tenant_id,
        payload={"slice_ids": list(slice_ids), "root_hash": root_hash},
    )
    return record


def verify_replay(store: Store, replay_id: str) -> dict[str, Any]:
    """Re-verify a Replay Certificate: recompute hashes and compare."""
    record = store.query_one(
        "SELECT * FROM replay_certificates WHERE replay_id = ?", (replay_id,)
    )
    if record is None:
        raise KeyError(f"unknown replay {replay_id}")
    slice_ids = record["slice_ids"]
    content_hashes = record["content_hashes"]
    problems: list[str] = []
    for sid, expected in zip(slice_ids, content_hashes):
        row = store.query_one("SELECT * FROM river_slices WHERE slice_id = ?", (sid,))
        if row is None:
            problems.append(f"missing slice {sid}")
            continue
        path = Path(row["path"])
        if not path.exists():
            problems.append(f"missing payload for {sid}")
            continue
        frame = pd.read_parquet(path)
        actual = frame_content_hash(frame)
        if actual != expected:
            problems.append(f"content hash mismatch for {sid}: {actual} != {expected}")
    recomputed_root = hash_obj(
        {
            "slice_ids": list(slice_ids),
            "content_hashes": content_hashes,
            "schema_version": record["schema_version"],
        }
    )
    if recomputed_root != record["root_hash"]:
        problems.append("root hash mismatch")
    return {
        "replay_id": replay_id,
        "valid": not problems,
        "problems": problems,
        "root_hash": record["root_hash"],
        "recomputed_root_hash": recomputed_root,
        "slice_ids": list(slice_ids),
        "content_hashes": list(content_hashes),
        "n_slices": len(slice_ids),
    }


def load_slice(store: Store, slice_id: str) -> pd.DataFrame:
    """Load the payload of a slice by id."""
    row = store.query_one("SELECT * FROM river_slices WHERE slice_id = ?", (slice_id,))
    if row is None:
        raise KeyError(f"unknown slice {slice_id}")
    return pd.read_parquet(row["path"])


def new_replay_id() -> str:  # pragma: no cover - helper for external callers
    return new_id("replay")
