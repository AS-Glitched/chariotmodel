"""Canonical hashing helpers.

CHARIOT lineage is cryptographic: every artifact (slice, capsule, dataset,
model version, scenario batch, rejection, confidence report, advisory) carries
a SHA-256 content hash computed from a *canonical* byte representation so the
same logical content always produces the same digest on any machine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def canonical_json(obj: Any) -> bytes:
    """Serialize ``obj`` to canonical, deterministic JSON bytes."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    return str(obj)


def sha256_hex(payload: bytes) -> str:
    """Hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def hash_obj(obj: Any) -> str:
    """Hex SHA-256 digest of the canonical JSON encoding of ``obj``."""
    return sha256_hex(canonical_json(obj))


def hash_bytes_join(*parts: bytes) -> str:
    """Hash a sequence of byte payloads joined by a length-prefixed separator.

    Used for capsule ids: ``sha256(payload_bytes || canonical_header)``.
    """
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


def frame_content_hash(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Deterministic content hash of a DataFrame.

    Rows are sorted by ``(site_id, timestamp_utc)`` when present so the hash is
    invariant to ingestion order; the CSV rendering is canonical (fixed float
    formatting, no index).
    """
    sort_cols = [c for c in ("site_id", "timestamp_utc") if c in frame.columns]
    ordered = frame.sort_values(sort_cols) if sort_cols else frame
    if columns is not None:
        keep = [c for c in columns if c in ordered.columns]
        ordered = ordered[keep]
    payload = ordered.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return sha256_hex(payload)


def file_content_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def short_hash(digest: str, length: int = 12) -> str:
    """Short, human-readable form of a digest (used in ids)."""
    return digest[:length]
