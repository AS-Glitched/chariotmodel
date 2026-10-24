"""CHARIOT v3 persistence layer.

A single SQLite database holds every auditable object: the Site Master,
calibration versions, admissibility decisions, river slices, replay
certificates, capsules, training datasets, model versions, residual versions,
scenario batches, the Rejection Echo Memory, confidence reports, advisories,
governor events, feedback jobs and an append-only audit log.

The database is *not* the source of truth for large payloads: sealed Parquet
files live on disk under ``var/`` and are referenced by content hash.  The
database is the lineage spine.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sites (
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    rated_pv_w REAL,
    usable_battery_capacity_wh REAL,
    max_charge_w REAL,
    max_discharge_w REAL,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    latitude REAL,
    longitude REAL,
    fleet_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    updated_by TEXT NOT NULL DEFAULT 'system',
    PRIMARY KEY (tenant_id, site_id)
);

CREATE TABLE IF NOT EXISTS calibration_versions (
    calibration_version TEXT PRIMARY KEY,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    tenant_id TEXT,
    site_id TEXT,
    capacity_factor REAL NOT NULL DEFAULT 1.0,
    charge_loss_factor REAL NOT NULL DEFAULT 0.0,
    discharge_loss_factor REAL NOT NULL DEFAULT 0.0,
    pv_correction_factor REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'draft',
    method TEXT NOT NULL DEFAULT 'engineering_declaration',
    notes TEXT NOT NULL DEFAULT '',
    created_at_utc TEXT NOT NULL,
    approved_at_utc TEXT,
    approved_by TEXT,
    retired_at_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_calib_scope ON calibration_versions(scope_type, scope_id, status);

CREATE TABLE IF NOT EXISTS admissibility_decisions (
    decision_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    decision TEXT NOT NULL,
    hard_failures TEXT NOT NULL DEFAULT '[]',
    soft_warnings TEXT NOT NULL DEFAULT '[]',
    rule_set_version TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adm_site_ts ON admissibility_decisions(tenant_id, site_id, timestamp_utc);

CREATE TABLE IF NOT EXISTS river_slices (
    slice_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    slice_type TEXT NOT NULL,
    parameter TEXT,
    time_start TEXT,
    time_end TEXT,
    n_rows INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    path TEXT NOT NULL,
    replay_id TEXT,
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_slice_replay ON river_slices(replay_id);

CREATE TABLE IF NOT EXISTS replay_certificates (
    replay_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    slice_ids TEXT NOT NULL,
    content_hashes TEXT NOT NULL,
    root_hash TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS capsules (
    capsule_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    calibration_version TEXT,
    admissibility_versions TEXT NOT NULL DEFAULT '[]',
    slice_ids TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT NOT NULL,
    payload_path TEXT NOT NULL,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL,
    created_by TEXT NOT NULL DEFAULT 'system'
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_version TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    feature_set_id TEXT NOT NULL,
    capsule_ids TEXT NOT NULL DEFAULT '[]',
    calibration_version TEXT,
    admissibility_versions TEXT NOT NULL DEFAULT '[]',
    trainable_row_count INTEGER NOT NULL DEFAULT 0,
    total_row_count INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    path TEXT NOT NULL,
    time_start TEXT,
    time_end TEXT,
    split_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL,
    created_by TEXT NOT NULL DEFAULT 'system'
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    dataset_content_hash TEXT NOT NULL,
    params_json TEXT NOT NULL DEFAULT '{}',
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'draft',
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS residual_versions (
    residual_version TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    target TEXT NOT NULL,
    model_version TEXT,
    dataset_version TEXT,
    w_max REAL NOT NULL DEFAULT 0.2,
    params_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'draft',
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenario_batches (
    batch_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    origin_utc TEXT NOT NULL,
    horizon_steps INTEGER NOT NULL,
    dt_h REAL NOT NULL,
    n_scenarios INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    residual_version TEXT,
    residual_weight REAL NOT NULL DEFAULT 0.0,
    dataset_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenario_scenarios (
    batch_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    lineage_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    PRIMARY KEY (batch_id, scenario_id)
);

CREATE TABLE IF NOT EXISTS tribunal_batches (
    batch_id TEXT PRIMARY KEY,
    n_proposed INTEGER NOT NULL,
    n_retained INTEGER NOT NULL,
    n_rejected INTEGER NOT NULL,
    survival_rate REAL NOT NULL,
    retained_scenario_ids TEXT NOT NULL DEFAULT '[]',
    rejection_ids TEXT NOT NULL DEFAULT '[]',
    tribunal_policy_version TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tribunal_rejections (
    rejection_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    chariot_model_version TEXT NOT NULL,
    residual_version TEXT,
    residual_weight_w REAL NOT NULL DEFAULT 0.0,
    solver_version TEXT NOT NULL,
    tribunal_policy_version TEXT NOT NULL,
    failed_timestep INTEGER NOT NULL,
    constraint_codes TEXT NOT NULL DEFAULT '[]',
    max_balance_error_wh REAL NOT NULL DEFAULT 0.0,
    violation_magnitude REAL NOT NULL DEFAULT 0.0,
    exception_fingerprint TEXT NOT NULL DEFAULT '',
    lineage_json TEXT NOT NULL DEFAULT '{}',
    scenario_content_hash TEXT NOT NULL DEFAULT '',
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reject_batch ON tribunal_rejections(batch_id);
CREATE INDEX IF NOT EXISTS idx_reject_created ON tribunal_rejections(created_at_utc);

CREATE TABLE IF NOT EXISTS confidence_reports (
    report_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL,
    survival_rate REAL NOT NULL,
    uncertainty REAL NOT NULL,
    confidence REAL NOT NULL,
    policy_version TEXT NOT NULL,
    report_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS advisories (
    advisory_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    role_mode TEXT NOT NULL,
    horizon TEXT NOT NULL,
    status TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    report_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS governor_events (
    event_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    previous_state TEXT NOT NULL,
    state TEXT NOT NULL,
    w REAL NOT NULL,
    c REAL NOT NULL DEFAULT 0.0,
    rho REAL NOT NULL DEFAULT 0.0,
    signals_json TEXT NOT NULL DEFAULT '{}',
    action TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback_jobs (
    job_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    path TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id TEXT PRIMARY KEY,
    tenant_id TEXT,
    actor TEXT NOT NULL DEFAULT 'system',
    action TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_subject ON audit_log(subject_type, subject_id);

CREATE TABLE IF NOT EXISTS policy_versions (
    policy_version TEXT PRIMARY KEY,
    policy_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'approved',
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_policy_type ON policy_versions(policy_type);
"""


def utc_now() -> str:
    """Current UTC timestamp in ISO-8601 with a trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    """Short, collision-resistant identifier with a readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _dumps(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


#: Columns that hold JSON documents.  They are decoded on read so that the
#: rest of the platform can work with real Python structures.
JSON_COLUMNS: frozenset[str] = frozenset(
    {
        "metadata_json",
        "manifest_json",
        "split_json",
        "params_json",
        "diagnostics_json",
        "lineage_json",
        "report_json",
        "payload_json",
        "signals_json",
        "reason_codes_json",
        "slice_ids",
        "content_hashes",
        "admissibility_versions",
        "capsule_ids",
        "rejection_ids",
        "values",
        "tags",
    }
)


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    """Decode the JSON columns of a row."""
    for key, value in row.items():
        if key in JSON_COLUMNS and isinstance(value, str) and value:
            try:
                row[key] = json.loads(value)
            except (TypeError, ValueError):
                pass
    return row


class Store:
    """Thin, dependency-free SQLite wrapper used by every CHARIOT layer."""

    def __init__(self, path: str | Path = "var/chariot.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- helpers -----------------------------------------------------------
    def execute(self, sql: str, params: Any = ()) -> sqlite3.Cursor:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur

    def query(self, sql: str, params: Any = ()) -> list[dict[str, Any]]:
        cur = self._conn.execute(sql, params)
        return [_decode_row(dict(row)) for row in cur.fetchall()]

    def query_one(self, sql: str, params: Any = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def upsert(self, table: str, row: dict[str, Any], conflict: str | None = None) -> None:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        target = conflict or cols[0]
        conflict_cols = [c.strip() for c in target.split(",")] if conflict else [cols[0]]
        assignments = ", ".join(
            f"{c}=excluded.{c}" for c in cols if c not in conflict_cols
        )
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT({target}) DO UPDATE SET {assignments}"
        )
        self.execute(sql, tuple(self._encode(row[c]) for c in cols))

    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        self.execute(sql, tuple(self._encode(row[c]) for c in cols))

    @staticmethod
    def _encode(value: Any) -> Any:
        """JSON-encode structured values; SQLite cannot bind lists/dicts."""
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value, sort_keys=True, default=str)
        return value

    # -- audit -------------------------------------------------------------
    def audit(
        self,
        action: str,
        subject_type: str,
        subject_id: str | None = None,
        tenant_id: str | None = None,
        actor: str = "system",
        payload: Any = None,
    ) -> str:
        audit_id = new_id("aud")
        self.insert(
            "audit_log",
            {
                "audit_id": audit_id,
                "tenant_id": tenant_id,
                "actor": actor,
                "action": action,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "payload_json": _dumps(payload or {}),
                "created_at_utc": utc_now(),
            },
        )
        return audit_id

    # -- convenience readers ----------------------------------------------
    def get_sites(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        if tenant_id:
            return self.query(
                "SELECT * FROM sites WHERE tenant_id = ? ORDER BY site_id", (tenant_id,)
            )
        return self.query("SELECT * FROM sites ORDER BY tenant_id, site_id")

    def get_site(self, tenant_id: str, site_id: str) -> dict[str, Any] | None:
        return self.query_one(
            "SELECT * FROM sites WHERE tenant_id = ? AND site_id = ?", (tenant_id, site_id)
        )

    def rejections_for_batch(self, batch_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM tribunal_rejections WHERE batch_id = ? ORDER BY rejection_id",
            (batch_id,),
        )

    def all_rejections(self, limit: int = 100) -> list[dict[str, Any]]:
        """Every rejection in the echo, newest first.

        The Echo is append-only and never pruned — this is the audit trail
        behind "rejections are never silently dropped".
        """
        return self.query(
            "SELECT * FROM tribunal_rejections ORDER BY created_at_utc DESC, rejection_id"
            " LIMIT ?",
            (int(limit),),
        )

    def get_rejection(self, rejection_id: str) -> dict[str, Any] | None:
        return self.query_one(
            "SELECT * FROM tribunal_rejections WHERE rejection_id = ?", (rejection_id,)
        )

    def record_policy_version(
        self,
        policy_type: str,
        policy_version: str,
        payload: dict[str, Any],
        *,
        status: str = "approved",
        actor: str = "system",
    ) -> dict[str, Any]:
        """Persist a versioned policy and audit the transition.

        Policies are never edited in place: a change mints a *new* version and
        the previous one stays on record ("unversioned tribunal policy" is a
        prohibited design).
        """
        row = {
            "policy_version": policy_version,
            "policy_type": policy_type,
            "payload_json": _dumps(payload),
            "status": status,
            "created_at_utc": utc_now(),
        }
        self.insert("policy_versions", row)
        self.audit(
            actor=actor,
            action="policy_version_created",
            subject_type="policy",
            subject_id=policy_version,
            payload={"policy_type": policy_type, "policy_version": policy_version},
        )
        return row

    def policy_versions(self, policy_type: str | None = None) -> list[dict[str, Any]]:
        if policy_type is None:
            return self.query(
                "SELECT * FROM policy_versions ORDER BY created_at_utc DESC, policy_version"
            )
        return self.query(
            "SELECT * FROM policy_versions WHERE policy_type = ?"
            " ORDER BY created_at_utc DESC, policy_version",
            (policy_type,),
        )
