"""Sealed training extract: schema, hash, time-ordered split."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

REQUIRED = [
    "tenant_id",
    "site_id",
    "timestamp_utc",
    "solar_w",
    "load_w",
    "soc_percent",
    "clear_sky_w",
    "rated_pv_w",
    "is_gap",
]


def validate_schema(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame.duplicated(["site_id", "timestamp_utc"]).any():
        raise ValueError("Duplicate site_id + timestamp_utc")


def content_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seal(frame: pd.DataFrame, dataset_version: str) -> dict:
    validate_schema(frame)
    return {
        "dataset_version": dataset_version,
        "schema_version": "extract_v0",
        "n_rows": int(len(frame)),
        "content_hash": content_hash(frame),
        "time_start": str(frame["timestamp_utc"].min()),
        "time_end": str(frame["timestamp_utc"].max()),
        "site_id": str(frame["site_id"].iloc[0]),
        "rated_pv_w": float(frame["rated_pv_w"].iloc[0]),
        "usable_battery_capacity_wh": float(frame["usable_battery_capacity_wh"].iloc[0]),
    }


def time_split(frame: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    cut = max(1, min(len(frame) - 1, int(len(frame) * train_frac)))
    train = frame.iloc[:cut].copy()
    test = frame.iloc[cut:].copy()
    if train["timestamp_utc"].max() > test["timestamp_utc"].min():
        raise ValueError("Time split leaked the future")
    return train, test


def dumps(meta: dict) -> str:
    return json.dumps(meta, indent=2)
