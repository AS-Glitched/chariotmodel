"""Training Extract — materialized, versioned, model-ready dataset (PDF §4.6).

Design decisions frozen by the master reference:

* **Option B — materialized**: the extract is a versioned Parquet dataset plus
  a manifest and a content hash.  The Data Layer never trains.
* **Clear-sky is mandatory** for a row to be trainable.  A row without a
  clear-sky anchor is kept (with ``is_trainable=false`` and the gate code
  ``MISSING_CLEAR_SKY``) but is masked out of training.
* The build **fails** if zero trainable rows are produced.
* ``R_t = P_solar_meas / max(P_clear, eps)`` is stored as
  ``clear_sky_ratio_r``.

The dataset also carries the deterministic, time-ordered
``train / validate / test`` split assignment used by the CHARIOT model
(60 / 20 / 20 by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chariot.config import FEATURE_SET_ID, TRAINING_EXTRACT_SCHEMA_VERSION
from chariot.hashing import frame_content_hash, hash_obj, short_hash
from chariot.store import Store, utc_now

CLEAR_SKY_EPS = 1.0

#: Canonical output columns of the training extract.
DATASET_COLUMNS: tuple[str, ...] = (
    "tenant_id",
    "site_id",
    "timestamp_utc",
    "capsule_id",
    "dataset_version",
    "solar_w",
    "load_w",
    "battery_w",
    "soc_percent",
    "grid_w",
    "clear_sky_power_w",
    "clear_sky_ratio_r",
    "solar_effective_w",
    "rated_pv_w",
    "usable_battery_capacity_wh",
    "timezone",
    "is_gap",
    "quality_flags",
    "is_trainable",
    "train_gate_codes",
    "split",
    "calibration_version",
    "capacity_factor",
    "charge_loss_factor",
    "discharge_loss_factor",
    "pv_correction_factor",
)


def assign_splits(
    frame: pd.DataFrame,
    *,
    train_frac: float = 0.60,
    validate_frac: float = 0.20,
    test_frac: float = 0.20,
    time_column: str = "timestamp_utc",
) -> pd.Series:
    """Assign a deterministic, time-ordered train/validate/test split.

    The split is chronological and never shuffles: the first ``train_frac`` of
    the timeline is train, the next ``validate_frac`` is validation, and the
    remainder is test.  Boundary rows are inclusive on the train/validate
    edge and exclusive elsewhere, so a row belongs to exactly one split and
    no timestamp appears in two splits.
    """
    total = train_frac + validate_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    ordered = frame.sort_values(time_column).index
    n = len(ordered)
    n_train = int(np.floor(n * train_frac))
    n_val = int(np.floor(n * validate_frac))
    labels = pd.Series("test", index=frame.index, dtype="object")
    labels.loc[ordered[:n_train]] = "train"
    labels.loc[ordered[n_train : n_train + n_val]] = "validate"
    return labels


def _dataset_root() -> Path:
    return Path("var") / "datasets"


def build_training_extract(
    store: Store,
    frame: pd.DataFrame,
    *,
    tenant_id: str,
    site_id: str,
    capsule_ids: list[str],
    calibration_version: str | None,
    calibration_factors: dict[str, float],
    admissibility_versions: list[str],
    slice_ids: list[str],
    ctx: Any,
    dataset_version: str | None = None,
    train_frac: float = 0.60,
    validate_frac: float = 0.20,
    test_frac: float = 0.20,
    created_by: str = "system",
    honour_split: bool = False,
) -> dict[str, Any]:
    """Build the materialized training extract.

    Returns the manifest.  Raises ``ValueError`` when no trainable row is
    produced (the build must fail closed).
    """
    if len(frame) == 0:
        raise ValueError("training extract build failed: empty admitted frame")

    ds_version = dataset_version or (
        f"dataset_{short_hash(hash_obj([tenant_id, site_id, capsule_ids]), 12)}"
    )

    out = pd.DataFrame(index=frame.index)
    out["tenant_id"] = tenant_id
    out["site_id"] = site_id
    out["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    out["capsule_id"] = capsule_ids[0] if capsule_ids else ""
    out["dataset_version"] = ds_version

    for col in ("solar_w", "load_w", "battery_w", "soc_percent", "grid_w"):
        out[col] = (
            pd.to_numeric(frame[col], errors="coerce").astype("float64")
            if col in frame.columns
            else np.nan
        )

    clear = (
        pd.to_numeric(frame["clear_sky_w"], errors="coerce").astype("float64")
        if "clear_sky_w" in frame.columns
        else np.nan
    )
    out["clear_sky_power_w"] = clear
    ratio = out["solar_w"] / clear.clip(lower=CLEAR_SKY_EPS)
    out["clear_sky_ratio_r"] = ratio.where(clear > CLEAR_SKY_EPS)

    pv_corr = float(calibration_factors.get("pv_correction_factor", 1.0))
    out["solar_effective_w"] = out["solar_w"] * pv_corr

    out["rated_pv_w"] = getattr(ctx, "rated_pv_w", None)
    out["usable_battery_capacity_wh"] = getattr(ctx, "usable_battery_capacity_wh", None)
    out["timezone"] = getattr(ctx, "timezone", "UTC")

    out["is_gap"] = (
        frame["is_gap"].astype(bool).to_numpy()
        if "is_gap" in frame.columns
        else False
    )
    out["quality_flags"] = (
        frame["quality_flags"].apply(list).to_numpy()
        if "quality_flags" in frame.columns
        else [[] for _ in range(len(frame))]
    )

    # -- trainability gates -------------------------------------------------
    codes: list[list[str]] = []
    for idx, row in out.iterrows():
        row_codes: list[str] = []
        if not np.isfinite(row["clear_sky_power_w"]) or row["clear_sky_power_w"] <= CLEAR_SKY_EPS:
            row_codes.append("MISSING_CLEAR_SKY")
        if not np.isfinite(row["solar_w"]):
            row_codes.append("MISSING_SOLAR")
        if bool(row["is_gap"]):
            row_codes.append("GAP_ROW")
        if row["solar_w"] < 0:
            row_codes.append("NEGATIVE_SOLAR")
        codes.append(row_codes)
    out["train_gate_codes"] = codes
    out["is_trainable"] = [len(c) == 0 for c in codes]

    if honour_split and "split" in frame.columns:
        # The 2022 -> 2023 protocol assigns the split *upstream* (train and
        # validate live in 2022, the report-only test lives in 2023), so the
        # chronological re-split must be skipped.  The assignment is opt-in:
        # with the default the behaviour is bit-identical to v3.
        pre = frame["split"].astype(str).str.strip().str.lower()
        if pre.isin(["train", "validate", "test"]).all():
            out["split"] = pre.to_numpy()
            split_policy = "pre_assigned (upstream protocol)"
        else:  # pragma: no cover - defensive
            out["split"] = assign_splits(
                out,
                train_frac=train_frac,
                validate_frac=validate_frac,
                test_frac=test_frac,
            )
            split_policy = "time_ordered_60_20_20"
    else:
        out["split"] = assign_splits(
            out,
            train_frac=train_frac,
            validate_frac=validate_frac,
            test_frac=test_frac,
        )
        split_policy = "time_ordered_60_20_20"

    out["calibration_version"] = calibration_version or "calibration_identity_v1"
    out["capacity_factor"] = float(calibration_factors.get("capacity_factor", 1.0))
    out["charge_loss_factor"] = float(calibration_factors.get("charge_loss_factor", 0.0))
    out["discharge_loss_factor"] = float(calibration_factors.get("discharge_loss_factor", 0.0))
    out["pv_correction_factor"] = pv_corr

    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    trainable = int(out["is_trainable"].sum())
    if trainable == 0:
        raise ValueError(
            "training extract build failed: zero trainable rows produced "
            "(check MISSING_CLEAR_SKY gates)"
        )

    split_counts = out["split"].value_counts().to_dict()
    trainable_counts = out.loc[out["is_trainable"], "split"].value_counts().to_dict()

    content_hash = frame_content_hash(out, columns=list(DATASET_COLUMNS))
    root = _dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{ds_version}.parquet"
    out.to_parquet(path, index=False)

    manifest = {
        "dataset_version": ds_version,
        "feature_set_id": FEATURE_SET_ID,
        "schema_version": TRAINING_EXTRACT_SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "site_id": site_id,
        "capsule_ids": list(capsule_ids),
        "calibration_version": calibration_version,
        "admissibility_versions": list(admissibility_versions),
        "slice_ids": list(slice_ids),
        "trainable_row_count": trainable,
        "total_row_count": int(len(out)),
        "content_hash": content_hash,
        "path": str(path),
        "time_start": str(out["timestamp_utc"].min()),
        "time_end": str(out["timestamp_utc"].max()),
        "split_policy": split_policy,
        "split_fractions": {
            "train": train_frac,
            "validate": validate_frac,
            "test": test_frac,
        },
        "split_row_counts": {k: int(v) for k, v in split_counts.items()},
        "split_boundaries": {
            str(k): {
                "min": str(v["min"]),
                "max": str(v["max"]),
                "count": int(v["count"]),
            }
            for k, v in out.groupby("split")["timestamp_utc"]
            .agg(["min", "max", "count"])
            .to_dict("index")
            .items()
        },
        "trainable_row_counts_by_split": {k: int(v) for k, v in trainable_counts.items()},
        "train_gate_code_counts": _gate_counts(out),
        "created_at_utc": utc_now(),
        "created_by": created_by,
        "site_ratings": {
            "rated_pv_w": getattr(ctx, "rated_pv_w", None),
            "usable_battery_capacity_wh": getattr(ctx, "usable_battery_capacity_wh", None),
            "timezone": getattr(ctx, "timezone", "UTC"),
            "fleet_id": getattr(ctx, "fleet_id", None),
        },
    }
    store.upsert(
        "datasets",
        {
            "dataset_version": ds_version,
            "tenant_id": tenant_id,
            "site_id": site_id,
            "feature_set_id": FEATURE_SET_ID,
            "capsule_ids": list(capsule_ids),
            "calibration_version": calibration_version,
            "admissibility_versions": list(admissibility_versions),
            "trainable_row_count": trainable,
            "total_row_count": int(len(out)),
            "content_hash": content_hash,
            "path": str(path),
            "time_start": manifest["time_start"],
            "time_end": manifest["time_end"],
            "split_json": {
                "policy": "time_ordered_60_20_20",
                "fractions": manifest["split_fractions"],
                "row_counts": manifest["split_row_counts"],
            },
            "created_at_utc": utc_now(),
            "created_by": created_by,
        },
    )
    store.audit(
        "training_extract.build",
        "dataset",
        ds_version,
        tenant_id=tenant_id,
        actor=created_by,
        payload={"content_hash": content_hash, "trainable_row_count": trainable},
    )
    return manifest


def _gate_counts(out: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for codes in out["train_gate_codes"]:
        for code in codes:
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def load_dataset(store: Store, dataset_version: str) -> pd.DataFrame:
    """Load a materialized dataset by version."""
    row = store.query_one(
        "SELECT * FROM datasets WHERE dataset_version = ?", (dataset_version,)
    )
    if row is None:
        raise KeyError(f"unknown dataset {dataset_version}")
    return pd.read_parquet(row["path"])


def get_manifest(store: Store, dataset_version: str) -> dict[str, Any]:
    """Return the manifest of a dataset version."""
    row = store.query_one(
        "SELECT * FROM datasets WHERE dataset_version = ?", (dataset_version,)
    )
    if row is None:
        raise KeyError(f"unknown dataset {dataset_version}")
    manifest = {
        "dataset_version": row["dataset_version"],
        "feature_set_id": row["feature_set_id"],
        "capsule_ids": row["capsule_ids"],
        "calibration_version": row["calibration_version"],
        "admissibility_versions": row["admissibility_versions"],
        "trainable_row_count": row["trainable_row_count"],
        "total_row_count": row["total_row_count"],
        "content_hash": row["content_hash"],
        "time_start": row["time_start"],
        "time_end": row["time_end"],
        "created_at_utc": row["created_at_utc"],
    }
    return manifest
