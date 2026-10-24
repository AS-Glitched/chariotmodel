#!/usr/bin/env python3
"""Prepare the DKASC Alice Springs telemetry for a CHARIOT run.

Two protocols are supported.

``single`` (default, unchanged)
    One window, split chronologically into 60/20/20 train/validate/test.
    This is what the v3 reference run used.

``year2022_2023``
    The **2022 → 2023** evaluation protocol:

    ==================  ====================================================
    ``train_2022``      the first 60 % of the 2022 window (training only)
    ``validate_2022``   the remaining 40 % of the 2022 window (selection only)
    ``test_2023``       the whole 2023 window (report only, never fitted)
    ==================  ====================================================

    The 2022 window is *not* split 60/20/20: the 2023 year replaces the test
    split, so no 2022 row is ever used as a test target and nothing is fitted
    or selected on 2023.

The prepared slice is written to ``var/data/`` (git-ignored) together with a
manifest that records the content hash and the exact split boundaries, so the
split can be verified without shipping the bulk payload into git.

Usage::

    python scripts/prepare_dataset.py --source <dka_alice_springs_solar_only_15min_2022.csv>

    python scripts/prepare_dataset.py --protocol year2022_2023 \
        --source      <dka_alice_springs_solar_only_15min_2022.csv> \
        --source-test <dka_alice_springs_solar_only_15min_2023.csv> \
        --out runs/w1_data_2022_2023
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.dataset_qc import qc_frame, render_qc_markdown, summarise  # noqa: E402
from chariot.hashing import frame_content_hash  # noqa: E402
from chariot.training_extract import assign_splits  # noqa: E402

DEFAULT_START = "2022-01-01"
DEFAULT_END = "2023-01-01"
DEFAULT_OUT = ROOT / "var" / "data" / "dka_alice_springs_2022_15min.parquet"

#: The 2022 → 2023 protocol windows.
PROTOCOL_YEAR_2022 = ("2022-01-01", "2023-01-01")
PROTOCOL_YEAR_2023 = ("2023-01-01", "2024-01-01")
PROTOCOL_DATA_DIR = ROOT / "var" / "data"


def _read_source(path: Path, start: str, end: str) -> pd.DataFrame:
    """Read a CSV export and slice it to [start, end)."""
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    mask = (frame["timestamp_utc"] >= pd.Timestamp(start, tz="UTC")) & (
        frame["timestamp_utc"] < pd.Timestamp(end, tz="UTC")
    )
    sliced = frame.loc[mask].sort_values("timestamp_utc").reset_index(drop=True)
    if sliced.empty:
        raise SystemExit(f"no rows in {start}..{end} for {path}")
    if "is_gap" in sliced.columns:
        sliced["is_gap"] = sliced["is_gap"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        sliced["is_gap"] = False
    return sliced


def _boundaries(frame: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("train", "validate", "test"):
        sel = frame.loc[labels == name, "timestamp_utc"]
        if len(sel) == 0:
            continue
        out[name] = {
            "min": str(sel.min()),
            "max": str(sel.max()),
            "count": int(len(sel)),
        }
    return out


def prepare_single(args: argparse.Namespace) -> int:
    """The original single-window 60/20/20 protocol."""
    source = Path(args.source)
    if not source.exists():
        print(f"source not found: {source}", file=sys.stderr)
        return 2

    frame = pd.read_csv(source)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    mask = (frame["timestamp_utc"] >= pd.Timestamp(args.start, tz="UTC")) & (
        frame["timestamp_utc"] < pd.Timestamp(args.end, tz="UTC")
    )
    sliced = frame.loc[mask].sort_values("timestamp_utc").reset_index(drop=True)
    if sliced.empty:
        print(f"no rows in {args.start}..{args.end}", file=sys.stderr)
        return 3

    if "is_gap" in sliced.columns:
        sliced["is_gap"] = sliced["is_gap"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        sliced["is_gap"] = False

    sliced["split"] = assign_splits(
        sliced,
        train_frac=args.train_frac,
        validate_frac=args.validate_frac,
        test_frac=args.test_frac,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_parquet(out, index=False)

    boundaries = _boundaries(sliced, sliced["split"])
    manifest = {
        "source": str(source),
        "source_rows": int(len(frame)),
        "source_window": [str(frame["timestamp_utc"].min()), str(frame["timestamp_utc"].max())],
        "slice_window": [args.start, args.end],
        "season": "full year" if (args.start, args.end) == (DEFAULT_START, DEFAULT_END) else "custom window",
        "rows": int(len(sliced)),
        "cadence_min": 15,
        "split_policy": "time_ordered_60_20_20",
        "split_fractions": {
            "train": args.train_frac,
            "validate": args.validate_frac,
            "test": args.test_frac,
        },
        "split_boundaries": boundaries,
        "content_hash": frame_content_hash(sliced),
        "output": str(out),
        "columns": list(sliced.columns),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


def prepare_year2022_2023(args: argparse.Namespace) -> int:
    """Build train_2022 / validate_2022 / test_2023 and the QC report."""
    src_2022 = Path(args.source)
    src_2023 = Path(args.source_test)
    for p in (src_2022, src_2023):
        if not p.exists():
            print(f"source not found: {p}", file=sys.stderr)
            return 2

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- the 2022 training window (train + validation only) ---------------
    f22 = _read_source(src_2022, *PROTOCOL_YEAR_2022)
    labels = assign_splits(
        f22,
        train_frac=args.train_frac,
        validate_frac=1.0 - args.train_frac,
        test_frac=0.0,
    )
    f22["split"] = labels
    f22["split_year"] = 2022
    train_2022 = f22.loc[labels == "train"].reset_index(drop=True)
    validate_2022 = f22.loc[labels == "validate"].reset_index(drop=True)

    # -- the 2023 report-only window --------------------------------------
    f23 = _read_source(src_2023, *PROTOCOL_YEAR_2023)
    f23["split"] = "test"
    f23["split_year"] = 2023
    test_2023 = f23.reset_index(drop=True)

    slices = {
        "train_2022": train_2022,
        "validate_2022": validate_2022,
        "test_2023": test_2023,
    }
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name, frame in slices.items():
        path = data_dir / f"dka_alice_springs_{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = str(path)
        hashes[name] = frame_content_hash(frame)

    # -- the combined slice the spine consumes ----------------------------
    combined = (
        pd.concat([train_2022, validate_2022, test_2023], ignore_index=True)
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    combined_path = data_dir / "dka_alice_springs_2022_2023.parquet"
    combined.to_parquet(combined_path, index=False)
    combined_hash = frame_content_hash(combined)

    manifest = {
        "protocol": "year2022_2023",
        "protocol_version": "chariot_protocol_2022_2023_v1.0",
        "description": (
            "TRAIN = the first 60 % of the 2022 window; VALIDATE = the remaining "
            "40 % of the 2022 window; TEST = the whole 2023 window (report only, "
            "never fitted, never selected on).  The 2022 test tail of the legacy "
            "60/20/20 protocol is discarded on purpose."
        ),
        "sources": {
            "train_validate_2022": str(src_2022),
            "test_2023": str(src_2023),
        },
        "windows": {
            "train_validate_2022": list(PROTOCOL_YEAR_2022),
            "test_2023": list(PROTOCOL_YEAR_2023),
        },
        "split_fractions": {
            "train_2022": args.train_frac,
            "validate_2022": 1.0 - args.train_frac,
        },
        "split_boundaries": {
            "train_2022": _boundaries(train_2022, train_2022["split"]),
            "validate_2022": _boundaries(validate_2022, validate_2022["split"]),
            "test_2023": _boundaries(test_2023, test_2023["split"]),
        },
        "content_hashes": {
            **{name: hashes[name] for name in slices},
            "combined": combined_hash,
        },
        "rows": {name: int(len(frame)) for name, frame in slices.items()},
        "rows_combined": int(len(combined)),
        "outputs": {**paths, "combined": str(combined_path)},
        "cadence_min": 15,
        "columns": list(combined.columns),
        "selection_split": "validate_2022",
        "test_split_usage": "report only",
    }
    manifest_path = data_dir / "dka_alice_springs_2022_2023.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # -- QC ----------------------------------------------------------------
    qc_slices = {
        "train_2022": qc_frame(train_2022, rated_pv_w=args.rated_pv_w, label="train_2022"),
        "validate_2022": qc_frame(validate_2022, rated_pv_w=args.rated_pv_w, label="validate_2022"),
        "test_2023": qc_frame(test_2023, rated_pv_w=args.rated_pv_w, label="test_2023"),
    }
    qc = {
        "stage": "W1",
        "protocol": "year2022_2023",
        "slices": qc_slices,
        "content_hashes": manifest["content_hashes"],
        "split_boundaries": manifest["split_boundaries"],
        "notes": [
            "The two years use **different clear-sky models**: the 2022 export was "
            "built with a 15 degree tilt, the 2023 export with a 17.5 degree tilt "
            "(both single-plane equivalents inferred from the measured POA).  That "
            "is a change of the *reference*, not of the site, and it is the first "
            "suspect for the year-to-year shift in the clearness ratio.",
            "The clearness ratio above 1.0 (the 'clear-sky under-binding') is the "
            "known defect this stage quantifies; W2 (the clear-sky upgrade) is "
            "gated on the 2022 validation slice and the 2023 slice is reported once.",
            "The load, battery and SOC channels are **declared** site-card "
            "assumptions; nothing in this stage measures them.",
        ],
    }
    (run_dir / "qc.json").write_text(json.dumps(qc, indent=2, default=str), encoding="utf-8")
    (run_dir / "summary.md").write_text(
        render_qc_markdown(
            qc,
            title="W1 — data + the 2022→2023 protocol (QC)",
            extra=(
                "Protocol: **2022 train → 2022 validate → 2023 test**.\n\n"
                f"Rows: train_2022 {manifest['rows']['train_2022']}, "
                f"validate_2022 {manifest['rows']['validate_2022']}, "
                f"test_2023 {manifest['rows']['test_2023']}.\n\n"
                "Content hashes: "
                + ", ".join(f"`{k}` {v[:16]}…" for k, v in manifest["content_hashes"].items())
            ),
        ),
        encoding="utf-8",
    )

    print(json.dumps({"manifest": str(manifest_path), "qc": summarise(qc)}, indent=2))
    print(f"wrote {run_dir / 'summary.md'} and {run_dir / 'qc.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--protocol", default="single", choices=["single", "year2022_2023"])
    parser.add_argument("--source", required=True, help="source CSV (2022 export)")
    parser.add_argument("--source-test", help="source CSV (2023 export, year2022_2023 only)")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--data-dir", default=str(PROTOCOL_DATA_DIR))
    parser.add_argument("--rated-pv-w", type=float, default=220_000.0)
    parser.add_argument(
        "--train-frac", type=float, default=0.60, help="train fraction (default 0.60)"
    )
    parser.add_argument(
        "--validate-frac", type=float, default=0.20, help="validate fraction (single protocol)"
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.20, help="test fraction (single protocol)"
    )
    args = parser.parse_args()

    if args.protocol == "year2022_2023":
        if not args.source_test:
            print("--source-test is required for the year2022_2023 protocol", file=sys.stderr)
            return 2
        return prepare_year2022_2023(args)
    return prepare_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
