#!/usr/bin/env python3
"""Prepare the DKASC Alice Springs telemetry for a CHARIOT run.

The source file is the full-year 15-minute solar-only DKASC export (2022,
35 040 rows).  By default the **whole year** is used: the training window
then covers every solar elevation and every season, which matters because
the clear-sky clearness profile is indexed by solar elevation and drifts
seasonally.

The prepared slice is written to ``var/data/`` (git-ignored) together with a
manifest that records the content hash and the deterministic 60/20/20
train/validate/test boundaries, so the split can be verified without shipping
the bulk payload into git.

Usage:
    python scripts/prepare_dataset.py --source <dka_alice_springs_solar_only_15min_2022.csv>
    python scripts/prepare_dataset.py --source <csv> --start 2022-09-01 --end 2022-12-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.hashing import frame_content_hash  # noqa: E402
from chariot.training_extract import assign_splits  # noqa: E402

DEFAULT_START = "2022-01-01"
DEFAULT_END = "2023-01-01"
DEFAULT_OUT = ROOT / "var" / "data" / "dka_alice_springs_2022_15min.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source CSV (full-year export)")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--train-frac", type=float, default=0.60, help="train fraction (default 0.60)"
    )
    parser.add_argument(
        "--validate-frac", type=float, default=0.20, help="validate fraction (default 0.20)"
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.20, help="test fraction (default 0.20)"
    )
    args = parser.parse_args()

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

    boundaries = (
        sliced.groupby("split")["timestamp_utc"].agg(["min", "max", "count"]).to_dict("index")
    )
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
        "split_boundaries": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in boundaries.items()},
        "content_hash": frame_content_hash(sliced),
        "output": str(out),
        "columns": list(sliced.columns),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
