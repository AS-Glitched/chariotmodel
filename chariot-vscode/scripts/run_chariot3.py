#!/usr/bin/env python3
"""Run the CHARIOT v3 end-to-end spine.

The spine is:

    Telemetry → Site Master → Admissibility → River Flow → Calibration →
    Capsule → Training Extract → CHARIOT model → Residual (capped) →
    Scenarios → Reality Tribunal / Rejection Echo → Confidence → Advisory

Usage (full DKASC Alice Springs 2022 year, prepared with
``scripts/prepare_dataset.py``):

    python scripts/run_chariot3.py

    python scripts/run_chariot3.py \\
        --source var/data/dka_alice_springs_2022_15min.parquet \\
        --out    runs/chariot3_alice_springs_2022 \\
        --scenarios 500 --max-windows 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.pipeline import PipelineConfig, run  # noqa: E402

DEFAULT_SOURCE = ROOT / "var" / "data" / "dka_alice_springs_2022_15min.parquet"
DEFAULT_OUT = ROOT / "runs" / "chariot3_alice_springs_2022"
DEFAULT_STORE = ROOT / "var" / "chariot.db"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="prepared parquet slice (or the raw CSV export)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="run output directory")
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="SQLite lineage store")
    parser.add_argument("--scenarios", type=int, default=500, help="scenarios per batch")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--horizon",
        default="operational",
        choices=["short", "operational", "long"],
        help="advisory horizon (must not exceed the certified trajectory length)",
    )
    parser.add_argument("--max-windows", type=int, default=None, help="cap the number of windows")
    parser.add_argument("--no-residual", action="store_true", help="disable the Residual Chamber")
    parser.add_argument(
        "--no-spread-calibration",
        action="store_true",
        help="skip validation-fitted ensemble spread calibration",
    )
    parser.add_argument("--keep-parquet", action="store_true", help="keep the materialised dataset")
    parser.add_argument(
        "--evaluate-split",
        default="test",
        choices=["test", "validate", "train"],
        help="split the rolling-origin evaluation windows are drawn from",
    )
    parser.add_argument("--start", default=None, help="analysis window start (ISO)")
    parser.add_argument("--end", default=None, help="analysis window end (ISO, exclusive)")
    parser.add_argument(
        "--honour-split",
        action="store_true",
        help=(
            "honour the split assigned in the prepared slice instead of "
            "re-splitting it chronologically (the 2022 -> 2023 protocol: "
            "train/validate in 2022, report-only test in 2023)"
        ),
    )
    args = parser.parse_args()

    config = PipelineConfig(
        source_csv=args.source,
        **({"start": args.start} if args.start else {}),
        **({"end": args.end} if args.end else {}),
        out_dir=args.out,
        store_path=args.store,
        n_scenarios=args.scenarios,
        seed=args.seed,
        horizon=args.horizon,
        max_windows=args.max_windows,
        use_residual=not args.no_residual,
        calibrate_spread=not args.no_spread_calibration,
        keep_parquet=args.keep_parquet,
        honour_split=args.honour_split,
        evaluate_split=args.evaluate_split,
    )
    manifest = run(config)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
