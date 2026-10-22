#!/usr/bin/env python3
"""Run the CHARIOT lab prototype and print the manifest."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.pipeline import run  # noqa: E402


def main() -> None:
    manifest = run(out_dir=ROOT / "runs" / "latest", days=7, n_scenarios=80, seed=7)
    print(json.dumps(manifest, indent=2))
    print("\nWrote runs/latest/{manifest,grid_engineer,researcher,investor_planner}.json")


if __name__ == "__main__":
    main()
