#!/usr/bin/env python3
"""DSM-style block error — daylight only (clear_sky_w > 50)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.admissibility import apply_gate
from chariot.hcbsswm import fit, scenarios
from chariot.pipeline import load_raw
from chariot.tribunal import filter_scenarios

HORIZON = 96
MIN_TRAIN = 96 * 7
N_SCENARIOS = 60
SEED = 7
ORIGIN_HOUR_UTC = 6


def main() -> None:
    raw, source = load_raw()
    clean = apply_gate(raw).sort_values("timestamp_utc").reset_index(drop=True)
    rated = float(clean["rated_pv_w"].iloc[0])
    hours = pd.to_datetime(clean["timestamp_utc"], utc=True).dt.hour.to_numpy()

    starts = [
        i
        for i in range(MIN_TRAIN, len(clean) - HORIZON)
        if int(hours[i]) == ORIGIN_HOUR_UTC
    ]

    block_errors: list[float] = []
    origin_rows = []

    for t0 in starts:
        train = clean.iloc[:t0].copy()
        future = clean.iloc[t0 : t0 + HORIZON].copy()
        actual = future["solar_w"].to_numpy(dtype=float)
        clear = future["clear_sky_w"].to_numpy(dtype=float)

        model = fit(train, seed=SEED, use_weather=True)
        gate = filter_scenarios(
            scenarios(model, future, n_scenarios=N_SCENARIOS),
            rated_pv_w=rated,
        )
        if gate["S_retained"] < 5:
            origin_rows.append(
                {
                    "t0": str(future["timestamp_utc"].iloc[0]),
                    "status": "ABSTAIN",
                    "survival_rate": gate["survival_rate"],
                }
            )
            continue

        p50 = np.quantile(gate["retained"]["solar_w"], 0.5, axis=0)
        day = clear > 50.0
        if not np.any(day):
            origin_rows.append(
                {
                    "t0": str(future["timestamp_utc"].iloc[0]),
                    "status": "NO_DAYLIGHT",
                    "survival_rate": round(gate["survival_rate"], 4),
                }
            )
            continue

        err_pct = 100.0 * np.abs(p50[day] - actual[day]) / max(rated, 1.0)
        block_errors.extend(err_pct.tolist())

        origin_rows.append(
            {
                "t0": str(future["timestamp_utc"].iloc[0]),
                "status": "OK",
                "survival_rate": round(gate["survival_rate"], 4),
                "n_daylight_blocks": int(day.sum()),
                "mean_block_err_pct": round(float(np.mean(err_pct)), 4),
                "pct_blocks_within_5": round(float(np.mean(err_pct <= 5.0)), 4),
                "pct_blocks_within_10": round(float(np.mean(err_pct <= 10.0)), 4),
                "pct_blocks_within_15": round(float(np.mean(err_pct <= 15.0)), 4),
            }
        )

    arr = np.asarray(block_errors, dtype=float)
    ok = [r for r in origin_rows if r.get("status") == "OK"]

    summary = {
        "design": "lab_dsm_proxy_v0_daylight",
        "definition": "100 * |P50 - actual_solar_w| / rated_pv_w per 15-min daylight block",
        "daylight_rule": "clear_sky_w > 50",
        "note": "Lab proxy using rated_pv as AvC stand-in; not CERC settlement-grade; night blocks excluded",
        "data_source": source,
        "model": "hcbsswm_v1.7_residual_ar",
        "use_weather": False,
        "n_origins": len(starts),
        "n_scored_origins": len(ok),
        "n_blocks": int(arr.size),
        "mean_block_err_pct": round(float(np.mean(arr)), 4) if arr.size else None,
        "median_block_err_pct": round(float(np.median(arr)), 4) if arr.size else None,
        "pct_blocks_within_5": round(float(np.mean(arr <= 5.0)), 4) if arr.size else None,
        "pct_blocks_within_10": round(float(np.mean(arr <= 10.0)), 4) if arr.size else None,
        "pct_blocks_within_15": round(float(np.mean(arr <= 15.0)), 4) if arr.size else None,
        "origins": origin_rows,
    }

    out = ROOT / "runs" / "latest" / "score_dsm_blocks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    printable = {k: v for k, v in summary.items() if k != "origins"}
    print(json.dumps(printable, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()