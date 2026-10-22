#!/usr/bin/env python3
"""Rolling-origin solar score: many forecast starts, no single-day luck."""

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


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> None:
    raw, source = load_raw()
    clean = apply_gate(raw).sort_values("timestamp_utc").reset_index(drop=True)

    horizon = 96          # 1 day
    min_train = 96 * 7    # at least 7 days history
    step = 96             # move origin by 1 day
    n_scenarios = 60
    seed = 7

    starts = list(range(min_train, len(clean) - horizon, step))
    rows = []

    for t0 in starts:
        train = clean.iloc[:t0].copy()
        future = clean.iloc[t0 : t0 + horizon].copy()
        actual = future["solar_w"].to_numpy(dtype=float)
        clear = future["clear_sky_w"].to_numpy(dtype=float)

        pers = np.empty_like(actual)
        pers[0] = actual[0]
        pers[1:] = actual[:-1]

        model = fit(train, seed=seed)
        gate = filter_scenarios(
            scenarios(model, future, n_scenarios=n_scenarios),
            rated_pv_w=float(clean["rated_pv_w"].iloc[0]),
        )
        if gate["S_retained"] < 5:
            rows.append(
                {
                    "t0": str(future["timestamp_utc"].iloc[0]),
                    "survival_rate": gate["survival_rate"],
                    "status": "ABSTAIN",
                }
            )
            continue

        p50 = np.quantile(gate["retained"]["solar_w"], 0.50, axis=0)
        p10 = np.quantile(gate["retained"]["solar_w"], 0.10, axis=0)
        p90 = np.quantile(gate["retained"]["solar_w"], 0.90, axis=0)
        cover = float(np.mean((actual >= p10) & (actual <= p90)))

        rows.append(
            {
                "t0": str(future["timestamp_utc"].iloc[0]),
                "survival_rate": round(gate["survival_rate"], 4),
                "mae_persistence": round(mae(actual, pers), 3),
                "mae_clear_sky": round(mae(actual, clear), 3),
                "mae_model_p50": round(mae(actual, p50), 3),
                "bias_p50": round(float(np.mean(p50 - actual)), 3),
                "coverage_p10_p90": round(cover, 4),
                "status": "OK",
            }
        )

    ok = [r for r in rows if r.get("status") == "OK"]
    summary = {
        "data_source": source,
        "model_version": "from_fit",
        "n_starts": len(rows),
        "n_scored": len(ok),
        "mean_mae_persistence": round(float(np.mean([r["mae_persistence"] for r in ok])), 3) if ok else None,
        "mean_mae_clear_sky": round(float(np.mean([r["mae_clear_sky"] for r in ok])), 3) if ok else None,
        "mean_mae_model_p50": round(float(np.mean([r["mae_model_p50"] for r in ok])), 3) if ok else None,
        "mean_coverage": round(float(np.mean([r["coverage_p10_p90"] for r in ok])), 4) if ok else None,
        "mean_survival": round(float(np.mean([r["survival_rate"] for r in ok])), 4) if ok else None,
        "starts": rows,
    }
    if ok:
        mp = summary["mean_mae_model_p50"]
        pp = summary["mean_mae_persistence"]
        if mp < pp:
            summary["verdict"] = "MODEL_BEATS_PERSISTENCE_ON_AVERAGE"
        elif mp < 1.15 * pp:
            summary["verdict"] = "MODEL_NEAR_PERSISTENCE_ON_AVERAGE"
        else:
            summary["verdict"] = "PERSISTENCE_STRONGER_ON_AVERAGE"

    out = ROOT / "runs" / "latest" / "score_rolling_origin.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "starts"}, indent=2))
    print(f"\nWrote {out} ({len(rows)} starts)")


if __name__ == "__main__":
    main()