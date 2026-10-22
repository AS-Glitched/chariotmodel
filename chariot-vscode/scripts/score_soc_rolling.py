#!/usr/bin/env python3
"""Rolling-origin SOC (and light load) score for CHARIOT lab Track A."""

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
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main() -> None:
    raw, source = load_raw()
    clean = apply_gate(raw).sort_values("timestamp_utc").reset_index(drop=True)

    horizon = 96
    min_train = 96 * 7
    step = 96
    n_scenarios = 60
    seed = 7
    rated = float(clean["rated_pv_w"].iloc[0])

    starts = list(range(min_train, len(clean) - horizon, step))
    rows = []

    for t0 in starts:
        train = clean.iloc[:t0].copy()
        future = clean.iloc[t0 : t0 + horizon].copy()

        act_soc = future["soc_percent"].to_numpy(dtype=float)
        act_load = future["load_w"].to_numpy(dtype=float)

        pers_soc = np.empty_like(act_soc)
        pers_soc[0] = act_soc[0]
        pers_soc[1:] = act_soc[:-1]

        pers_load = np.empty_like(act_load)
        pers_load[0] = act_load[0]
        pers_load[1:] = act_load[:-1]

        model = fit(train, seed=seed)
        gate = filter_scenarios(
            scenarios(model, future, n_scenarios=n_scenarios),
            rated_pv_w=rated,
        )
        if gate["S_retained"] < 5:
            rows.append({"t0": str(future["timestamp_utc"].iloc[0]), "status": "ABSTAIN",
                         "survival_rate": gate["survival_rate"]})
            continue

        soc = gate["retained"]["soc_percent"]
        load = gate["retained"]["load_w"]
        soc_p50 = np.quantile(soc, 0.50, axis=0)
        soc_p10 = np.quantile(soc, 0.10, axis=0)
        soc_p90 = np.quantile(soc, 0.90, axis=0)
        load_p50 = np.quantile(load, 0.50, axis=0)

        cover = float(np.mean((act_soc >= soc_p10) & (act_soc <= soc_p90)))
        in_bounds = float(np.mean((soc_p50 >= 0.0) & (soc_p50 <= 100.0)))

        rows.append(
            {
                "t0": str(future["timestamp_utc"].iloc[0]),
                "status": "OK",
                "survival_rate": round(gate["survival_rate"], 4),
                "soc_mae_persistence": round(mae(act_soc, pers_soc), 3),
                "soc_mae_model_p50": round(mae(act_soc, soc_p50), 3),
                "soc_bias_p50": round(float(np.mean(soc_p50 - act_soc)), 3),
                "soc_coverage_p10_p90": round(cover, 4),
                "soc_p50_in_bounds_frac": round(in_bounds, 4),
                "load_mae_persistence": round(mae(act_load, pers_load), 3),
                "load_mae_model_p50": round(mae(act_load, load_p50), 3),
            }
        )

    ok = [r for r in rows if r.get("status") == "OK"]

    def mean_key(key):
        return round(float(np.mean([r[key] for r in ok])), 4) if ok else None

    summary = {
        "data_source": source,
        "n_starts": len(rows),
        "n_scored": len(ok),
        "mean_survival": mean_key("survival_rate"),
        "soc": {
            "mean_mae_persistence": mean_key("soc_mae_persistence"),
            "mean_mae_model_p50": mean_key("soc_mae_model_p50"),
            "mean_bias_p50": mean_key("soc_bias_p50"),
            "mean_coverage": mean_key("soc_coverage_p10_p90"),
            "mean_p50_in_bounds": mean_key("soc_p50_in_bounds_frac"),
        },
        "load": {
            "mean_mae_persistence": mean_key("load_mae_persistence"),
            "mean_mae_model_p50": mean_key("load_mae_model_p50"),
        },
        "starts": rows,
    }

    if ok:
        ms = summary["soc"]["mean_mae_model_p50"]
        ps = summary["soc"]["mean_mae_persistence"]
        if ms is not None and ps is not None:
            if ms < ps:
                summary["soc_verdict"] = "MODEL_BEATS_PERSISTENCE_ON_AVERAGE"
            elif ms < 1.25 * ps:
                summary["soc_verdict"] = "MODEL_NEAR_PERSISTENCE_ON_AVERAGE"
            else:
                summary["soc_verdict"] = "PERSISTENCE_STRONGER_ON_AVERAGE"
        ml = summary["load"]["mean_mae_model_p50"]
        pl = summary["load"]["mean_mae_persistence"]
        if ml is not None and pl is not None:
            if ml < pl:
                summary["load_verdict"] = "MODEL_BEATS_PERSISTENCE_ON_AVERAGE"
            elif ml < 1.25 * pl:
                summary["load_verdict"] = "MODEL_NEAR_PERSISTENCE_ON_AVERAGE"
            else:
                summary["load_verdict"] = "PERSISTENCE_STRONGER_ON_AVERAGE"

    out = ROOT / "runs" / "latest" / "score_soc_rolling.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    printable = {k: summary[k] for k in summary if k != "starts"}
    print(json.dumps(printable, indent=2))
    print(f"\nWrote {out} ({len(rows)} starts)")


if __name__ == "__main__":
    main()