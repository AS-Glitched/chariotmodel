#!/usr/bin/env python3
"""Score retained HC-BSSWM solar P50 against persistence on the same window."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.admissibility import apply_gate
from chariot.extract import time_split
from chariot.hcbsswm import fit, scenarios
from chariot.pipeline import load_raw
from chariot.tribunal import filter_scenarios


def mae(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    raw, source = load_raw()
    clean = apply_gate(raw)
    train, test = time_split(clean, train_frac=0.7)
    future = test.iloc[: min(96, len(test))].copy()

    model = fit(train, seed=7)
    draws = scenarios(model, future, n_scenarios=80)
    gate = filter_scenarios(draws, rated_pv_w=float(clean["rated_pv_w"].iloc[0]))

    actual = future["solar_w"].to_numpy(dtype=float)
    clear = future["clear_sky_w"].to_numpy(dtype=float)

    # Persistence on the same window: y_hat(t) = y(t-1)
    pers = np.empty_like(actual)
    pers[0] = actual[0]
    pers[1:] = actual[:-1]

    if gate["S_retained"] == 0:
        print("ABSTAIN: no retained scenarios")
        return

    retained = gate["retained"]["solar_w"]
    p50 = np.quantile(retained, 0.50, axis=0)
    p10 = np.quantile(retained, 0.10, axis=0)
    p90 = np.quantile(retained, 0.90, axis=0)

    # Coverage: fraction of actuals inside P10–P90
    cover = float(np.mean((actual >= p10) & (actual <= p90)))

    report = {
        "data_source": source,
        "window_steps": int(len(future)),
        "window_start": str(future["timestamp_utc"].iloc[0]),
        "window_end": str(future["timestamp_utc"].iloc[-1]),
        "survival_rate": gate["survival_rate"],
        "S_retained": gate["S_retained"],
        "solar_mae": {
            "persistence_1step": round(mae(actual, pers), 3),
            "clear_sky_naive": round(mae(actual, clear), 3),
            "hcbsswm_retained_p50": round(mae(actual, p50), 3),
        },
        "solar_rmse": {
            "persistence_1step": round(rmse(actual, pers), 3),
            "clear_sky_naive": round(rmse(actual, clear), 3),
            "hcbsswm_retained_p50": round(rmse(actual, p50), 3),
        },
        "p10_p90_coverage": round(cover, 4),
        "verdict": None,
    }

    m_model = report["solar_mae"]["hcbsswm_retained_p50"]
    m_pers = report["solar_mae"]["persistence_1step"]
    if m_model < m_pers:
        report["verdict"] = "MODEL_BEATS_PERSISTENCE"
    elif m_model < 1.1 * m_pers:
        report["verdict"] = "MODEL_NEAR_PERSISTENCE"
    else:
        report["verdict"] = "PERSISTENCE_STRONGER"

    out = ROOT / "runs" / "latest" / "score_vs_persistence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()