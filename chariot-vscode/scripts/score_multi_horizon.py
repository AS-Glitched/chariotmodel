#!/usr/bin/env python3
"""E1b+E2: multi-horizon score, day-origin H1, optional cloud forecast covariate."""

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

HORIZONS = {"H1_4h": 16, "H2_24h": 96, "H3_72h": 288}
DT_H = 0.25
MIN_TRAIN = 96 * 7
N_SCENARIOS = 60
SEED = 7
ORIGIN_HOUR_UTC = 6  # avoid all-night H1 windows


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def energy_wh(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=float) * DT_H))


def score_window(actual, clear, p50, p10, p90) -> dict:
    pers = np.empty_like(actual)
    pers[0] = actual[0]
    pers[1:] = actual[:-1]
    return {
        "mae_persistence": round(mae(actual, pers), 3),
        "mae_clear_sky": round(mae(actual, clear), 3),
        "mae_model_p50": round(mae(actual, p50), 3),
        "energy_mae_clear_sky_wh": round(abs(energy_wh(clear) - energy_wh(actual)), 3),
        "energy_mae_model_p50_wh": round(abs(energy_wh(p50) - energy_wh(actual)), 3),
        "coverage_p10_p90": round(float(np.mean((actual >= p10) & (actual <= p90))), 4),
        "sharpness_mean_w": round(float(np.mean(p90 - p10)), 3),
        "bias_p50": round(float(np.mean(p50 - actual)), 3),
    }


def attach_weather_forecast(clean: pd.DataFrame, t0: int, steps: int, rng: np.random.Generator) -> pd.DataFrame:
    """Operational-style cloud forecast issued at t0 (no meter leakage of future solar).

    Uses true future cloud_cover_pct + noise as a synthetic NWP stand-in when column exists.
    Also builds persistence weather from last observed cloud.
    """
    future = clean.iloc[t0 : t0 + steps].copy()
    if "cloud_cover_pct" not in clean.columns:
        future["cloud_forecast_pct"] = np.nan
        future["cloud_persist_pct"] = np.nan
        return future

    last_cloud = float(clean.iloc[t0 - 1]["cloud_cover_pct"])
    true_cloud = future["cloud_cover_pct"].to_numpy(dtype=float)
    # Pseudo-forecast available at t0: noisy truth (oracle degraded) — label in lineage.
    noise = rng.normal(0.0, 8.0, size=len(true_cloud))
    future["cloud_forecast_pct"] = np.clip(true_cloud + noise, 0.0, 100.0)
    future["cloud_persist_pct"] = np.clip(last_cloud, 0.0, 100.0)
    return future


def main() -> None:
    raw, source = load_raw()
    clean = apply_gate(raw).sort_values("timestamp_utc").reset_index(drop=True)
    rated = float(clean["rated_pv_w"].iloc[0])
    max_h = max(HORIZONS.values())
    hours = pd.to_datetime(clean["timestamp_utc"], utc=True).dt.hour.to_numpy()

    # Origins at 06:00 UTC so H1 includes morning daylight.
    starts = [
        i
        for i in range(MIN_TRAIN, len(clean) - max_h)
        if int(hours[i]) == ORIGIN_HOUR_UTC
    ]
    # Thin to ~daily if 15-min data has multiple 06:00... actually one per day.
    rng = np.random.default_rng(SEED)

    modes = ("no_weather", "weather_forecast")
    summary = {
        "design": "E1b_E2_multi_horizon_weather",
        "data_source": source,
        "origin_hour_utc": ORIGIN_HOUR_UTC,
        "n_origins": len(starts),
        "modes": {},
    }

    for mode in modes:
        per_h = {k: [] for k in HORIZONS}
        for t0 in starts:
            train = clean.iloc[:t0].copy()
            model = fit(train, seed=SEED, use_weather=(mode == "weather_forecast"))

            for name, steps in HORIZONS.items():
                future = attach_weather_forecast(clean, t0, steps, rng)
                if mode == "no_weather":
                    # Ensure model does not see forecast columns as active weather.
                    future = future.copy()
                    future["cloud_forecast_pct"] = np.nan

                actual = future["solar_w"].to_numpy(dtype=float)
                clear = future["clear_sky_w"].to_numpy(dtype=float)

                gate = filter_scenarios(
                    scenarios(model, future, n_scenarios=N_SCENARIOS),
                    rated_pv_w=rated,
                )
                if gate["S_retained"] < 5:
                    per_h[name].append({"t0": str(future["timestamp_utc"].iloc[0]), "status": "ABSTAIN"})
                    continue

                sol = gate["retained"]["solar_w"]
                p50, p10, p90 = [np.quantile(sol, q, axis=0) for q in (0.5, 0.1, 0.9)]
                row = score_window(actual, clear, p50, p10, p90)
                row.update(
                    {
                        "t0": str(future["timestamp_utc"].iloc[0]),
                        "status": "OK",
                        "survival_rate": round(gate["survival_rate"], 4),
                    }
                )
                per_h[name].append(row)

        mode_block = {}
        for name, rows in per_h.items():
            ok = [r for r in rows if r.get("status") == "OK"]

            def mean_key(key, ok=ok):
                return round(float(np.mean([r[key] for r in ok])), 4) if ok else None

            block = {
                "n_scored": len(ok),
                "mean_mae_persistence": mean_key("mae_persistence"),
                "mean_mae_clear_sky": mean_key("mae_clear_sky"),
                "mean_mae_model_p50": mean_key("mae_model_p50"),
                "mean_energy_mae_clear_sky_wh": mean_key("energy_mae_clear_sky_wh"),
                "mean_energy_mae_model_p50_wh": mean_key("energy_mae_model_p50_wh"),
                "mean_coverage": mean_key("coverage_p10_p90"),
                "mean_sharpness_w": mean_key("sharpness_mean_w"),
                "mean_survival": mean_key("survival_rate"),
            }
            if ok and name.startswith("H1"):
                m, p = block["mean_mae_model_p50"], block["mean_mae_persistence"]
                if m is not None and p is not None:
                    block["verdict"] = (
                        "MODEL_BEATS_PERSISTENCE"
                        if m < p
                        else "MODEL_NEAR_PERSISTENCE"
                        if m < 1.15 * p
                        else "PERSISTENCE_STRONGER"
                    )
            elif ok:
                m, c = block["mean_energy_mae_model_p50_wh"], block["mean_energy_mae_clear_sky_wh"]
                block["verdict"] = (
                    "MODEL_BEATS_CLEAR_SKY_ENERGY"
                    if m is not None and c is not None and m < c
                    else "CLEAR_SKY_STRONGER_OR_EQUAL_ENERGY"
                )
            mode_block[name] = block
        summary["modes"][mode] = mode_block

    # Weather lift on H2/H3 energy
    lift = {}
    for h in ("H2_24h", "H3_72h"):
        a = summary["modes"]["no_weather"][h].get("mean_energy_mae_model_p50_wh")
        b = summary["modes"]["weather_forecast"][h].get("mean_energy_mae_model_p50_wh")
        if a and b:
            lift[h] = {
                "no_weather_energy_mae_wh": a,
                "weather_energy_mae_wh": b,
                "improvement_wh": round(a - b, 3),
                "improvement_pct": round(100.0 * (a - b) / a, 2) if a else None,
            }
    summary["weather_lift"] = lift

    out = ROOT / "runs" / "latest" / "score_multi_horizon_e2.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()