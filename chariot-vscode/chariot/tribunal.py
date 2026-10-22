"""Fail-closed Reality Tribunal. Model proposes; physics disposes."""

from __future__ import annotations

import numpy as np


def judge(solar: np.ndarray, load: np.ndarray, soc: np.ndarray, rated_pv_w: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (np.isfinite(solar).all() and np.isfinite(load).all() and np.isfinite(soc).all()):
        reasons.append("NON_FINITE")
    if np.any(solar < -1e-6):
        reasons.append("SOLAR_NEGATIVE")
    if np.any(solar > 1.2 * rated_pv_w):
        reasons.append("SOLAR_OVERSCALE")
    if np.any(soc < 0.0) or np.any(soc > 100.0):
        reasons.append("SOC_BOUNDS")
    if np.any(load < 0.0):
        reasons.append("LOAD_NEGATIVE")
    return (len(reasons) == 0), reasons


def filter_scenarios(bundle: dict[str, np.ndarray], rated_pv_w: float) -> dict:
    solar, load, soc = bundle["solar_w"], bundle["load_w"], bundle["soc_percent"]
    n = solar.shape[0]
    keep: list[int] = []
    rejected: list[dict] = []
    for i in range(n):
        ok, reasons = judge(solar[i], load[i], soc[i], rated_pv_w)
        if ok:
            keep.append(i)
        else:
            rejected.append({"scenario_id": i, "reasons": reasons})
    retained = {
        "solar_w": solar[keep] if keep else solar[:0],
        "load_w": load[keep] if keep else load[:0],
        "soc_percent": soc[keep] if keep else soc[:0],
    }
    return {
        "retained_idx": keep,
        "rejected": rejected,
        "S_total": n,
        "S_retained": len(keep),
        "survival_rate": (len(keep) / n) if n else 0.0,
        "retained": retained,
    }
