"""Three advisory modes from Tribunal-retained scenarios only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bands(x: np.ndarray) -> dict[str, list[float]]:
    if x.size == 0:
        return {"p10": [], "p50": [], "p90": []}
    return {
        "p10": np.quantile(x, 0.10, axis=0).round(2).tolist(),
        "p50": np.quantile(x, 0.50, axis=0).round(2).tolist(),
        "p90": np.quantile(x, 0.90, axis=0).round(2).tolist(),
    }


def _status(survival_rate: float) -> str:
    if survival_rate < 0.2:
        return "ABSTAIN"
    if survival_rate < 0.6:
        return "REDUCED_CONFIDENCE"
    return "OK"


def build(
    future: pd.DataFrame,
    gate: dict,
    baselines: list[dict],
    lineage: dict,
) -> dict:
    status = _status(gate["survival_rate"])
    retained = gate["retained"]
    empty = status == "ABSTAIN" or gate["S_retained"] == 0
    solar = retained["solar_w"]
    load = retained["load_w"]
    soc = retained["soc_percent"]
    common = {
        "status": status,
        "survival_rate": round(float(gate["survival_rate"]), 4),
        "S_total": gate["S_total"],
        "S_retained": gate["S_retained"],
        "advisory_only": True,
        "lineage": lineage,
        "timestamps": [str(t) for t in future["timestamp_utc"]],
    }
    if empty:
        abstain = {**common, "message": "Physics survival too low. No advisory numbers."}
        return {"grid_engineer": abstain, "researcher": {**abstain, "baselines": baselines, "rejection_sample": gate["rejected"][:12]}, "investor_planner": abstain}

    soc_p10 = np.quantile(soc, 0.10, axis=0)
    low_soc_prob = float(np.mean(np.min(soc, axis=1) < 20.0))
    import_proxy = np.maximum(0.0, load - solar)
    yield_kwh = float(np.mean(solar.sum(axis=1)) * 0.25 / 1000.0)

    grid = {
        **common,
        "horizon": "operational",
        "solar_w": _bands(solar),
        "load_w": _bands(load),
        "soc_percent": _bands(soc),
        "events": {
            "p_soc_below_20": round(low_soc_prob, 4),
            "min_soc_p10": round(float(np.min(soc_p10)), 2),
            "mean_import_proxy_w_p50": round(float(np.median(import_proxy.mean(axis=1))), 2),
        },
    }
    researcher = {
        **common,
        "baselines": baselines,
        "rejection_count": len(gate["rejected"]),
        "rejection_sample": gate["rejected"][:12],
        "notes": [
            "HC-BSSWM v1 is single-site clear-sky state-space inference.",
            "Quantiles use Tribunal-retained scenarios only.",
            "Evaluation split is time-ordered.",
        ],
    }
    investor = {
        **common,
        "horizon": "planning_aggregate",
        "expected_solar_yield_kwh": round(yield_kwh, 3),
        "solar_yield_p10_kwh": round(float(np.quantile(solar.sum(axis=1), 0.1) * 0.25 / 1000.0), 3),
        "solar_yield_p90_kwh": round(float(np.quantile(solar.sum(axis=1), 0.9) * 0.25 / 1000.0), 3),
        "site_confidence": status,
        "not_a_financial_guarantee": True,
    }
    return {"grid_engineer": grid, "researcher": researcher, "investor_planner": investor}
