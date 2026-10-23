"""Evaluation metrics for CHARIOT advisories.

Everything is computed against the *test* split of the time-ordered
60/20/20 split and is reported with the physics survival rate attached, so a
good point score can never hide a low survival rate.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def _clean(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    return a[mask], p[mask]


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / bias for a point forecast."""
    a, p = _clean(actual, predicted)
    if a.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n": 0}
    err = p - a
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "n": int(a.size),
    }


def band_coverage(actual: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    """Fraction of observations inside the P10-P90 band."""
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(p10, dtype=float)
    hi = np.asarray(p90, dtype=float)
    mask = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    if mask.sum() == 0:
        return float("nan")
    inside = (a[mask] >= lo[mask]) & (a[mask] <= hi[mask])
    return float(np.mean(inside))


def energy_error_wh(actual: np.ndarray, predicted: np.ndarray, dt_h: float) -> float:
    """Absolute error of the integrated energy over the window (Wh)."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(abs(np.sum(p[mask]) - np.sum(a[mask])) * dt_h)


def score_window(
    actual: pd.DataFrame,
    forecast: dict[str, Any],
    *,
    dt_h: float = 0.25,
) -> dict[str, Any]:
    """Score one advisory payload against the realised window."""
    out: dict[str, Any] = {}
    n = len(actual)
    for signal in ("solar_w", "load_w", "soc_percent", "grid_w"):
        bands = forecast.get(signal) or {}
        p50 = np.asarray(bands.get("p50", []), dtype=float)
        p10 = np.asarray(bands.get("p10", []), dtype=float)
        p90 = np.asarray(bands.get("p90", []), dtype=float)
        if p50.size == 0 or signal not in actual.columns:
            out[signal] = {"mae": float("nan"), "rmse": float("nan"), "coverage": float("nan")}
            continue
        truth = pd.to_numeric(actual[signal], errors="coerce").to_numpy(dtype=float)[: p50.size]
        out[signal] = {
            **point_metrics(truth, p50),
            "coverage": band_coverage(truth, p10[: truth.size], p90[: truth.size]),
            "energy_error_wh": energy_error_wh(truth, p50, dt_h),
        }
    out["n_steps"] = int(n)
    return out


def baseline_scoreboard(
    actual: pd.DataFrame,
    *,
    last_solar: float,
    last_load: float,
    last_soc: float,
    train_hour_means: dict[int, float] | None = None,
    dt_h: float = 0.25,
) -> list[dict[str, Any]]:
    """Honest point baselines over the same window."""
    rows: list[dict[str, Any]] = []
    n = len(actual)
    solar_truth = pd.to_numeric(actual["solar_w"], errors="coerce").to_numpy(dtype=float)
    load_truth = (
        pd.to_numeric(actual["load_w"], errors="coerce").to_numpy(dtype=float)
        if "load_w" in actual.columns
        else np.full(n, np.nan)
    )

    rows.append(
        {
            "target": "solar_w",
            "model": "persistence_origin",
            **point_metrics(solar_truth, np.full(n, last_solar)),
        }
    )
    if "clear_sky_w" in actual.columns:
        clear = pd.to_numeric(actual["clear_sky_w"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "target": "solar_w",
                "model": "clear_sky_naive",
                **point_metrics(solar_truth, clear),
                "energy_error_wh": energy_error_wh(solar_truth, clear, dt_h),
            }
        )
    if train_hour_means:
        hours = pd.to_datetime(actual["timestamp_utc"], utc=True).dt.hour.to_numpy()
        clim = np.array([train_hour_means.get(int(h), np.nan) for h in hours])
        rows.append(
            {
                "target": "solar_w",
                "model": "train_hour_climatology",
                **point_metrics(solar_truth, clim),
                "energy_error_wh": energy_error_wh(solar_truth, clim, dt_h),
            }
        )
    if np.isfinite(load_truth).any():
        rows.append(
            {
                "target": "load_w",
                "model": "persistence_origin",
                **point_metrics(load_truth, np.full(n, last_load)),
            }
        )
    return rows


def aggregate_windows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-window metric dicts into a summary."""
    out: dict[str, Any] = {}
    for signal in ("solar_w", "load_w", "soc_percent", "grid_w"):
        entries = [r[signal] for r in rows if signal in r and np.isfinite(r[signal].get("mae", np.nan))]
        if not entries:
            continue
        out[signal] = {
            "mae_mean": float(np.mean([e["mae"] for e in entries])),
            "rmse_mean": float(np.mean([e["rmse"] for e in entries])),
            "bias_mean": float(np.mean([e.get("bias", 0.0) for e in entries])),
            "coverage_mean": float(np.mean([e.get("coverage", np.nan) for e in entries])),
            "energy_error_wh_mean": float(
                np.mean([e.get("energy_error_wh", np.nan) for e in entries])
            ),
            "n_windows": len(entries),
        }
    return out
