"""Honest point baselines on a time-ordered test window."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def scoreboard(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shifted = test.copy()
    for col in ("solar_w", "load_w", "soc_percent"):
        pred = shifted[col].shift(1)
        mask = pred.notna()
        rows.append(
            {
                "target": col,
                "model": "persistence_1step",
                "mae": _mae(shifted.loc[mask, col], pred[mask]),
                "rmse": _rmse(shifted.loc[mask, col], pred[mask]),
            }
        )
    rows.append(
        {
            "target": "solar_w",
            "model": "clear_sky_naive",
            "mae": _mae(test["solar_w"], test["clear_sky_w"]),
            "rmse": _rmse(test["solar_w"], test["clear_sky_w"]),
        }
    )
    return pd.DataFrame(rows)
