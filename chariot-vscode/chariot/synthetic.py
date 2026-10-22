"""One-site synthetic telemetry for a sealed demo extract."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_site_frame(days: int = 7, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 96
    idx = pd.date_range("2026-07-20", periods=n, freq="15min", tz="UTC")
    hours = idx.hour + idx.minute / 60.0
    daylight = np.clip((hours - 6.0) / 12.0, 0.0, 1.0)
    clear = np.where((hours >= 6) & (hours <= 18), 5000.0 * np.sin(np.pi * daylight), 0.0)
    solar = np.maximum(0.0, clear * rng.uniform(0.45, 0.92, n) + rng.normal(0, 40, n))
    load = 800.0 + 180.0 * np.sin((hours - 8.0) / 12.0 * np.pi) + rng.normal(0, 35, n)
    soc = np.clip(55.0 + np.cumsum(rng.normal(0, 0.25, n)), 5.0, 95.0)
    return pd.DataFrame(
        {
            "tenant_id": "lab",
            "site_id": "site_1",
            "timestamp_utc": idx,
            "solar_w": solar,
            "load_w": load,
            "soc_percent": soc,
            "clear_sky_w": clear,
            "rated_pv_w": 5000.0,
            "usable_battery_capacity_wh": 10000.0,
            "is_gap": False,
        }
    )
