#!/usr/bin/env python3
"""Diagnostics for the CHARIOT clearness channel fit.

Prints the fitted state-space parameters, the implied vs. observed spread of
the clear-sky ratio, the one-step-ahead error, and the ramp statistics of the
simulated ensemble versus the observed record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.model import (  # noqa: E402
    clear_sky_series,
    fit,
    filter_state_series,
    one_step_prediction,
    simulate,
    solar_series,
)
from chariot.pipeline import alice_springs_site  # noqa: E402
from chariot.config import SiteContext  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "var/data/dka_alice_springs_spring2022_15min.parquet")
    frame = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)

    site = alice_springs_site()
    ctx = SiteContext(
        tenant_id=site["tenant_id"],
        site_id=site["site_id"],
        rated_pv_w=site["rated_pv_w"],
        usable_battery_capacity_wh=site["usable_battery_capacity_wh"],
        max_charge_w=site["max_charge_w"],
        max_discharge_w=site["max_discharge_w"],
        timezone=site["timezone"],
        latitude=site["latitude"],
        longitude=site["longitude"],
        fleet_id=site["fleet_id"],
        metadata=site["metadata"],
        source="site_master",
        missing_rating=False,
    )

    train = frame.iloc[: int(len(frame) * 0.6)].reset_index(drop=True)
    model = fit(train, ctx, tenant_id=ctx.tenant_id, site_id=ctx.site_id, fleet_id="dkasc")
    ch = model.sites[ctx.site_id].solar
    print("=== fitted clearness channel ===")
    for key in ("mu", "rho", "sigma", "gamma", "obs_scale", "n_obs"):
        print(f"  {key:12s} {getattr(ch, key):.6f}")
    print("  se:", {k: round(v, 6) for k, v in ch.se.items()})
    print("=== diagnostics ===")
    for key, value in ch.diagnostics.items():
        print(f"  {key:28s} {value}")

    # --- one step ahead check over the validation window -----------------
    filt = filter_state_series(model, ctx.site_id, frame)
    clear = clear_sky_series(frame)
    solar = solar_series(frame)
    mu1 = one_step_prediction(model, ctx.site_id, frame)
    mask = (clear > 1.0) & np.isfinite(solar)
    err = solar[mask] - mu1[mask]
    print("=== one-step-ahead (all rows) ===")
    print(f"  MAE  {np.mean(np.abs(err)):,.0f} W")
    print(f"  RMSE {np.sqrt(np.mean(err**2)):,.0f} W")
    print(f"  bias {np.mean(err):,.0f} W")
    print(f"  persistence MAE {np.mean(np.abs(np.diff(solar[mask]))):,.0f} W (proxy)")

    # --- simulate an ensemble and compare the ramp statistics ------------
    future = frame.iloc[-96:].reset_index(drop=True)
    origin_state = {
        "v_mean": float(filt["filtered"][-97]),
        "v_var": float(filt["filtered_var"][-97]),
    }
    sim = simulate(model, ctx.site_id, future, n_scenarios=300, seed=3, state=origin_state)
    sim_solar = sim["solar_w"]
    ramps = np.abs(np.diff(sim_solar, axis=1))
    obs_ramps = np.abs(np.diff(solar_series(frame)))
    print("=== ramp statistics ===")
    print(f"  observed   p50 {np.percentile(obs_ramps,50):,.0f}  p90 {np.percentile(obs_ramps,90):,.0f}  max {obs_ramps.max():,.0f}")
    print(f"  simulated  p50 {np.percentile(ramps,50):,.0f}  p90 {np.percentile(ramps,90):,.0f}  max {ramps.max():,.0f}")
    print(f"  simulated solar max {sim_solar.max():,.0f} W  (ceiling 1.2*rated = {1.2*220000:,.0f})")
    print(f"  simulated clearness u: p50 {np.percentile(sim['clearness_u'],50):.3f} p99 {np.percentile(sim['clearness_u'],99):.3f} max {sim['clearness_u'].max():.3f}")
    fut_clear = clear_sky_series(future)
    ratio = sim["solar_w"] / np.maximum(fut_clear[None, :], 1.0)
    day = np.broadcast_to(fut_clear[None, :] > 1.0, ratio.shape)
    print(f"  simulated R (daylight): p50 {np.percentile(ratio[day],50):.3f} p99 {np.percentile(ratio[day],99):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
