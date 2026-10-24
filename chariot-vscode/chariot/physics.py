"""Deterministic power-flow solver shared by the scenario generator.

This module encodes the *site physics* of the CHARIOT platform:

* PV generation (measured or modelled),
* site load (measured or declared),
* battery storage with charge/discharge efficiencies,
* grid import/export as the residual of the bus balance,
* state-of-charge as an energy-conserving state.

The solver is deliberately kept separate from the Reality Tribunal: the
Tribunal re-derives the energy balance **independently** in
``chariot.tribunal`` so that a bug in this solver cannot silently certify
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BatterySpec:
    """Certified battery ratings and operating reserves."""

    usable_capacity_wh: float
    max_charge_w: float
    max_discharge_w: float
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    #: Operational reserve band the dispatch policy targets (percent).
    soc_floor_pct: float = 5.0
    soc_ceiling_pct: float = 95.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable_capacity_wh": self.usable_capacity_wh,
            "max_charge_w": self.max_charge_w,
            "max_discharge_w": self.max_discharge_w,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
            "soc_floor_pct": self.soc_floor_pct,
            "soc_ceiling_pct": self.soc_ceiling_pct,
        }


def solve_flow(
    solar_w: np.ndarray,
    load_w: np.ndarray,
    soc_initial_pct: float,
    battery: BatterySpec,
    dt_h: float = 0.25,
) -> dict[str, np.ndarray]:
    """Solve the bus balance for a trajectory.

    Dispatch policy (declared operating policy, v1):

    * surplus (``solar - load > 0``) charges the battery up to
      ``max_charge_w`` while the modelled SOC is below the reserve ceiling;
      the remainder is exported,
    * a deficit discharges the battery up to ``max_discharge_w`` while the
      modelled SOC is above the reserve floor; the remainder is imported.

    Grid limits are **not** enforced here: the grid flow is the residual of
    the balance and the Reality Tribunal checks it against the certified
    interconnection ratings.

    Sign conventions: ``battery_w`` is positive when discharging to the bus,
    ``grid_w`` is positive when importing from the grid.
    """
    solar = np.asarray(solar_w, dtype=float)
    load = np.asarray(load_w, dtype=float)
    n = solar.size
    battery_w = np.zeros(n, dtype=float)
    grid_w = np.zeros(n, dtype=float)
    soc = np.empty(n + 1, dtype=float)
    soc[0] = float(np.clip(soc_initial_pct, -1e9, 1e9))

    cap = float(battery.usable_capacity_wh)
    eta_c = float(battery.charge_efficiency)
    eta_d = float(battery.discharge_efficiency)
    floor = float(battery.soc_floor_pct)
    ceiling = float(battery.soc_ceiling_pct)

    for t in range(n):
        net = float(solar[t] - load[t])
        soc_prev = soc[t]
        if net >= 0.0:
            charge = min(net, battery.max_charge_w)
            if soc_prev >= ceiling:
                charge = 0.0
            else:
                # Do not charge past the certified ceiling.
                headroom_w = (ceiling - soc_prev) / 100.0 * cap / max(dt_h * eta_c, 1e-9)
                charge = min(charge, max(headroom_w, 0.0))
            battery_w[t] = -charge
            grid_w[t] = -(net - charge)  # export (negative)
        else:
            deficit = -net
            discharge = min(deficit, battery.max_discharge_w)
            if soc_prev <= floor:
                discharge = 0.0
            else:
                # Do not discharge past the certified floor.
                headroom_w = (soc_prev - floor) / 100.0 * cap * eta_d / max(dt_h, 1e-9)
                discharge = min(discharge, max(headroom_w, 0.0))
            battery_w[t] = discharge
            grid_w[t] = deficit - discharge  # import (positive)

        net_to_batt = float(solar[t] + grid_w[t] - load[t])
        if net_to_batt >= 0.0:
            delta_wh = net_to_batt * dt_h * eta_c
        else:
            delta_wh = net_to_batt * dt_h / eta_d
        soc[t + 1] = soc_prev + delta_wh / cap * 100.0 if cap > 0 else soc_prev

    return {
        "battery_w": battery_w,
        "grid_w": grid_w,
        "soc_percent": soc[1:],
        "soc_initial_percent": np.array([soc[0]]),
    }


def balance_residual_wh(
    solar_w: np.ndarray,
    load_w: np.ndarray,
    grid_w: np.ndarray,
    soc_start_pct: np.ndarray,
    soc_end_pct: np.ndarray,
    usable_capacity_wh: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    dt_h: float = 0.25,
) -> np.ndarray:
    """Per-step absolute energy-balance residual (Wh).

    ``residual_t = | ΔE_reported - ΔE_recomputed |`` where ``ΔE_recomputed``
    is derived from the reported bus flows.
    """
    solar = np.asarray(solar_w, dtype=float)
    load = np.asarray(load_w, dtype=float)
    grid = np.asarray(grid_w, dtype=float)
    soc_start = np.asarray(soc_start_pct, dtype=float)
    soc_end = np.asarray(soc_end_pct, dtype=float)

    delta_reported = (soc_end - soc_start) / 100.0 * float(usable_capacity_wh)
    net_to_batt = solar + grid - load
    delta_recomputed = np.where(
        net_to_batt >= 0.0,
        net_to_batt * dt_h * charge_efficiency,
        net_to_batt * dt_h / discharge_efficiency,
    )
    return np.abs(delta_reported - delta_recomputed)
