"""The CHARIOT clear-sky reference (W2).

Why this module exists
----------------------
The DKASC exports ship a ``clear_sky_w`` column, and the two years do **not**
share a reference: the 2022 export used a 15° fixed plane, the 2023 export a
17.5° one.  That alone makes a year-over-year comparison meaningless: the
clearness the model sees changes because the *reference* changed, not because
the plant changed.

On top of that, the exported reference is physically impossible in places: on
clear days the measured plane-of-array irradiance exceeds the exported
"clear sky", so the clearness index routinely runs above 1 (25.8 % of the
2022 daylight rows above 1.05, with a maximum above 3).  A reference that the
plant beats is not a clear sky.

What this module does
---------------------
It rebuilds the reference from first principles, with an explicit, versioned
recipe that is identical for every year:

1. **Geometry from the data.**  The array layout (fixed / single-axis /
   dual-axis, per-array capacity shares) is *not* in the export, so no
   tracker mix is invented.  Instead the best fixed plane is inferred from
   the *measured* POA of the training year by a grid search over tilt and
   azimuth.  For the Alice Springs export a 15° north-facing plane fits the
   measured POA roughly four times better than any single-axis tracker, so
   the documented equivalent is a single fixed plane.
2. **A published clear-sky model.**  Ineichen/Perez clear-sky irradiance
   with the pvlib Linke-turbidity climatology, transposed to the plane with
   the Hay-Davies model.  The Linke climatology is a static monthly table:
   it carries no information about any particular year, so applying it to the
   report-only year leaks nothing.
3. **A published DC conversion.**  ``clear_sky_w = poa_w_m2 * rated_pv_w /
   1176`` -- the same nominal conversion the export used (rated power
   reached at 1176 W/m2), so the new reference lives on the same scale as
   the old one and the two can be compared directly.

Nothing here is fitted to a measurement of the *target* year: the geometry
comes from the training year's measured POA, and the atmospheric model is a
climatology.

The module is a **data-preparation** step.  It is never imported by the
operational model path, which stays pure NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

#: The version of the clear-sky reference.  Bump it whenever the recipe
#: changes; it is written into the data manifest and the stage reports.
CLEAR_SKY_VERSION = "chariot_clearsky_v2.0_fixed_plane"

#: The nominal irradiance at which the site reaches its rated DC power.
#: This is the conversion the DKASC export used (clear_sky_w = 187.0 x
#: clear_sky_poa_w_m2 for a 220 kW array).
STC_IRRADIANCE_W_M2 = 1176.0

#: Default search grid for the fixed-plane geometry (degrees).
DEFAULT_TILTS: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
DEFAULT_AZIMUTHS: tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)

#: The transposition model used for the geometry search (fast, stable) and
#: the model used for the clear-sky reference itself.
SEARCH_TRANSPOSITION_MODEL = "isotropic"
REFERENCE_TRANSPOSITION_MODEL = "haydavies"


@dataclass(frozen=True)
class FixedPlaneGeometry:
    """A single fixed plane, expressed as tilt and compass azimuth."""

    tilt_deg: float
    azimuth_deg: float
    transposition_model: str = REFERENCE_TRANSPOSITION_MODEL
    clear_sky_model: str = "ineichen"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tilt_deg": self.tilt_deg,
            "azimuth_deg": self.azimuth_deg,
            "azimuth_convention": "0 = north, positive = east",
            "transposition_model": self.transposition_model,
            "clear_sky_model": self.clear_sky_model,
            "stc_irradiance_w_m2": STC_IRRADIANCE_W_M2,
            "clear_sky_version": CLEAR_SKY_VERSION,
        }


def _require_pvlib():
    try:
        import pvlib  # noqa: PLC0415
        from pvlib import irradiance as pvir  # noqa: PLC0415
        from pvlib import clearsky as pvcs  # noqa: PLC0415
        from pvlib.atmosphere import get_absolute_airmass  # noqa: PLC0415
        from pvlib.solarposition import get_solarposition  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the CHARIOT clear-sky reference needs pvlib "
            "(pip install pvlib==0.16.1); the operational model path does not"
        ) from exc
    return pvlib, pvir, pvcs, get_absolute_airmass, get_solarposition


def _solar_position(frame: pd.DataFrame, get_solarposition):
    lat = float(frame["latitude"].iloc[0])
    lon = float(frame["longitude"].iloc[0])
    elev = float(frame["elevation_m"].iloc[0]) if "elevation_m" in frame.columns else 0.0
    return get_solarposition(
        frame["timestamp_utc"], lat, lon, altitude=elev, method="nrel_numpy"
    )


def _poa_from_components(
    pvir,
    *,
    surface_tilt: np.ndarray | float,
    surface_azimuth: np.ndarray | float,
    solar_zenith: np.ndarray,
    solar_azimuth: np.ndarray,
    dni: np.ndarray,
    ghi: np.ndarray,
    dhi: np.ndarray,
    dni_extra: np.ndarray | None,
    model: str,
) -> np.ndarray:
    out = pvir.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        model=model,
        albedo=0.2,
        surface_type=None,
    )
    poa = np.asarray(out["poa_global"], dtype=float)
    poa[~np.isfinite(poa)] = 0.0
    return poa


def infer_fixed_plane_geometry(
    frame: pd.DataFrame,
    *,
    tilts: Iterable[float] = DEFAULT_TILTS,
    azimuths: Iterable[float] = DEFAULT_AZIMUTHS,
    min_poa_w_m2: float = 5.0,
) -> dict[str, Any]:
    """Infer the single fixed plane that best reproduces the measured POA.

    The search uses the *measured* plane-of-array irradiance of the frame it
    is given, so it must be called with the training year only.  The result
    is the documented single-plane equivalent of the (unknown) array layout.
    """
    _, pvir, _, get_absolute_airmass, get_solarposition = _require_pvlib()
    if "poa_w_m2" not in frame.columns:
        raise ValueError("the geometry search needs the measured poa_w_m2 column")

    sp = _solar_position(frame, get_solarposition)
    zen = sp["apparent_zenith"].to_numpy(dtype=float)
    az = sp["azimuth"].to_numpy(dtype=float)
    ghi = np.nan_to_num(frame["ghi_w_m2"].to_numpy(dtype=float), nan=0.0)
    dni = np.nan_to_num(frame["dni_w_m2"].to_numpy(dtype=float), nan=0.0)
    dhi = np.nan_to_num(frame["dhi_w_m2"].to_numpy(dtype=float), nan=0.0)
    measured = frame["poa_w_m2"].to_numpy(dtype=float)
    day = np.isfinite(measured) & (measured > min_poa_w_m2)
    if day.sum() < 100:
        raise ValueError("not enough daylight rows to infer the plane geometry")

    scored: list[dict[str, Any]] = []
    for tilt in tilts:
        for azimuth in azimuths:
            poa = _poa_from_components(
                pvir,
                surface_tilt=float(tilt),
                surface_azimuth=float(azimuth),
                solar_zenith=zen,
                solar_azimuth=az,
                dni=dni,
                ghi=ghi,
                dhi=dhi,
                dni_extra=None,
                model=SEARCH_TRANSPOSITION_MODEL,
            )
            err = poa[day] - measured[day]
            scored.append(
                {
                    "tilt_deg": float(tilt),
                    "azimuth_deg": float(azimuth),
                    "rmse_w_m2": float(np.sqrt(np.mean(err**2))),
                    "bias_w_m2": float(np.mean(err)),
                    "n_rows": int(day.sum()),
                }
            )
    scored.sort(key=lambda r: (r["rmse_w_m2"], abs(r["bias_w_m2"])))
    best = scored[0]
    geometry = FixedPlaneGeometry(
        tilt_deg=best["tilt_deg"],
        azimuth_deg=best["azimuth_deg"],
    )
    return {
        "geometry": geometry,
        "best": best,
        "grid": scored[:10],
        "search": {
            "model": SEARCH_TRANSPOSITION_MODEL,
            "target": "measured poa_w_m2",
            "n_rows": int(day.sum()),
        },
    }


def compute_clear_sky(
    frame: pd.DataFrame,
    geometry: FixedPlaneGeometry,
    *,
    rated_pv_w: float | None = None,
) -> pd.DataFrame:
    """Compute the clear-sky reference for ``frame``.

    Returns a frame with ``clear_sky_poa_w_m2``, ``clear_sky_w`` and ``kt``
    (the clearness index ``solar_w / clear_sky_w``).
    """
    _, pvir, pvcs, get_absolute_airmass, get_solarposition = _require_pvlib()
    times = frame["timestamp_utc"]
    lat = float(frame["latitude"].iloc[0])
    lon = float(frame["longitude"].iloc[0])
    elev = float(frame["elevation_m"].iloc[0]) if "elevation_m" in frame.columns else 0.0

    sp = _solar_position(frame, get_solarposition)
    zen = sp["apparent_zenith"].to_numpy(dtype=float)
    airmass = get_absolute_airmass(zen, pressure=elev * 101.325 / 273.15)

    # The Linke turbidity climatology is a static monthly table (no
    # year-specific information), so the same reference is produced for
    # every year.
    index = pd.DatetimeIndex(times)
    linke = np.asarray(pvcs.lookup_linke_turbidity(index, lat, lon), dtype=float)
    cs = pvcs.ineichen(zen, airmass, linke, altitude=elev)

    doy = index.day_of_year.to_numpy(dtype=float)
    dni_extra = 1367.0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0))

    poa = _poa_from_components(
        pvir,
        surface_tilt=geometry.tilt_deg,
        surface_azimuth=geometry.azimuth_deg,
        solar_zenith=zen,
        solar_azimuth=sp["azimuth"].to_numpy(dtype=float),
        dni=np.asarray(cs["dni"], dtype=float),
        ghi=np.asarray(cs["ghi"], dtype=float),
        dhi=np.asarray(cs["dhi"], dtype=float),
        dni_extra=dni_extra,
        model=geometry.transposition_model,
    )

    if rated_pv_w is None:
        rated_pv_w = float(frame["rated_pv_w"].iloc[0])
    scale = rated_pv_w / STC_IRRADIANCE_W_M2
    clear_w = poa * scale

    solar = pd.to_numeric(frame["solar_w"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(clear_w > 1.0, solar / clear_w, np.nan)

    out = pd.DataFrame(index=frame.index)
    out["clear_sky_poa_w_m2"] = poa
    out["clear_sky_w"] = clear_w
    out["kt"] = kt
    out["clear_sky_version"] = CLEAR_SKY_VERSION
    return out


def clearness_diagnostics(
    solar_w: np.ndarray, clear_sky_w: np.ndarray, *, min_clear_sky_w: float = 1000.0
) -> dict[str, Any]:
    """The W2 diagnostics: the daylight clearness distribution."""
    solar = np.asarray(solar_w, dtype=float)
    clear = np.asarray(clear_sky_w, dtype=float)
    day = np.isfinite(solar) & np.isfinite(clear) & (clear > min_clear_sky_w)
    if not day.any():
        return {"n_daylight_rows": 0}
    ratio = solar[day] / clear[day]
    ratio = ratio[np.isfinite(ratio)]
    return {
        "n_daylight_rows": int(day.sum()),
        "pct_gt_1_05": float(np.mean(ratio > 1.05) * 100.0),
        "pct_gt_1_20": float(np.mean(ratio > 1.20) * 100.0),
        "p50_ratio": float(np.percentile(ratio, 50)),
        "p95_ratio": float(np.percentile(ratio, 95)),
        "max_ratio": float(np.max(ratio)),
        "mae_w": float(np.mean(np.abs(solar[day] - clear[day]))),
    }


def tracker_reference_rmse(frame: pd.DataFrame, geometry: FixedPlaneGeometry) -> float:
    """RMSE of a single-axis tracker against the measured POA.

    Reported next to the fixed-plane fit so the "no tracker mix" decision is
    evidence-based rather than an assumption.
    """
    _, pvir, _, _, get_solarposition = _require_pvlib()
    sp = _solar_position(frame, get_solarposition)
    zen = sp["apparent_zenith"].to_numpy(dtype=float)
    az = sp["azimuth"].to_numpy(dtype=float)
    ghi = np.nan_to_num(frame["ghi_w_m2"].to_numpy(dtype=float), nan=0.0)
    dni = np.nan_to_num(frame["dni_w_m2"].to_numpy(dtype=float), nan=0.0)
    dhi = np.nan_to_num(frame["dhi_w_m2"].to_numpy(dtype=float), nan=0.0)
    measured = frame["poa_w_m2"].to_numpy(dtype=float)
    day = np.isfinite(measured) & (measured > 5.0)
    # A horizontal north-south axis tracker: the surface follows the sun.
    tilt = np.clip(90.0 - zen, 0.0, 60.0)
    poa = _poa_from_components(
        pvir,
        surface_tilt=tilt,
        surface_azimuth=az,
        solar_zenith=zen,
        solar_azimuth=az,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=None,
        model=SEARCH_TRANSPOSITION_MODEL,
    )
    err = poa[day] - measured[day]
    return float(np.sqrt(np.mean(err**2)))


def _degrees(value: float) -> float:  # pragma: no cover - tiny helper
    return float(math.degrees(value))
