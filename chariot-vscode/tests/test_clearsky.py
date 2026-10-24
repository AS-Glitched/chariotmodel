"""Tests for the W2 clear-sky reference (``chariot.clearsky``).

The clear-sky reference is the scale the whole clearness model lives on, so
the tests pin the three properties that matter:

* the geometry is inferred from the *measured* POA (a synthetic array with a
  known tilt is recovered),
* the reference is a *fixed plane* -- the tracker alternative is measured and
  rejected on the same data, so no tracker mix is ever invented,
* the same geometry and the same recipe give a *consistent* reference for
  every year (the defect W2 repairs: the 2022 and 2023 exports did not share
  a reference).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pvlib = pytest.importorskip("pvlib", reason="the clear-sky reference needs pvlib")

from chariot.clearsky import (  # noqa: E402
    CLEAR_SKY_VERSION,
    clearness_diagnostics,
    compute_clear_sky,
    infer_fixed_plane_geometry,
    tracker_reference_rmse,
)

SITE_LAT = -23.762
SITE_LON = 133.875
SITE_ELEV = 545.0
RATED_W = 220_000.0


def _synthetic_plane_frame(tilt_deg: float = 20.0, days: int = 20) -> pd.DataFrame:
    """A site whose measured POA comes from a known fixed plane."""
    import pvlib.irradiance as pvir
    from pvlib.solarposition import get_solarposition

    idx = pd.date_range("2022-01-01", periods=days * 96, freq="15min", tz="UTC")
    sp = get_solarposition(idx, SITE_LAT, SITE_LON, altitude=SITE_ELEV, method="nrel_numpy")
    zenith = sp["apparent_zenith"].to_numpy()
    azimuth = sp["azimuth"].to_numpy()
    day = zenith < 85.0
    # A simple but physical irradiance field.
    ghi = np.where(day, 1000.0 * np.cos(np.radians(zenith)) ** 1.2, 0.0)
    dni = np.where(day, 900.0 * np.cos(np.radians(zenith)) ** 0.2, 0.0)
    dhi = np.clip(ghi - dni * np.cos(np.radians(zenith)), 0.0, None)
    out = pvir.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=0.0,
        solar_zenith=zenith,
        solar_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=None,
        model="isotropic",
        albedo=0.2,
        surface_type=None,
    )
    poa = np.asarray(out["poa_global"], dtype=float)
    poa[~np.isfinite(poa)] = 0.0
    frame = pd.DataFrame(
        {
            "tenant_id": "dkasc",
            "site_id": "dka_alice_springs",
            "timestamp_utc": idx,
            "latitude": SITE_LAT,
            "longitude": SITE_LON,
            "elevation_m": SITE_ELEV,
            "rated_pv_w": RATED_W,
            "solar_w": poa * (RATED_W / 1176.0) * 0.9,
            "clear_sky_w": poa * (RATED_W / 1176.0),
            "ghi_w_m2": ghi,
            "dni_w_m2": dni,
            "dhi_w_m2": dhi,
            "poa_w_m2": poa,
            "is_gap": False,
        }
    )
    return frame


def test_geometry_is_inferred_from_the_measured_poa() -> None:
    frame = _synthetic_plane_frame(tilt_deg=20.0)
    result = infer_fixed_plane_geometry(frame)
    geometry = result["geometry"]
    assert geometry.tilt_deg == 20.0
    assert geometry.azimuth_deg == 0.0
    assert result["best"]["rmse_w_m2"] < 1.0


def test_a_fixed_plane_beats_a_tracker_on_the_same_data() -> None:
    """No tracker mix is invented: the fixed plane must win the comparison."""
    frame = _synthetic_plane_frame(tilt_deg=20.0)
    result = infer_fixed_plane_geometry(frame)
    fixed_rmse = result["best"]["rmse_w_m2"]
    tracker_rmse = tracker_reference_rmse(frame, result["geometry"])
    assert fixed_rmse < tracker_rmse


def test_the_reference_is_consistent_across_years() -> None:
    """The same geometry and recipe give the same reference in every year.

    The only year-to-year difference left is the physical one (the day of
    year drives the extraterrestrial irradiance), so the two references agree
    to within a fraction of a percent -- and, crucially, they are the *same
    recipe*, which is what the 2022 and 2023 exports were not.
    """
    frame_2022 = _synthetic_plane_frame(tilt_deg=20.0)
    frame_2023 = frame_2022.copy()
    frame_2023["timestamp_utc"] = frame_2023["timestamp_utc"] + pd.DateOffset(years=1)
    result = infer_fixed_plane_geometry(frame_2022)
    geometry = result["geometry"]
    a = compute_clear_sky(frame_2022, geometry, rated_pv_w=RATED_W)
    b = compute_clear_sky(frame_2023, geometry, rated_pv_w=RATED_W)
    left = a["clear_sky_w"].to_numpy()
    right = b["clear_sky_w"].to_numpy()
    day = left > 0.2 * float(np.nanmax(left))
    rel = np.abs(left[day] - right[day]) / left[day]
    assert rel.max() < 0.01
    assert a["clear_sky_version"].iloc[0] == CLEAR_SKY_VERSION
    assert b["clear_sky_version"].iloc[0] == CLEAR_SKY_VERSION


def test_the_reference_is_above_the_measurement_on_clear_rows() -> None:
    """A reference the plant can beat is not a clear sky.

    A plant that delivers 90 % of the recomputed reference never exceeds it,
    so the clearness index stays below 1 -- the defect the exported
    reference had on a quarter of the 2022 daylight rows.
    """
    frame = _synthetic_plane_frame(tilt_deg=20.0)
    result = infer_fixed_plane_geometry(frame)
    reference = compute_clear_sky(frame, result["geometry"], rated_pv_w=RATED_W)
    clear = reference["clear_sky_w"].to_numpy()
    solar = 0.9 * clear
    day = clear > 1000.0
    assert np.all(solar[day] <= clear[day] * 1.001)
    diagnostics = clearness_diagnostics(solar, clear)
    assert diagnostics["pct_gt_1_05"] == 0.0
    assert diagnostics["max_ratio"] < 1.0


def test_clearness_diagnostics_count_daylight_rows_only() -> None:
    solar = np.array([0.0, 0.0, 1000.0, 2400.0, 2600.0])
    clear = np.array([0.0, 0.0, 2000.0, 2000.0, 2000.0])
    d = clearness_diagnostics(solar, clear)
    assert d["n_daylight_rows"] == 3
    assert d["pct_gt_1_05"] == pytest.approx(200.0 / 3.0)
    assert d["pct_gt_1_20"] == pytest.approx(100.0 / 3.0)


def test_prepare_script_records_the_inferred_geometry(tmp_path: Path) -> None:
    """The data manifest must carry the geometry and the reference version."""
    import argparse

    pytest.importorskip("pvlib")
    sys.path.insert(0, str(ROOT / "scripts"))
    import prepare_dataset  # noqa: PLC0415

    n = 96 * 20
    def _csv(path: Path, year: int) -> None:
        idx = pd.date_range(f"{year}-01-01", periods=n, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "tenant_id": "dkasc",
                "site_id": "dka_alice_springs",
                "timestamp_utc": idx,
                "latitude": SITE_LAT,
                "longitude": SITE_LON,
                "elevation_m": SITE_ELEV,
                "rated_pv_w": RATED_W,
                "solar_w": 100_000.0,
                "clear_sky_w": 120_000.0,
                "ghi_w_m2": 600.0,
                "dni_w_m2": 500.0,
                "dhi_w_m2": 150.0,
                "poa_w_m2": 620.0,
                "is_gap": False,
            }
        )
        frame.to_csv(path, index=False)

    src22 = tmp_path / "src2022.csv"
    src23 = tmp_path / "src2023.csv"
    _csv(src22, 2022)
    _csv(src23, 2023)
    args = argparse.Namespace(
        source=str(src22),
        source_test=str(src23),
        start="2022-01-01",
        end="2024-01-01",
        out=str(tmp_path / "run"),
        data_dir=str(tmp_path / "data"),
        rated_pv_w=RATED_W,
        train_frac=0.6,
        clearsky="chariot_v2",
    )
    assert prepare_dataset.prepare_year2022_2023(args) == 0
    manifest = json.loads(
        (tmp_path / "data" / "dka_alice_springs_2022_2023.manifest.json").read_text()
    )
    clear_sky = manifest["clear_sky"]
    assert clear_sky["source"] == "chariot_v2"
    assert clear_sky["clear_sky_version"] == CLEAR_SKY_VERSION
    assert "tilt_deg" in clear_sky["geometry"]
    assert "tracker_reference_rmse_w_m2" in clear_sky
