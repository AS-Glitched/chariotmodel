"""Tests for the W3 clearness calibration (``chariot.clearness_calibration``).

The calibration is a bin-wise level correction on the clearness ratio.  The
tests pin the three properties that make it safe:

* it is **off** unless it is asked for (the v3 behaviour is unchanged),
* the correction is a *level* correction on a ratio and it never breaks the
  clearness cap,
* the shrinkage selection is a pure function of the validation score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chariot.clearness_calibration import (  # noqa: E402
    CALIBRATION_VERSION,
    ClearnessCalibration,
    apply_calibration_to_paths,
    apply_shrinkage,
    calibration_multiplier,
)


def _calibration(offset: float = 0.1) -> ClearnessCalibration:
    offsets = tuple(tuple(float(offset) for _ in range(6)) for _ in range(24))
    return ClearnessCalibration(
        version=CALIBRATION_VERSION,
        offsets=offsets,
        global_offset=offset,
        shrinkage=1.0,
        n_rows=100,
    )


def test_an_inactive_calibration_is_a_no_op() -> None:
    cal = ClearnessCalibration(shrinkage=0.0)
    assert cal.active is False
    frame = pd.DataFrame({"timestamp_utc": pd.date_range("2022-01-01", periods=4, freq="15min", tz="UTC")})
    mult = calibration_multiplier(cal, frame, np.array([10.0, 30.0, 50.0, 70.0]))
    assert np.allclose(mult, 1.0)


def test_the_multiplier_follows_the_hour_and_elevation_bins() -> None:
    cal = _calibration(0.1)
    frame = pd.DataFrame(
        {"timestamp_utc": pd.date_range("2022-01-01 05:00", periods=2, freq="12h", tz="UTC")}
    )
    mult = calibration_multiplier(cal, frame, np.array([5.0, 50.0]))
    assert np.allclose(mult, np.exp(0.1))


def test_the_clearness_cap_survives_the_correction() -> None:
    """A calibrated scenario can never claim more than the clear sky."""
    clear = np.array([100_000.0, 100_000.0])
    solar = np.array([[150_000.0, 150_000.0]])  # ratio 1.5 -> above the cap
    out = apply_calibration_to_paths(
        solar, np.array([2.0, 2.0]), clear, clearness_max=1.25
    )
    assert np.allclose(out[0], 1.25 * clear)
    # A night column (no clear sky) is left alone, not zeroed.
    clear_night = np.array([0.0, 100_000.0])
    out2 = apply_calibration_to_paths(
        np.array([[10_000.0, 150_000.0]]), np.array([2.0, 2.0]), clear_night, clearness_max=1.25
    )
    assert out2[0, 0] == 20_000.0
    assert out2[0, 1] == 1.25 * 100_000.0


def test_shrinkage_scales_the_offsets_towards_the_global_mean() -> None:
    cal = ClearnessCalibration(
        offsets=tuple(tuple(0.2 for _ in range(6)) for _ in range(24)),
        global_offset=0.0,
        shrinkage=1.0,
    )
    half = apply_shrinkage(cal, 0.5)
    assert np.allclose(np.asarray(half.offsets), 0.1)
    none = apply_shrinkage(cal, 0.0)
    assert np.allclose(np.asarray(none.offsets), 0.0)
    assert none.active is False


def test_the_calibration_round_trips_through_a_dict() -> None:
    cal = _calibration(0.05)
    payload = cal.to_dict()
    restored = ClearnessCalibration.from_dict(payload)
    assert restored.shrinkage == cal.shrinkage
    assert np.allclose(np.asarray(restored.offsets), np.asarray(cal.offsets))
    assert restored.version == CALIBRATION_VERSION


def test_the_pipeline_default_has_no_calibration(pipeline_run: dict) -> None:
    """The v3 default path is untouched: no calibration is fitted."""
    assert pipeline_run["clearness_calibration"]["enabled"] is False


def test_the_pipeline_can_fit_the_calibration(
    reference_parquet: Path, tmp_path: Path, pipeline_run: dict
) -> None:
    """With the flag on, the calibration is fitted on train and selected on validate."""
    from chariot.pipeline import PipelineConfig, run

    config = PipelineConfig(
        source_csv=reference_parquet,
        out_dir=tmp_path / "run",
        store_path=tmp_path / "store" / "chariot.db",
        n_scenarios=40,
        seed=11,
        horizon="operational",
        max_windows=2,
        spread_windows=1,
        spread_scenarios=20,
        clearness_calibration=True,
        calibration_scenarios=20,
    )
    manifest = run(config)
    info = manifest["clearness_calibration"]
    assert info["enabled"] is True
    assert info["fitted_on"].startswith("train")
    assert info["selected_on"] == "validate_2022"
    assert info["selected_shrinkage"] in (0.0, 0.25, 0.5, 0.75, 1.0)
    # The calibration is part of the model, so the version must move.
    assert manifest["model"]["model_version"] != pipeline_run["model"]["model_version"]
