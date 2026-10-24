"""Tests for the W1 dataset QC module (``chariot.dataset_qc``).

The QC is descriptive: it must report what the data contains without ever
changing a model, a split or a prediction.  These tests pin the behaviour the
2022 -> 2023 protocol relies on:

* the cadence check catches duplicates / broken steps,
* the night check catches a non-zero night solar,
* the clear-sky check catches a copy of the measurement,
* the kt-over-one counters report the *daylight* share (never night),
* the markdown renderer produces a table with the headline numbers.
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

from chariot.dataset_qc import qc_frame, render_qc_markdown, summarise  # noqa: E402


def _frame(n: int = 96) -> pd.DataFrame:
    idx = pd.date_range("2022-06-01", periods=n, freq="15min", tz="UTC")
    clear = np.where(np.arange(n) % 96 < 40, 150_000.0, 0.0)
    solar = clear * 0.9
    return pd.DataFrame(
        {
            "timestamp_utc": idx,
            "solar_w": solar,
            "clear_sky_w": clear,
            "rated_pv_w": 220_000.0,
            "is_gap": False,
        }
    )


def test_cadence_ok_on_a_clean_frame() -> None:
    qc = qc_frame(_frame())
    assert qc["cadence"]["cadence_ok"] is True
    assert qc["cadence"]["duplicates"] == 0
    assert qc["cadence"]["non_15min_steps"] == 0


def test_cadence_flags_duplicate_and_broken_steps() -> None:
    frame = _frame()
    dup = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    qc_dup = qc_frame(dup)
    assert qc_dup["cadence"]["duplicates"] == 1
    assert qc_dup["cadence"]["cadence_ok"] is False

    broken = frame.copy()
    broken.loc[10, "timestamp_utc"] = broken.loc[10, "timestamp_utc"] + pd.Timedelta("20min")
    qc_broken = qc_frame(broken)
    assert qc_broken["cadence"]["non_15min_steps"] >= 1
    assert qc_broken["cadence"]["cadence_ok"] is False


def test_night_solar_must_be_zero() -> None:
    frame = _frame()
    frame.loc[95, "solar_w"] = 12.0  # night row (clear sky is zero there)
    qc = qc_frame(frame)
    assert qc["night"]["night_nonzero_rows"] == 1
    assert qc["night"]["max_night_solar_w"] > 0
    assert qc["night"]["ok"] is False


def test_clear_sky_must_differ_from_the_measurement() -> None:
    frame = _frame()
    frame["clear_sky_w"] = frame["solar_w"]
    qc = qc_frame(frame)
    assert qc["clear_sky_distinct"]["identical_to_solar"] is True
    assert qc["clear_sky_distinct"]["ok"] is False


def test_kt_over_one_counts_daylight_rows_only() -> None:
    frame = _frame()
    # 40 daylight rows; lift 10 of them above 1.05 and 4 above 1.20.
    day = frame.index[frame["clear_sky_w"] > 1.0]
    frame.loc[day[:10], "solar_w"] = frame.loc[day[:10], "clear_sky_w"] * 1.10
    frame.loc[day[:4], "solar_w"] = frame.loc[day[:4], "clear_sky_w"] * 1.25
    qc = qc_frame(frame)
    kt = qc["kt_over_one"]
    assert kt["n_daylight_rows"] == 40
    assert kt["pct_gt_1_05"] == pytest.approx(25.0, abs=1e-6)
    assert kt["pct_gt_1_20"] == pytest.approx(10.0, abs=1e-6)


def test_nan_solar_does_not_poison_the_counters() -> None:
    """Gap rows carry NaN solar; the QC must mask non-finite values."""
    frame = _frame()
    frame.loc[3, "solar_w"] = np.nan
    qc = qc_frame(frame)
    kt = qc["kt_over_one"]
    assert np.isfinite(kt["pct_gt_1_05"])
    assert np.isfinite(kt["p50_ratio"])
    assert qc["gaps"]["nan_solar_rows"] == 1


def test_markdown_renderer_contains_the_headline_numbers() -> None:
    qc = {"slices": {"train_2022": qc_frame(_frame())}}
    md = render_qc_markdown(qc)
    assert "train_2022" in md
    assert "kt" in md.lower()
    assert md.strip().startswith("#") or md.strip().startswith("|")
    assert "train_2022" in summarise(qc)
