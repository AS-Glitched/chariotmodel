"""Shared fixtures for the CHARIOT v3 test suite.

The tests run against a *synthetic* reference site that mirrors the DKASC
Alice Springs shape (solar-only telemetry, 15-minute cadence, a clear-sky
reference and a physically plausible clearness process).  The synthetic
frame is generated here so that the suite never depends on the (large,
git-ignored) DKASC export.
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

from chariot.model import solar_elevation_deg  # noqa: E402

SITE_ID = "dka_alice_springs"
TENANT_ID = "dkasc"
RATED_PV_W = 220_000.0


def make_reference_frame(days: int = 12, seed: int = 5) -> pd.DataFrame:
    """Synthetic solar-only telemetry for the DKASC reference site."""
    rng = np.random.default_rng(seed)
    n = days * 96
    idx = pd.date_range("2022-09-01", periods=n, freq="15min", tz="UTC")
    elevation = solar_elevation_deg(idx, -23.762, 133.875)

    # Clear-sky reference: a bell over the solar elevation.
    shape = np.clip(np.sin(np.radians(np.clip(elevation, 0.0, None))), 0.0, None)
    clear = RATED_PV_W * 0.92 * shape

    # Clearness: a persistent weather regime plus fast noise, bounded.
    regime = np.zeros(n)
    for i in range(1, n):
        regime[i] = 0.94 * regime[i - 1] + rng.normal(0.0, 0.06)
    clearness = np.clip(0.88 + regime + rng.normal(0.0, 0.03, n), 0.05, 1.15)
    clearness[clear <= 0.0] = 0.0
    solar = np.maximum(clear * clearness, 0.0)

    frame = pd.DataFrame(
        {
            "tenant_id": TENANT_ID,
            "site_id": SITE_ID,
            "timestamp_utc": idx,
            "solar_w": solar,
            "clear_sky_w": clear,
            "rated_pv_w": RATED_PV_W,
            "is_gap": False,
            "clear_sky_method": "synthetic_reference",
        }
    )
    return frame


@pytest.fixture(scope="session")
def reference_frame() -> pd.DataFrame:
    return make_reference_frame()


@pytest.fixture(scope="session")
def reference_parquet(tmp_path_factory, reference_frame: pd.DataFrame) -> Path:
    path = tmp_path_factory.mktemp("telemetry") / "reference.parquet"
    reference_frame.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, reference_parquet: Path) -> dict:
    """One end-to-end CHARIOT v3 run over the synthetic reference site."""
    from chariot.pipeline import PipelineConfig, run

    out_dir = tmp_path_factory.mktemp("run")
    store_path = tmp_path_factory.mktemp("store") / "chariot.db"
    config = PipelineConfig(
        source_csv=reference_parquet,
        out_dir=out_dir,
        store_path=store_path,
        n_scenarios=60,
        seed=11,
        horizon="operational",
        max_windows=2,
        spread_windows=2,
        spread_scenarios=40,
    )
    manifest = run(config)
    manifest["_out_dir"] = str(out_dir)
    manifest["_store"] = str(store_path)
    return manifest
