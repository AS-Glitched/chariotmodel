"""Tests for the W1 2022 -> 2023 protocol (splits, manifest, spine wiring).

The protocol is the backbone of the whole programme:

* ``train_2022``  - the first 60 % of the 2022 window (fitting only),
* ``validate_2022`` - the remaining 40 % of 2022 (selection only),
* ``test_2023``   - the whole 2023 year (report only, never fitted).

These tests pin three things: the extract honours a pre-assigned split when
asked (and re-splits by default, so the v3 behaviour is untouched), the
prepare script produces the three slices with content hashes and exact
boundaries, and the spine can be pointed at the validation split for the
selection metrics.
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

from chariot import training_extract  # noqa: E402
from chariot.store import Store  # noqa: E402
from chariot.training_extract import assign_splits, build_training_extract  # noqa: E402


@pytest.fixture()
def dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the materialised dataset into a temporary directory."""
    root = tmp_path / "datasets"
    monkeypatch.setattr(training_extract, "_dataset_root", lambda: root)
    return root


def _protocol_frame() -> pd.DataFrame:
    """A frame that already carries the protocol's split column."""
    n = 400
    idx = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "tenant_id": "dkasc",
            "site_id": "dka_alice_springs",
            "timestamp_utc": idx,
            "solar_w": np.linspace(0.0, 1000.0, n),
            "clear_sky_w": np.linspace(1.0, 1200.0, n),
            "rated_pv_w": 220_000.0,
            "is_gap": False,
        }
    )
    labels = assign_splits(frame, train_frac=0.6, validate_frac=0.4, test_frac=0.0)
    frame["split"] = labels
    return frame


def test_default_extract_resplits_chronologically(dataset_root: Path, tmp_path: Path) -> None:
    """Without the flag the extract re-splits 60/20/20 -- the v3 behaviour."""
    frame = _protocol_frame()
    store = Store(tmp_path / "store.db")
    info = build_training_extract(
        store,
        frame,
        tenant_id="dkasc",
        site_id="dka_alice_springs",
        capsule_ids=["cap_test"],
        calibration_version="calib_test",
        calibration_factors={"a": 1.0},
        admissibility_versions=["admissibility_v3.0"],
        slice_ids=["slice_test"],
        ctx={},
        created_by="test",
        honour_split=False,
    )
    assert info["split_policy"] == "time_ordered_60_20_20"
    assert set(info["split_row_counts"]) == {"train", "validate", "test"}


def test_honour_split_keeps_the_pre_assigned_split(dataset_root: Path, tmp_path: Path) -> None:
    frame = _protocol_frame()
    store = Store(tmp_path / "store.db")
    info = build_training_extract(
        store,
        frame,
        tenant_id="dkasc",
        site_id="dka_alice_springs",
        capsule_ids=["cap_test"],
        calibration_version="calib_test",
        calibration_factors={"a": 1.0},
        admissibility_versions=["admissibility_v3.0"],
        slice_ids=["slice_test"],
        ctx={},
        created_by="test",
        honour_split=True,
    )
    assert info["split_policy"].startswith("pre_assigned")
    # The protocol frame has no test rows: the 2023 window lives in a
    # different frame, so the extract must not invent one.
    assert info["split_row_counts"].get("test", 0) == 0
    assert info["split_row_counts"]["train"] > 0
    assert info["split_row_counts"]["validate"] > 0


def test_prepare_script_builds_the_three_protocol_slices(tmp_path: Path) -> None:
    """The year2022_2023 protocol writes the slices, hashes and boundaries."""
    import argparse

    sys.path.insert(0, str(ROOT / "scripts"))
    import prepare_dataset  # noqa: PLC0415

    def _csv(path: Path, year: int) -> None:
        n = 96 * 20
        idx = pd.date_range(f"{year}-01-01", periods=n, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {
                "tenant_id": "dkasc",
                "site_id": "dka_alice_springs",
                "timestamp_utc": idx,
                "solar_w": np.full(n, 1000.0),
                "clear_sky_w": np.full(n, 1200.0),
                "rated_pv_w": 220_000.0,
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
        rated_pv_w=220_000.0,
        train_frac=0.6,
        clearsky="export",
    )
    rc = prepare_dataset.prepare_year2022_2023(args)
    assert rc == 0

    import json

    manifest_path = tmp_path / "data" / "dka_alice_springs_2022_2023.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == "year2022_2023"
    assert set(manifest["content_hashes"]) == {
        "train_2022",
        "validate_2022",
        "test_2023",
        "combined",
    }
    boundaries = manifest["split_boundaries"]
    assert boundaries["train_2022"]["train"]["max"] < boundaries["validate_2022"]["validate"]["min"]
    assert boundaries["validate_2022"]["validate"]["max"] < boundaries["test_2023"]["test"]["min"]
    assert boundaries["test_2023"]["test"]["min"].startswith("2023")
    assert boundaries["train_2022"]["train"]["min"].startswith("2022")

    # The combined slice carries the split column for the spine.
    combined = pd.read_parquet(tmp_path / "data" / "dka_alice_springs_2022_2023.parquet")
    assert set(combined["split"]) == {"train", "validate", "test"}
    assert (combined.loc[combined["split"] == "test", "timestamp_utc"].dt.year == 2023).all()
    assert (combined.loc[combined["split"] != "test", "timestamp_utc"].dt.year == 2022).all()

    # The QC report exists and covers the three slices.
    qc = json.loads((tmp_path / "run" / "qc.json").read_text())
    assert set(qc["slices"]) == {"train_2022", "validate_2022", "test_2023"}
    assert (tmp_path / "run" / "summary.md").exists()


def test_pipeline_can_evaluate_the_validation_split(reference_parquet: Path, tmp_path: Path) -> None:
    """``evaluate_split='validate'`` scores the validation windows instead."""
    from chariot.pipeline import PipelineConfig, run

    config = PipelineConfig(
        source_csv=reference_parquet,
        out_dir=tmp_path / "run",
        store_path=tmp_path / "store" / "chariot.db",
        n_scenarios=40,
        seed=11,
        horizon="operational",
        max_windows=1,
        spread_windows=1,
        spread_scenarios=20,
        evaluate_split="validate",
    )
    manifest = run(config)
    assert manifest["evaluation"]["split"] == "validate"
    assert manifest["evaluation"]["solar_w"]["n_windows"] == 1
    # The model is still fitted on the train split only.
    assert manifest["model"]["diagnostics"]["train_time_start"] < "2022-12-31"
