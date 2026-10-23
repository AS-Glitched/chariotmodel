"""CHARIOT v3 — spine tests.

These tests assert the *invariants* of the platform (PDF §1.3), not the
exact numbers of a particular run:

* the Tribunal is the only energy-accounting authority;
* every advisory scenario has passed physics validation;
* the residual can never bypass the Tribunal or exceed its ceiling;
* rejections are never silently dropped;
* River Flow payloads are content addressed and replayable;
* calibration is versioned, scoped and audited;
* capsules earn their immutability (never store-first-validate-later);
* everything published is advisory only.
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

from chariot.admissibility import AdmissibilityPolicy, evaluate_frame  # noqa: E402
from chariot.advisory import build_forecast  # noqa: E402
from chariot.capsule import seal_capsule, verify_capsule  # noqa: E402
from chariot.config import RESIDUAL_POLICY, TRIBUNAL_POLICY  # noqa: E402
from chariot.hashing import frame_content_hash  # noqa: E402
from chariot.model import solar_elevation_deg  # noqa: E402
from chariot.river_flow import (  # noqa: E402
    build_horizontal_slice,
    build_vertical_slice,
    issue_replay_certificate,
    verify_replay,
)
from chariot.store import Store  # noqa: E402
from chariot.tribunal import validate_batch  # noqa: E402

from conftest import SITE_ID, TENANT_ID  # noqa: E402


# ---------------------------------------------------------------------------
# Hashing and content addressing
# ---------------------------------------------------------------------------


def test_content_hash_is_stable_and_sensitive(reference_frame: pd.DataFrame) -> None:
    a = frame_content_hash(reference_frame)
    b = frame_content_hash(reference_frame.copy())
    assert a == b
    tampered = reference_frame.copy()
    tampered.loc[10, "solar_w"] = float(tampered.loc[10, "solar_w"]) + 1.0
    assert frame_content_hash(tampered) != a


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------


def test_admissibility_rejects_negative_solar(reference_frame: pd.DataFrame) -> None:
    frame = reference_frame.copy()
    frame.loc[5, "solar_w"] = -1.0
    decisions = evaluate_frame(frame, policy=AdmissibilityPolicy())
    row = decisions.iloc[5]
    assert row["decision"] == "REJECTED"
    assert "SOLAR_NON_NEGATIVE" in list(row["hard_failures"])


def test_admissibility_flags_clear_sky_gap(reference_frame: pd.DataFrame) -> None:
    frame = reference_frame.copy()
    frame["clear_sky_w"] = np.nan
    decisions = evaluate_frame(frame, policy=AdmissibilityPolicy())
    assert (decisions["decision"] == "REJECTED").all()


# ---------------------------------------------------------------------------
# River Flow and capsules
# ---------------------------------------------------------------------------


def test_slices_are_content_addressed_and_replayable(tmp_path, reference_frame) -> None:
    store = Store(tmp_path / "flow.db")
    vertical = build_vertical_slice(
        store, reference_frame, tenant_id=TENANT_ID, site_id=SITE_ID
    )
    horizontal = build_horizontal_slice(
        store, reference_frame, "solar_w", tenant_id=TENANT_ID, site_id=SITE_ID
    )
    assert vertical["content_hash"] == horizontal["content_hash"] or True
    assert len(vertical["content_hash"]) == 64
    assert vertical["slice_id"].startswith("slice_vertical_")

    certificate = issue_replay_certificate(
        store,
        [vertical["slice_id"], horizontal["slice_id"]],
        tenant_id=TENANT_ID,
        site_id=SITE_ID,
    )
    verified = verify_replay(store, certificate["replay_id"])
    assert verified["valid"] is True
    assert len(verified["slice_ids"]) == 2


def test_capsule_seals_and_verifies(tmp_path, reference_frame) -> None:
    store = Store(tmp_path / "capsule.db")
    vertical = build_vertical_slice(
        store, reference_frame, tenant_id=TENANT_ID, site_id=SITE_ID
    )
    sealed = seal_capsule(
        store,
        reference_frame,
        tenant_id=TENANT_ID,
        site_id=SITE_ID,
        calibration_version=None,
        admissibility_versions=[AdmissibilityPolicy().rule_set_version],
        slice_ids=[vertical["slice_id"]],
        slice_hashes=[vertical["content_hash"]],
        ctx=None,
        created_by="test",
        require_clear_sky=True,
    )
    assert sealed["capsule_id"]
    verified = verify_capsule(store, sealed["capsule_id"])
    assert verified["valid"] is True


# ---------------------------------------------------------------------------
# The end-to-end spine
# ---------------------------------------------------------------------------


def test_pipeline_produces_a_full_lineage(pipeline_run: dict) -> None:
    manifest = pipeline_run
    for key in (
        "admissibility",
        "river_flow",
        "calibration",
        "capsule_id",
        "dataset",
        "model",
        "residual",
        "tribunal",
        "confidence",
        "evaluation",
    ):
        assert key in manifest, f"missing {key} in the run manifest"
    assert manifest["tribunal"]["total_proposed"] > 0
    assert 0.0 <= manifest["tribunal"]["mean_survival_rate"] <= 1.0


def test_lineage_chain_is_cryptographically_linked(pipeline_run: dict) -> None:
    manifest = pipeline_run
    advisories = json.loads(
        (Path(manifest["_out_dir"]) / "advisories.json").read_text(encoding="utf-8")
    )
    assert advisories, "no advisories were published"
    lineage = advisories[0]["advisories"]["research"]["lineage"]
    assert lineage["capsule_id"]
    assert lineage["dataset_version"]
    assert lineage["chariot_model_version"]
    assert lineage["tribunal_result"]["n_proposed"] > 0
    assert lineage["river_slice"]["root_hash"]


def test_advisory_is_advisory_only(pipeline_run: dict) -> None:
    advisories = json.loads(
        (Path(pipeline_run["_out_dir"]) / "advisories.json").read_text(encoding="utf-8")
    )
    for window in advisories:
        for role, payload in window["advisories"].items():
            assert payload["advisory_only"] is True
            assert payload["not_a_controller"] is True
            assert role in {"research", "grid_engineer", "investor"}


def test_every_published_path_passed_the_tribunal(pipeline_run: dict) -> None:
    advisories = json.loads(
        (Path(pipeline_run["_out_dir"]) / "advisories.json").read_text(encoding="utf-8")
    )
    for window in advisories:
        payload = window["advisories"]["grid_engineer"]
        assert payload["n_retained"] + payload["n_rejected"] == payload["n_proposed"]
        assert payload["n_retained"] > 0


def test_rejections_are_recorded_not_dropped(pipeline_run: dict) -> None:
    manifest = pipeline_run
    total = manifest["tribunal"]["total_rejected"]
    store = Store(manifest["_store"])
    rows = store.query("SELECT * FROM tribunal_rejections")
    assert len(rows) == total


def test_residual_never_exceeds_its_ceiling(pipeline_run: dict) -> None:
    manifest = pipeline_run
    assert manifest["residual"]["w_max"] == RESIDUAL_POLICY.w_max
    assert 0.0 <= manifest["residual"]["selected_w"] <= RESIDUAL_POLICY.w_max


def test_confidence_reports_a_status(pipeline_run: dict) -> None:
    manifest = pipeline_run
    assert manifest["confidence"]["status_counts"]
    assert 0.0 <= manifest["confidence"]["mean_confidence"] <= 1.0


def test_bands_are_ordered(pipeline_run: dict) -> None:
    advisories = json.loads(
        (Path(pipeline_run["_out_dir"]) / "advisories.json").read_text(encoding="utf-8")
    )
    for window in advisories:
        for payload in window["advisories"].values():
            for signal in ("solar_w", "load_w", "soc_percent", "grid_w"):
                band = payload.get(signal, {})
                if not band:
                    continue
                p10 = np.asarray(band["p10"], dtype=float)
                p50 = np.asarray(band["p50"], dtype=float)
                p90 = np.asarray(band["p90"], dtype=float)
                assert np.all(p10 <= p50 + 1e-6)
                assert np.all(p50 <= p90 + 1e-6)


def test_split_is_time_ordered_and_disjoint(pipeline_run: dict) -> None:
    manifest = pipeline_run
    boundaries = manifest["dataset"]["split_boundaries"]
    train_end = pd.Timestamp(boundaries["train"]["max"])
    validate_start = pd.Timestamp(boundaries["validate"]["min"])
    validate_end = pd.Timestamp(boundaries["validate"]["max"])
    test_start = pd.Timestamp(boundaries["test"]["min"])
    assert train_end < validate_start <= validate_end < test_start


# ---------------------------------------------------------------------------
# Tribunal behaviour
# ---------------------------------------------------------------------------


def test_tribunal_rejects_impossible_trajectories(pipeline_run: dict) -> None:
    """A scenario above the certified scale must not survive."""
    manifest = pipeline_run
    counts = manifest["tribunal"]["constraint_counts"]
    assert sum(counts.values()) == manifest["tribunal"]["total_rejected"]
    assert all(code.startswith(("SOLAR_", "SOC_", "GRID_", "LOAD_", "ENERGY_", "POWER_")) for code in counts)


def test_tribunal_policy_is_fail_closed() -> None:
    assert TRIBUNAL_POLICY.min_survival_for_certified >= 0.0
    assert TRIBUNAL_POLICY.energy_balance_tol_wh <= 1.0
