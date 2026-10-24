"""Tests for the W6 solar-only advisory mode.

The DKASC Alice Springs site is **solar only**: there is no load meter and
no state-of-charge meter, so the load profile, the battery size and the
starting SOC are *declared* engineering assumptions.  The SOC and grid event
probabilities that the advisory publishes are therefore derived from
assumptions, not from observations, and they must not be published as if
they were skill.

The tests pin the contract:

* without measured load/SOC the hybrid event probabilities are nulled and
  the payload is marked ``hybrid_assumed``,
* with measured load/SOC the probabilities are published as numbers,
* the solar events are never affected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chariot.advisory import (  # noqa: E402
    HYBRID_EVENT_KEYS,
    build_events,
    hybrid_channels_measured,
)


class _Ctx:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.rated_pv_w = 220_000.0


class _Batch:
    """A minimal stand-in for a scenario batch."""

    horizon_steps = 16
    soc_percent = np.tile(np.linspace(30.0, 95.0, 16), (4, 1))
    grid_w = np.tile(np.linspace(-5_000.0, 30_000.0, 16), (4, 1))
    solar_w = np.tile(np.linspace(0.0, 200_000.0, 16), (4, 1))


def test_solar_only_nulls_the_hybrid_events_and_flags_them() -> None:
    ctx = _Ctx(
        {
            "load_observed": False,
            "soc_observed": False,
            "max_grid_import_w": 50_000.0,
            "max_grid_export_w": 50_000.0,
        }
    )
    assert hybrid_channels_measured(ctx) is False
    events = build_events(_Batch(), [0, 1, 2, 3], ctx=ctx, horizon_steps=16)
    for key in HYBRID_EVENT_KEYS:
        assert key in events
        assert events[key] is None
    assert events["hybrid_assumed"] is True
    assert "assumption" in events["hybrid_note"]


def test_measured_hybrid_channels_publish_the_probabilities() -> None:
    ctx = _Ctx(
        {
            "load_observed": True,
            "soc_observed": True,
            "max_grid_import_w": 50_000.0,
            "max_grid_export_w": 50_000.0,
        }
    )
    assert hybrid_channels_measured(ctx) is True
    events = build_events(_Batch(), [0, 1, 2, 3], ctx=ctx, horizon_steps=16)
    for key in HYBRID_EVENT_KEYS:
        assert events[key] is not None
        assert isinstance(events[key], float)
    assert "hybrid_assumed" not in events


def test_the_solar_events_are_never_affected() -> None:
    ctx = _Ctx({"load_observed": False, "soc_observed": False, "certified_ac_pv_w": 150_000.0})
    events = build_events(_Batch(), [0, 1, 2, 3], ctx=ctx, horizon_steps=16)
    assert events["p_pv_above_certified_ac"] is not None
    assert events["p_pv_above_certified_ac"] > 0.0


def test_the_pipeline_publishes_the_flag(pipeline_run: dict) -> None:
    """The operational run must mark the hybrid events as assumed."""
    import json

    advisories_path = Path(pipeline_run["_out_dir"]) / "advisories.json"
    payload = json.loads(advisories_path.read_text())
    rows = payload if isinstance(payload, list) else list(payload.values())
    assert rows, "the run produced no advisories"
    events = rows[0]["advisories"]["grid_engineer"]["events"]
    for key in HYBRID_EVENT_KEYS:
        if key in events:
            assert events[key] is None
    assert events.get("hybrid_assumed") is True
