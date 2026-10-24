"""Tests for the OFFLINE PyMC reference path.

The reference is an **optional** add-on: the operational spine must work
with or without PyMC installed, and nothing in the spine may import it.
The tests that genuinely need PyMC are marked ``pymc`` and are skipped when
the dependency is missing.

Run only the (slow) reference tests with::

    python -m pytest tests/test_pymc_reference.py -q -m pymc

The full suite (including the reference) with::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chariot.config import SiteContext  # noqa: E402
from chariot.model import ELEVATION_BINS, clear_sky_series  # noqa: E402
from chariot.pymc_clearness import (  # noqa: E402
    CLEARNESS_MAX,
    LIT_FRACTION,
    PymcClearnessFit,
    PymcReferenceConfig,
    PymcUnavailableError,
    build_pymc_model,
    fit_pymc_clearness,
    prepare_clearness_data,
    pymc_available,
    require_pymc,
)

from conftest import SITE_ID, TENANT_ID  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_ctx() -> SiteContext:
    """A site context that mirrors the DKASC reference site."""
    return SiteContext(
        tenant_id=TENANT_ID,
        site_id=SITE_ID,
        name="DKASC Alice Springs (reference)",
        rated_pv_w=220_000.0,
        usable_battery_capacity_wh=220_000.0,
        max_charge_w=110_000.0,
        max_discharge_w=110_000.0,
        timezone="Australia/Darwin",
        latitude=-23.762,
        longitude=133.875,
        fleet_id="dkasc",
        metadata={
            "load_profile_w": {str(h): 16000.0 + 2000.0 * (h % 7) for h in range(24)},
            "load_relative_sigma": 0.18,
            "initial_soc_percent": 50.0,
            "charge_loss_factor": 0.03,
            "discharge_loss_factor": 0.03,
        },
        source="site_master",
        missing_rating=False,
    )


def _fake_fit(n_draws: int = 4) -> PymcClearnessFit:
    """A reference fit built from synthetic posterior draws (no PyMC needed)."""
    rng = np.random.default_rng(3)
    n_bins = int(ELEVATION_BINS.size - 1)
    return PymcClearnessFit(
        backend="pymc_nuts",
        model_version="pymc_clearness_v3.0_nuts_test",
        model_name="chariot_pymc_clearness_reference",
        tenant_id=TENANT_ID,
        site_id=SITE_ID,
        fleet_id="dkasc",
        dataset_version="dataset_test",
        dataset_content_hash="0" * 64,
        posterior={
            "rho": rng.uniform(0.8, 0.97, size=n_draws),
            "sigma_process": rng.uniform(0.05, 0.2, size=n_draws),
            "obs_scale": rng.uniform(0.02, 0.08, size=n_draws),
        },
        log_profile=rng.normal(0.0, 0.05, size=(n_draws, n_bins)),
        profile_edges=[float(v) for v in ELEVATION_BINS],
        season_cos=np.zeros(n_draws),
        season_sin=np.zeros(n_draws),
        seasonal_selected="none",
        season_selection={"selected": "none"},
        origin_state={"v_mean": 0.0, "v_var": 0.05},
        diagnostics={"backend": "pymc_nuts", "chains": 2, "draws": n_draws},
        n_state_rows=96,
        n_obs_rows=40,
        config=PymcReferenceConfig(),
    )


# ---------------------------------------------------------------------------
# The spine must not depend on PyMC
# ---------------------------------------------------------------------------


def test_operational_spine_never_imports_pymc() -> None:
    """The ops spine must run with PyMC absent — it is not an import of it."""
    code = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "import chariot.pipeline, chariot.model, chariot.api, chariot.genesis;"
        "print('pymc' in sys.modules or 'pytensor' in sys.modules)"
        % str(ROOT)
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("False"), (
        "the operational spine must not pull PyMC/PyTensor in: " + out.stdout
    )


def test_reference_module_imports_and_reports_availability() -> None:
    assert isinstance(pymc_available(), bool)
    if pymc_available():
        require_pymc()
    else:  # pragma: no cover - depends on the environment
        with pytest.raises(PymcUnavailableError):
            require_pymc()


def test_config_validation_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        PymcReferenceConfig(backend="mcmc").validate()
    with pytest.raises(ValueError):
        PymcReferenceConfig(season="sometimes").validate()
    with pytest.raises(ValueError):
        PymcReferenceConfig(chains=0).validate()
    with pytest.raises(ValueError):
        PymcReferenceConfig(draws=0).validate()


# ---------------------------------------------------------------------------
# Data preparation mirrors the ops model
# ---------------------------------------------------------------------------


def test_prepared_data_mirrors_the_ops_lit_mask(
    reference_frame: pd.DataFrame, reference_ctx: SiteContext
) -> None:
    data = prepare_clearness_data(reference_frame, reference_ctx)
    assert data.n == len(reference_frame)
    assert data.n_obs > 0
    assert data.n_bins == int(ELEVATION_BINS.size - 1)

    clear = np.nan_to_num(clear_sky_series(reference_frame), nan=0.0)
    valid = np.isfinite(reference_frame["solar_w"].to_numpy()) & np.isfinite(clear)
    obs = valid & (clear > 1.0)
    peak = float(np.nanpercentile(clear[obs], 90))
    expected = obs & (clear > LIT_FRACTION * max(peak, 1.0))
    assert np.array_equal(data.lit, expected)
    # Every observed row carries a finite log clearness ratio.
    assert np.isfinite(data.log_obs[data.obs_index]).all()


# ---------------------------------------------------------------------------
# The adapter (no PyMC needed)
# ---------------------------------------------------------------------------


def test_reference_paths_are_night_zero_finite_and_capped(
    reference_frame: pd.DataFrame, reference_ctx: SiteContext
) -> None:
    fit = _fake_fit()
    future = reference_frame.head(96)
    paths = fit.simulate_paths(future, reference_ctx, n_paths=25, seed=11)

    solar = paths["solar_w"]
    load = paths["load_w"]
    assert solar.shape == (25, 96)
    assert load.shape == (25, 96)
    assert np.isfinite(solar).all()
    assert np.isfinite(load).all()
    assert (load >= 0.0).all()

    clear = np.nan_to_num(clear_sky_series(future), nan=0.0)
    night = clear <= 1.0
    assert night.any(), "the synthetic frame must contain night rows"
    assert np.all(solar[:, night] == 0.0), "night rows must be exactly zero"

    day = ~night
    ratio = solar[:, day] / np.maximum(clear[None, day], 1.0)
    assert ratio.max() <= CLEARNESS_MAX + 1e-9, "the clearness ceiling must hold"
    assert (solar >= 0.0).all()


def test_reference_paths_respect_the_site_rating(
    reference_frame: pd.DataFrame, reference_ctx: SiteContext
) -> None:
    fit = _fake_fit()
    paths = fit.simulate_paths(reference_frame.head(48), reference_ctx, n_paths=8, seed=5)
    assert paths["solar_w"].max() <= reference_ctx.rated_pv_w * 1.2 + 1e-6


def test_reference_channel_fit_adapter_shape() -> None:
    fit = _fake_fit()
    channel = fit.to_channel_fit()
    assert channel.gamma == 1.0
    assert channel.mu == 0.0
    assert len(channel.profile) == int(ELEVATION_BINS.size - 1)
    assert 0.0 <= channel.rho < 1.0
    assert channel.obs_floor > 0.0


def test_manifest_carries_every_required_field() -> None:
    fit = _fake_fit()
    manifest = fit.manifest()
    for key in (
        "inference_backend",
        "model_version",
        "chains",
        "draws",
        "divergences",
        "rhat_max",
        "ess_min",
        "wall_time_s",
        "train_extract_content_hash",
    ):
        assert key in manifest, f"missing manifest field: {key}"
    assert manifest["inference_backend"] == "pymc_nuts"


# ---------------------------------------------------------------------------
# The Genesis adapter seam
# ---------------------------------------------------------------------------


def test_generate_batch_accepts_reference_paths_without_changing_the_default(
    reference_frame: pd.DataFrame, reference_ctx: SiteContext
) -> None:
    from chariot.genesis import generate_batch
    from chariot.model import fit as fit_model

    future = reference_frame.head(96)
    model = fit_model(
        reference_frame.head(400),
        reference_ctx,
        tenant_id=TENANT_ID,
        site_id=SITE_ID,
        fleet_id="dkasc",
    )
    state = {
        "v_mean": 0.0,
        "v_var": 0.05,
        "last_solar": 0.0,
        "last_load": 16000.0,
        "soc_percent": 50.0,
    }
    ops_batch = generate_batch(
        model,
        SITE_ID,
        future,
        tenant_id=TENANT_ID,
        n_scenarios=6,
        seed=3,
        ctx=reference_ctx,
        state=state,
    )
    assert ops_batch.n_scenarios == 6
    assert ops_batch.solar_w.shape == (6, 96)

    fit = _fake_fit()
    paths = fit.simulate_paths(future, reference_ctx, n_paths=6, seed=3)
    ref_batch = generate_batch(
        model,
        SITE_ID,
        future,
        tenant_id=TENANT_ID,
        n_scenarios=6,
        seed=3,
        ctx=reference_ctx,
        state=state,
        paths=paths,
    )
    # The supplied ensemble is used verbatim ...
    assert np.allclose(ref_batch.solar_w, paths["solar_w"])
    # ... and the physics is still solved for every path.
    assert ref_batch.battery_w.shape == (6, 96)
    assert ref_batch.grid_w.shape == (6, 96)
    assert ref_batch.soc_percent.shape == (6, 96)
    assert np.isfinite(ref_batch.soc_percent).all()
    # The lineage still points at the model version.
    assert ref_batch.lineage["model_version"] == model.model_version
    assert ref_batch.content_hash


# ---------------------------------------------------------------------------
# The reference fit itself (needs PyMC)
# ---------------------------------------------------------------------------


@pytest.mark.pymc
def test_tiny_nuts_fit_produces_a_usable_posterior(
    reference_frame: pd.DataFrame, reference_ctx: SiteContext
) -> None:
    """A 1-chain, tiny NUTS fit on the synthetic reference site."""
    if not pymc_available():  # pragma: no cover - depends on the environment
        pytest.skip("PyMC is not installed")

    train = reference_frame.head(480).reset_index(drop=True)
    validate = reference_frame.iloc[480:576].reset_index(drop=True)
    config = PymcReferenceConfig(
        draws=25,
        tune=25,
        chains=1,
        cores=1,
        backend="nuts",
        season="off",
        random_seed=5,
    )
    fit = fit_pymc_clearness(
        train,
        reference_ctx,
        tenant_id=TENANT_ID,
        site_id=SITE_ID,
        fleet_id="dkasc",
        dataset_version="dataset_test",
        dataset_content_hash="a" * 64,
        validation=validate,
        config=config,
    )
    assert fit.backend == "pymc_nuts"
    assert np.isfinite(fit.posterior["rho"]).all()
    assert np.isfinite(fit.posterior["sigma_process"]).all()
    assert np.isfinite(fit.posterior["obs_scale"]).all()
    assert 0.0 < float(np.mean(fit.posterior["rho"])) < 1.0
    manifest = fit.manifest()
    assert manifest["chains"] == 1
    assert manifest["draws"] == 25
    assert manifest["inference_backend"] == "pymc_nuts"
    assert manifest["train_extract_content_hash"] == "a" * 64

    paths = fit.simulate_paths(reference_frame.head(96), reference_ctx, n_paths=10, seed=2)
    assert np.isfinite(paths["solar_w"]).all()
    clear = np.nan_to_num(clear_sky_series(reference_frame.head(96)), nan=0.0)
    assert np.all(paths["solar_w"][:, clear <= 1.0] == 0.0)


@pytest.mark.pymc
def test_model_builds_and_its_log_probability_is_finite(
    reference_frame: pd.DataFrame, reference_ctx: SiteContext
) -> None:
    """The AR(1) density is written as a Potential — no scan is compiled."""
    if not pymc_available():  # pragma: no cover - depends on the environment
        pytest.skip("PyMC is not installed")
    data = prepare_clearness_data(reference_frame, reference_ctx)
    model = build_pymc_model(data, season=True, config=PymcReferenceConfig())
    with model:
        logp = model.compile_logp()
        point = model.initial_point()
        value = float(logp(point))
    assert np.isfinite(value)
