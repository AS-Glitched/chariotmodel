"""PyMC reference inference for the solar clearness channel (OFFLINE ONLY).

This module is **not** part of the operational path.  The ops model
(:mod:`chariot.model`) remains a pure-NumPy method-of-moments + Kalman
estimator and stays the default for ``/model/chariot/train`` and
``scripts/run_chariot3.py``.  Nothing in the spine imports this module.

What it provides
----------------
A *reference* implementation of the same clearness world model, fitted with
Bayesian inference (NUTS, or a clearly-labelled VI fallback), so that the ops
estimator can be checked against a posterior rather than against itself.

The model mirrors the v3 observation model in algebra:

.. code-block:: text

    log(solar_t / clear_t)  ~  StudentT(nu,
                              log_profile[bin(elev_t)] + season(doy_t) + v_t,
                              obs_scale)
    v_t                    =  sigma_process * w_t
    w_t                    =  rho * w_{t-1} + z_t ,   z_t ~ N(0, 1)

* **multiplicative clearness in log space** — the clearness ratio is
  right-skewed and heavy-tailed, so the latent anomaly is additive in log
  space (this is exactly what the ops model does, and it is why the ops model
  uses a log-space EKF);
* **AR(1) latent weather state** — the state runs over the *whole* 15-minute
  grid (day and night), exactly like the ops model's Kalman filter, so the
  time structure of the persistence is identical;
* **elevation structure** — a hierarchical profile over the 15 solar-elevation
  bins (the fleet-level hyper-mean provides the partial pooling);
* **optional seasonal harmonic** — selected on the *validation* split, and
  dropped when it does not help (this mirrors the ops model, which also
  rejected the harmonic on the Alice Springs 2022 data);
* **robust observation noise** — a Student-t likelihood with a free degree of
  freedom, the Bayesian analogue of the ops model's Huberised EKF.

Why the AR(1) state is written as a Potential and not as ``pm.AR``
------------------------------------------------------------------
The latent state is an AR(1) process, so the obvious PyTensor implementation
is a ``scan``.  A ``scan`` inside a PyMC model is fragile (it needs a
symbolic, statically-known step count) and it also has to be *compiled*,
which fails outright on hosts without Python development headers.

``pm.AR`` looks like the answer: PyMC implements ``AR``'s log-probability
*analytically* (see ``pymc.distributions.timeseries.ar_logp``) —

.. code-block:: python

    innov_logp = sum(logp(Normal(0, sigma), value[..., 1:] - rho * value[..., :-1]))
    init_logp   = logp(init_dist, value[..., :1])

— which is a handful of vectorised numpy operations, not a recurrence.  But
``pm.AR`` is a *random variable*, and the gradient of a random variable is
taken through its ``OpFromGraph`` — which contains the forward AR ``scan``.
The reverse-mode gradient of that scan is a second scan, and on a 20 000-step
chain it costs roughly **fifty times** the log-probability (measured on the
reference host: 278 ms per gradient against 4.6 ms per log-probability).
NUTS needs thousands of gradients, so that is not affordable.

This module therefore writes the AR(1) density *directly*, as a
:func:`pymc.Potential` over a flat state variable:

.. code-block:: python

    innov = concat([a[0], a[1:] - rho * a[:-1]])
    potential = sum(Normal(0, 1).logpdf(innov))

That is the **same density** as ``pm.AR`` with unit innovations — PyMC's own
``ar_logp`` reduces to exactly this expression — but the state is now an
ordinary tensor, so its gradient is elementwise and no scan is compiled at
all.  The measured cost drops to the same order as the log-probability.

Parameterisation
----------------
The state is written as ``v = sigma_process * a`` where ``a`` is the
unit-innovation AR process, so ``sigma_process`` has exactly the meaning it
has in the ops model (the innovation scale of the state) and the two
estimators are directly comparable.

Hardware note
-------------
PyTensor compiles its graph with a C backend when it can.  When the host has
no ``Python.h`` (or no C compiler at all) PyTensor cannot emit **any** C
code — not even its own lazy-linker extension — and every compilation
fails.  This module therefore adapts the PyTensor flags *before* PyTensor is
imported:

* ``cxx=`` — disable C code generation and fall back to the Python VM;
* ``optimizer_excluding=fusion`` — without a C backend, PyTensor's fused
  ``Composite`` op degrades to a scalar Python loop that is roughly ten
  times slower than the unfused numpy ufuncs;
* ``mode=NUMBA`` — when ``numba`` is importable the whole graph is compiled
  to machine code, which needs neither C headers nor fusion.

The operational NumPy model needs none of this.
"""

from __future__ import annotations

import math
import os
import shutil
import sysconfig
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# PyTensor environment adaptation.  PyTensor reads ``PYTENSOR_FLAGS`` when it
# is first imported, so this has to happen before ``pymc`` is imported.
# --------------------------------------------------------------------------


def _c_toolchain_available() -> bool:
    """True when PyTensor can plausibly emit C code on this host."""
    include = sysconfig.get_paths().get("include") or ""
    if not include or not os.path.exists(os.path.join(include, "Python.h")):
        return False
    compiler = os.environ.get("CXX", "g++")
    return bool(shutil.which(compiler) or shutil.which("cc") or shutil.which("clang++"))


def _numba_available() -> bool:
    try:  # pragma: no cover - depends on the environment
        import numba  # noqa: F401

        return True
    except Exception:
        return False


def _pytensor_flags() -> str:
    """Build the ``PYTENSOR_FLAGS`` value for this host.

    Existing flags are respected — the operator's own settings always win.
    """
    existing = [p for p in os.environ.get("PYTENSOR_FLAGS", "").split(",") if p]
    keys = {p.split("=", 1)[0] for p in existing}
    flags = list(existing)
    has_c = _c_toolchain_available()
    if not has_c and "cxx" not in keys:
        # Without a usable C toolchain PyTensor cannot emit *any* C code --
        # not even its own lazy-linker extension -- and every compilation
        # fails.  Disable C code generation outright.
        flags.append("cxx=")
    if _numba_available() and "mode" not in keys:
        flags.append("mode=NUMBA")
    if "optimizer_excluding" not in keys and (not has_c or _numba_available()):
        # Without a C backend the fused ``Composite`` op degrades to a scalar
        # Python loop, which is roughly ten times slower than the unfused
        # numpy ufuncs.  The Numba backend cannot consume fused graphs
        # either.
        flags.append("optimizer_excluding=fusion")
    return ",".join(flags)


PYTENSOR_FLAGS = _pytensor_flags()
if PYTENSOR_FLAGS:
    os.environ["PYTENSOR_FLAGS"] = PYTENSOR_FLAGS

from chariot.hashing import hash_obj, short_hash  # noqa: E402
from chariot.model import (  # noqa: E402
    CLEARNESS_MAX,
    ELEVATION_BINS,
    ChannelFit,
    ChariotModel,
    FleetPosterior,
    SiteModel,
    clear_sky_series,
    elevation_bin,
    solar_elevation_deg,
    solar_series,
)

# --------------------------------------------------------------------------
# Optional dependency
# --------------------------------------------------------------------------

PYMC_AVAILABLE = False
_PYMC_IMPORT_ERROR = ""
try:  # pragma: no cover - exercised by the import guard
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    PYMC_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the environment
    pm = None  # type: ignore[assignment]
    pytensor = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]
    _PYMC_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


INSTALL_HINT = (
    "The PyMC reference path needs PyMC and PyTensor:\n"
    "    pip install 'pymc>=5' 'pytensor>=2.20'\n"
    "For a fast reference fit on a host without Python headers also install:\n"
    "    pip install 'numba>=0.59'\n"
    "The operational NumPy model does NOT need any of these."
)


class PymcUnavailableError(RuntimeError):
    """Raised when the reference path is requested but PyMC is missing."""


def pymc_available() -> bool:
    """True when the optional reference backend can be imported."""
    return PYMC_AVAILABLE


def require_pymc() -> None:
    """Raise a clear, actionable error when PyMC is unavailable."""
    if not PYMC_AVAILABLE:
        raise PymcUnavailableError(
            f"PyMC is not importable ({_PYMC_IMPORT_ERROR}).\n{INSTALL_HINT}"
        )


def _configure_pytensor() -> dict[str, Any]:
    """Apply the PyTensor settings this host needs (idempotent).

    The environment flags above are the primary mechanism, because PyTensor
    reads ``PYTENSOR_FLAGS`` once, at import time.  When something else has
    already imported PyTensor the flags are ignored, so the settings that
    *can* still be changed at runtime are applied here as a fallback:

    * ``cxx = ""`` — always possible, and mandatory on a host without
      Python headers (otherwise every compilation fails);
    * ``mode = "NUMBA"`` — only useful when the elementwise fusion rewrite
      is excluded as well, because the Numba backend cannot consume fused
      graphs.  ``FAST_COMPILE`` is the fallback: it never fuses, so the
      unfused numpy ufuncs are used.
    """
    info: dict[str, Any] = {
        "pytensor_flags": PYTENSOR_FLAGS,
        "c_toolchain_available": _c_toolchain_available(),
        "numba_available": _numba_available(),
    }
    if pytensor is None:  # pragma: no cover - PyMC is missing
        info["backend"] = "unavailable"
        return info
    if not info["c_toolchain_available"]:
        try:
            pytensor.config.cxx = ""
        except Exception:  # pragma: no cover - defensive
            pass
    excluding = str(getattr(pytensor.config, "optimizer_excluding", "") or "")
    fusion_off = "fusion" in excluding
    if info["numba_available"] and fusion_off:
        try:
            pytensor.config.mode = "NUMBA"
        except Exception:  # pragma: no cover - defensive
            pass
    elif info["numba_available"] and not fusion_off:
        try:
            pytensor.config.mode = "FAST_COMPILE"
        except Exception:  # pragma: no cover - defensive
            pass
    info["backend"] = str(getattr(pytensor.config, "mode", "default"))
    info["fusion_excluded"] = fusion_off
    info["cxx"] = str(getattr(pytensor.config, "cxx", ""))
    return info


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Additive floor (W) used when a clearness ratio has to be formed.
OBS_FLOOR_W = 50.0

#: Fraction of the 90th-percentile clear-sky level above which a row counts
#: as "lit" for the observation set.  Mirrors the ops model's ``lit`` mask.
LIT_FRACTION = 0.2

BACKENDS = ("nuts", "vi", "auto")
SEASON_MODES = ("auto", "on", "off")

#: Version tag of the reference model family.
REFERENCE_MODEL_VERSION = "pymc_clearness_reference_v1.0"


@dataclass(frozen=True)
class PymcReferenceConfig:
    """Inference settings for the offline reference fit."""

    #: Posterior draws per chain.
    draws: int = 500
    #: NUTS tuning steps per chain.
    tune: int = 500
    #: Number of chains (>= 2 is required for a meaningful R-hat).
    chains: int = 2
    #: Parallel chains.  The reference host has 2 cores and 3 GB of RAM, so
    #: the default is 1 (chains run sequentially) to avoid oversubscription.
    cores: int = 1
    #: NUTS target acceptance rate.
    target_accept: float = 0.9
    #: NUTS maximum tree depth.
    max_treedepth: int = 10
    #: ``auto`` (fit with the harmonic, keep it only if validation improves),
    #: ``on`` or ``off``.
    season: str = "auto"
    #: ``nuts``, ``vi`` or ``auto`` (NUTS, falling back to VI on failure).
    backend: str = "auto"
    #: ADVI iterations for the VI fallback.
    vi_iters: int = 20000
    #: ADVI samples drawn from the fitted approximation.
    vi_samples: int = 1000
    #: Sampler seed.
    random_seed: int = 7
    #: Show the PyMC progress bar.
    progressbar: bool = False
    #: Retain the per-observation latent state in the trace.  It is large
    #: (one value per 15-minute step); the reference only needs the
    #: hyperparameters, so it is dropped by default.
    keep_state: bool = False

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.season not in SEASON_MODES:
            raise ValueError(f"season must be one of {SEASON_MODES}, got {self.season!r}")
        if self.chains < 1:
            raise ValueError("chains must be >= 1")
        if self.draws < 1 or self.tune < 0:
            raise ValueError("draws must be >= 1 and tune >= 0")
        if self.cores < 1:
            raise ValueError("cores must be >= 1")


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------


@dataclass
class ClearnessData:
    """The full-grid clearness view of a telemetry frame.

    The latent AR(1) state runs over the *whole* grid (night included), the
    way the ops model's Kalman filter does, so the data object keeps every
    row and carries an observation mask instead of dropping the night.
    """

    timestamps: pd.Series
    solar: np.ndarray
    clear: np.ndarray
    elevation: np.ndarray
    doy: np.ndarray
    bins: np.ndarray
    #: Log of the observed clearness ratio; NaN on rows that are not lit.
    log_obs: np.ndarray
    #: Boolean mask of the observed (lit) rows.
    lit: np.ndarray
    #: Row indices of the lit rows.
    obs_index: np.ndarray
    n_bins: int

    @property
    def n(self) -> int:
        """Number of grid steps (the length of the latent state chain)."""
        return int(self.log_obs.size)

    @property
    def n_obs(self) -> int:
        """Number of observed (lit) rows."""
        return int(self.obs_index.size)


def prepare_clearness_data(frame: pd.DataFrame, ctx: Any) -> ClearnessData:
    """Build the clearness view of a frame.

    The selection mirrors the ops model exactly: a row is *observed* when it
    is a valid, daylight row whose clear-sky level is above ``LIT_FRACTION``
    of the 90th-percentile clear-sky level.  Every row — day and night —
    stays in the grid, because the AR(1) state is defined over time.
    """
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    solar = solar_series(ordered)
    clear = clear_sky_series(ordered)
    valid = np.isfinite(solar) & np.isfinite(clear)
    daylight = clear > 1.0
    if not valid.any():
        raise ValueError("no valid rows in the frame")

    obs = valid & daylight & (clear > 1.0)
    if int(obs.sum()) < 24:
        raise ValueError(
            f"only {int(obs.sum())} observed rows — the reference model needs at least 24"
        )
    lit_peak = float(np.nanpercentile(clear[obs], 90))
    lit = obs & (clear > LIT_FRACTION * max(lit_peak, 1.0))

    elevation = solar_elevation_deg(
        ordered["timestamp_utc"],
        float(getattr(ctx, "latitude", None) or 0.0),
        float(getattr(ctx, "longitude", None) or 0.0),
    )
    doy = (
        pd.to_datetime(ordered["timestamp_utc"], utc=True).dt.day_of_year.to_numpy().astype(float)
    )
    bins = elevation_bin(elevation)
    n_bins = int(ELEVATION_BINS.size - 1)

    ratio = np.where(clear > 1.0, solar / np.maximum(clear, 1.0), np.nan)
    log_obs = np.where(lit, np.log(np.clip(ratio, 1e-3, None)), np.nan)

    return ClearnessData(
        timestamps=ordered["timestamp_utc"],
        solar=solar,
        clear=clear,
        elevation=elevation,
        doy=doy,
        bins=bins,
        log_obs=log_obs,
        lit=lit,
        obs_index=np.flatnonzero(lit),
        n_bins=n_bins,
    )


# --------------------------------------------------------------------------
# A cheap, self-contained starting point
# --------------------------------------------------------------------------


def _initial_state_estimate(data: ClearnessData) -> dict[str, Any]:
    """A method-of-moments starting point for the NUTS chain.

    This is deliberately *not* the ops model: the reference must be able to
    start from something that was derived from the data alone, so that the
    comparison is not circular.  It only has to be roughly right — NUTS
    refines everything from here.
    """
    idx = data.obs_index
    log_obs = data.log_obs[idx]
    bins = data.bins[idx]
    n_bins = data.n_bins

    profile = np.zeros(n_bins, dtype=float)
    global_median = float(np.median(log_obs))
    for b in range(n_bins):
        sel = bins == b
        profile[b] = float(np.median(log_obs[sel])) if int(sel.sum()) >= 3 else np.nan
    known = np.flatnonzero(np.isfinite(profile))
    if known.size:
        for b in range(n_bins):
            if not np.isfinite(profile[b]):
                profile[b] = float(np.interp(b, known, profile[known]))
    else:  # pragma: no cover - defensive
        profile[:] = global_median

    anomaly = log_obs - profile[bins]
    sd = float(np.std(anomaly)) if anomaly.size else 0.1
    # AR(1) innovation scale from the lag-1 differences on *consecutive*
    # observed rows (a night breaks the chain, exactly as in the ops model).
    diffs = np.diff(anomaly)
    gap = np.diff(idx)
    consecutive = gap == 1
    if int(consecutive.sum()) >= 8:
        var_d = float(np.var(diffs[consecutive]))
        rho = 1.0 - var_d / max(2.0 * sd**2, 1e-9)
    else:  # pragma: no cover - defensive
        rho = 0.5
    rho = float(np.clip(rho, 0.0, 0.99))
    sigma_process = float(np.clip(sd * math.sqrt(max(1.0 - rho**2, 1e-3)), 1e-3, 2.0))
    # Robust observation scale: the median absolute deviation of the anomaly
    # relative to its own scale.
    mad = float(np.median(np.abs(anomaly - np.median(anomaly)))) * 1.4826
    obs_scale = float(np.clip(mad / math.sqrt(max(1.0 - rho**2, 1e-3)), 5e-3, 1.0))

    # Latent-state start: the anomaly on the unit-innovation scale.
    state = np.zeros(data.n, dtype=float)
    state[idx] = anomaly / max(sigma_process, 1e-6)
    return {
        "rho": rho,
        "sigma_process": sigma_process,
        "obs_scale": obs_scale,
        "log_profile": profile,
        "fleet_log_prof_mu": float(profile.mean()),
        "fleet_log_prof_sigma": max(float(np.std(profile)), 1e-3),
        "state": state,
    }


# --------------------------------------------------------------------------
# The PyMC model
# --------------------------------------------------------------------------


def build_pymc_model(
    data: ClearnessData,
    *,
    season: bool,
    config: PymcReferenceConfig,
) -> Any:
    """Build the PyMC model for the clearness channel.

    Priors (all weakly informative, documented in the module docstring):

    ==========================  ===============================================
    ``fleet_log_prof_mu``        N(0, 0.5) — the mean log-clearness level
    ``fleet_log_prof_sigma``     HalfNormal(0.25) — between-bin spread
    ``log_profile``              N(fleet_mu, fleet_sigma) per elevation bin
    ``rho``                      U(0, 0.999) — AR(1) persistence per 15 min
    ``sigma_process``            HalfNormal(0.3) — AR(1) innovation scale
    ``obs_scale``                HalfNormal(0.15) — observation scale (log units)
    ``nu_excess``                Exponential(0.2), so nu = 2 + excess >= 2
    ``season_cos`` / ``season_sin``  N(0, 0.1) — bounded annual harmonic
    ==========================  ===============================================
    """
    require_pymc()
    n = data.n
    n_bins = data.n_bins
    obs_index = np.asarray(data.obs_index, dtype=np.int64)
    theta = 2.0 * math.pi * data.doy / 365.25
    start = _initial_state_estimate(data)

    with pm.Model() as model:
        # -- hierarchical elevation structure (partial pooling over bins) ---
        fleet_mu = pm.Normal("fleet_log_prof_mu", 0.0, 0.5, shape=n_bins)
        fleet_sigma = pm.HalfNormal("fleet_log_prof_sigma", 0.25)
        log_profile = pm.Normal("log_profile", fleet_mu, fleet_sigma, shape=n_bins)

        # -- AR(1) latent weather state --------------------------------------
        # The state is a flat tensor; the AR(1) density is written out as a
        # Potential (see the module docstring).  This is the same density as
        # ``pm.AR`` with unit innovations, but its gradient is elementwise and
        # no scan is ever compiled.
        rho = pm.Uniform("rho", 0.0, 0.999)
        sigma_process = pm.HalfNormal("sigma_process", 0.3)
        state = pm.Flat("weather_state", shape=n)
        innov = pt.concatenate([state[:1], state[1:] - rho * state[:-1]])
        pm.Potential(
            "ar1_state",
            -0.5 * pt.sum(pt.square(innov)) - 0.5 * n * math.log(2.0 * math.pi),
        )
        v = sigma_process * state

        # -- optional bounded annual harmonic ---------------------------------
        if season:
            season_cos = pm.Normal("season_cos", 0.0, 0.1)
            season_sin = pm.Normal("season_sin", 0.0, 0.1)
            season_term = season_cos * np.cos(theta) + season_sin * np.sin(theta)
        else:
            season_term = pt.zeros(n)

        # -- observation ------------------------------------------------------
        mu = log_profile[data.bins] + season_term + v
        mu_obs = pt.take(mu, obs_index)
        obs_scale = pm.HalfNormal("obs_scale", 0.15)
        nu = pm.Exponential("nu_excess", 0.2) + 2.0
        pm.StudentT(
            "log_clearness",
            nu=nu,
            mu=mu_obs,
            sigma=obs_scale,
            observed=data.log_obs[obs_index],
        )

        # -- a self-contained starting point ----------------------------------
        try:
            model.set_initval(model["rho"], float(start["rho"]))
            model.set_initval(model["sigma_process"], float(start["sigma_process"]))
            model.set_initval(model["obs_scale"], float(start["obs_scale"]))
            model.set_initval(model["log_profile"], start["log_profile"])
            model.set_initval(model["fleet_log_prof_mu"], start["fleet_log_prof_mu"])
            model.set_initval(model["fleet_log_prof_sigma"], start["fleet_log_prof_sigma"])
            model.set_initval(model["weather_state"], start["state"])
            if season:
                model.set_initval(model["season_cos"], 0.0)
                model.set_initval(model["season_sin"], 0.0)
        except Exception:  # pragma: no cover - defensive, PyMC may re-key
            pass
    return model


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


@dataclass
class PymcClearnessFit:
    """The result of an offline reference fit."""

    backend: str
    model_version: str
    model_name: str
    tenant_id: str
    site_id: str
    fleet_id: str
    dataset_version: str
    dataset_content_hash: str
    #: Posterior draws of the hyperparameters.
    posterior: dict[str, np.ndarray]
    #: Posterior draws of the log elevation profile, shape (n_draws, n_bins).
    log_profile: np.ndarray
    profile_edges: list[float]
    season_cos: np.ndarray
    season_sin: np.ndarray
    #: Which seasonal treatment the validation split selected.
    seasonal_selected: str
    season_selection: dict[str, Any]
    #: Filtered state at the end of the training window (posterior mean).
    origin_state: dict[str, float]
    diagnostics: dict[str, Any]
    n_state_rows: int
    n_obs_rows: int
    config: PymcReferenceConfig
    extra: dict[str, Any] = field(default_factory=dict)

    # -- summary ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "tenant_id": self.tenant_id,
            "site_id": self.site_id,
            "fleet_id": self.fleet_id,
            "dataset_version": self.dataset_version,
            "dataset_content_hash": self.dataset_content_hash,
            "seasonal_selected": self.seasonal_selected,
            "season_selection": self.season_selection,
            "origin_state": self.origin_state,
            "diagnostics": self.diagnostics,
            "n_state_rows": self.n_state_rows,
            "n_obs_rows": self.n_obs_rows,
            "config": self.config.to_dict(),
            "posterior_mean": {
                key: float(np.mean(value)) for key, value in self.posterior.items()
            },
            "posterior_sd": {
                key: float(np.std(value)) for key, value in self.posterior.items()
            },
        }

    def manifest(self) -> dict[str, Any]:
        """The run-manifest fields required of the reference fit."""
        conv = self.diagnostics.get("convergence", {}) or {}
        return {
            "inference_backend": self.backend,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "chains": int(self.diagnostics.get("chains", 0)),
            "draws": int(self.diagnostics.get("draws", 0)),
            "tune": int(self.diagnostics.get("tune", 0)),
            "divergences": int(conv.get("divergences", 0)),
            "rhat_max": conv.get("rhat_max"),
            "ess_min": conv.get("ess_min"),
            "wall_time_s": self.diagnostics.get("wall_time_s"),
            "train_extract_content_hash": self.dataset_content_hash,
            "dataset_version": self.dataset_version,
            "pytensor_backend": self.diagnostics.get("pytensor_backend"),
            "pytensor_flags": self.diagnostics.get("pytensor_flags"),
            "seasonal_selected": self.seasonal_selected,
        }

    def to_channel_fit(self) -> ChannelFit:
        """Project the posterior onto the ops model's channel structure.

        This is the **adapter**: the PyMC posterior becomes a v3
        :class:`~chariot.model.ChannelFit`, so the ops filter, the origin-state
        machinery and the whole downstream spine can be reused unchanged.

        v3 stores the clearness level in ``gamma`` and the *shape* in
        ``profile``; the PyMC posterior carries both in ``log_profile``, so
        ``gamma`` is 1.0 and ``profile = exp(log_profile)``.
        """
        profile = np.exp(np.mean(self.log_profile, axis=0))
        return ChannelFit(
            name="clearness",
            mu=0.0,
            rho=float(np.mean(self.posterior["rho"])),
            sigma=float(np.mean(self.posterior["sigma_process"])),
            gamma=1.0,
            obs_scale=float(np.mean(self.posterior["obs_scale"])),
            obs_floor=OBS_FLOOR_W,
            n_obs=int(self.n_obs_rows),
            profile=[float(v) for v in profile],
            profile_edges=[float(v) for v in self.profile_edges],
            season_cos=float(np.mean(self.season_cos)),
            season_sin=float(np.mean(self.season_sin)),
            se={
                "rho": float(np.std(self.posterior["rho"])),
                "sigma": float(np.std(self.posterior["sigma_process"])),
                "obs_scale": float(np.std(self.posterior["obs_scale"])),
            },
            diagnostics=dict(self.diagnostics),
        )

    # -- forward simulation -------------------------------------------------
    def simulate_paths(
        self,
        future: pd.DataFrame,
        ctx: Any,
        *,
        n_paths: int = 500,
        seed: int = 7,
        spread_factor: float = 1.0,
        origin_state: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw scenario paths from the posterior.

        The paths are produced from the **actual posterior draws** (not a
        Gaussian summary), propagate the AR(1) with the same night-reset rule
        as the ops model, and respect the clearness ceiling.  Night rows are
        exactly zero.
        """
        frame = future.sort_values("timestamp_utc").reset_index(drop=True)
        t_steps = len(frame)
        clear = np.nan_to_num(clear_sky_series(frame), nan=0.0)
        elevation = solar_elevation_deg(
            frame["timestamp_utc"],
            float(getattr(ctx, "latitude", None) or 0.0),
            float(getattr(ctx, "longitude", None) or 0.0),
        )
        bins = elevation_bin(elevation)
        doy = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.day_of_year.to_numpy().astype(float)
        theta = 2.0 * math.pi * doy / 365.25
        daylight = clear > 1.0

        rng = np.random.default_rng(seed)
        n_draws = int(self.log_profile.shape[0])
        picks = rng.integers(0, n_draws, size=n_paths)

        log_profile = self.log_profile[picks]                # (n_paths, n_bins)
        centres = 0.5 * (np.asarray(self.profile_edges[:-1]) + np.asarray(self.profile_edges[1:]))
        profile = np.stack(
            [np.interp(elevation, centres, log_profile[i]) for i in range(n_paths)],
            axis=0,
        )                                                       # (n_paths, T)
        rho = np.clip(self.posterior["rho"][picks], 0.0, 0.995)
        sigma = np.clip(
            self.posterior["sigma_process"][picks] * float(spread_factor), 1e-4, 2.0
        )
        obs_scale = np.clip(self.posterior["obs_scale"][picks], 1e-4, 1.0)
        season_cos = (
            self.season_cos[picks]
            if self.season_cos.size > 1
            else np.full(n_paths, float(self.season_cos[0]))
        )
        season_sin = (
            self.season_sin[picks]
            if self.season_sin.size > 1
            else np.full(n_paths, float(self.season_sin[0]))
        )
        use_season = self.seasonal_selected != "none"

        # -- AR(1) propagation with the night reset -------------------------
        state = origin_state or self.origin_state
        v0 = float(state.get("v_mean", 0.0))
        var0 = float(state.get("v_var", 0.0))
        v = np.zeros((n_paths, t_steps), dtype=float)
        for k in range(t_steps):
            shock = rng.standard_normal(n_paths)
            nxt = rho * (v[:, k - 1] if k else np.full(n_paths, v0)) + sigma * shock
            if k == 0 and var0 > 0.0:
                nxt = nxt + rng.normal(0.0, math.sqrt(var0), size=n_paths)
            v[:, k] = np.where(daylight[k], nxt, 0.0)

        # -- clearness and power --------------------------------------------
        if use_season:
            season = season_cos[:, None] * np.cos(theta)[None, :] + season_sin[:, None] * np.sin(
                theta
            )[None, :]
        else:
            season = np.zeros((n_paths, t_steps), dtype=float)
        base = profile * np.exp(np.clip(season, -0.51, 0.34))
        clearness = np.clip(np.exp(np.clip(v, -6.0, 3.0)), 0.0, CLEARNESS_MAX)
        u = np.clip(base * clearness, 0.0, CLEARNESS_MAX)
        power = (clear[None, :] + OBS_FLOOR_W) * u - OBS_FLOOR_W
        # Robust observation noise in log space (the reference likelihood).
        nu = 5.0
        noise = obs_scale[:, None] * rng.standard_t(nu, size=(n_paths, t_steps)) / math.sqrt(
            nu / 2.0
        )
        power = power * np.exp(np.clip(noise, -1.5, 1.5))
        # Night (and sub-threshold clear sky) rows are exactly zero.
        power[:, ~daylight] = 0.0
        power = np.maximum(power, 0.0)
        # The clearness ceiling also caps the absolute power at the physical
        # envelope the Tribunal enforces.
        rated = float(getattr(ctx, "rated_pv_w", None) or 0.0)
        if rated > 0.0:
            power = np.minimum(power, rated * 1.2)

        load = build_load_ensemble(frame, ctx, n_paths=n_paths, rng=rng)
        return {
            "solar_w": power,
            "load_w": load,
            "clearness_u": u,
            "dt_h": np.array([0.25]),
        }


def build_load_ensemble(
    frame: pd.DataFrame,
    ctx: Any,
    *,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The declared load ensemble (identical construction to the ops model).

    The DKASC export has no load meter, so the load is a *declared*
    engineering assumption and is not learned by either model.  Sharing the
    construction keeps the Tribunal comparison about the solar channel.
    """
    from chariot.model import build_load_ensemble as _ops_build_load_ensemble
    from chariot.model import load_reference_profile

    profile = np.nan_to_num(load_reference_profile(ctx, frame["timestamp_utc"]), nan=0.0)
    meta = getattr(ctx, "metadata", {}) or {}
    sigma = float(meta.get("load_relative_sigma", 0.0) or 0.0)
    load, _bias = _ops_build_load_ensemble(
        profile,
        n_scenarios=n_paths,
        t_steps=len(frame),
        declared_load_sigma=sigma,
        rng=rng,
    )
    return load


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _seasonal_validation_mae(
    fit_summary: dict[str, float],
    data_like: pd.DataFrame,
    ctx: Any,
    *,
    use_season: bool,
) -> float:
    """One-step-ahead MAE (W) of the clearness model on a held-out frame.

    Mirrors the ops model's selection rule: the seasonal harmonic is kept
    only when it lowers the validation error.
    """
    ordered = data_like.sort_values("timestamp_utc").reset_index(drop=True)
    solar = solar_series(ordered)
    clear = np.nan_to_num(clear_sky_series(ordered), nan=0.0)
    elevation = solar_elevation_deg(
        ordered["timestamp_utc"],
        float(getattr(ctx, "latitude", None) or 0.0),
        float(getattr(ctx, "longitude", None) or 0.0),
    )
    bins = elevation_bin(elevation)
    doy = pd.to_datetime(ordered["timestamp_utc"], utc=True).dt.day_of_year.to_numpy().astype(float)
    theta = 2.0 * math.pi * doy / 365.25

    profile = np.asarray(fit_summary["profile"], dtype=float)[bins]
    if use_season:
        season = np.exp(
            fit_summary["season_cos"] * np.cos(theta) + fit_summary["season_sin"] * np.sin(theta)
        )
    else:
        season = np.ones(len(ordered), dtype=float)
    gain = (clear + OBS_FLOOR_W) * profile * season - OBS_FLOOR_W

    # The AR(1) is propagated with the *posterior mean* parameters; the
    # one-step prediction is gain * exp(v_pred).
    rho = float(fit_summary["rho"])
    sigma = float(fit_summary["sigma_process"])
    obs_scale = float(fit_summary["obs_scale"])
    mask = (clear > 1.0) & np.isfinite(solar)
    if not mask.any():
        return float("inf")

    v = np.zeros(len(ordered), dtype=float)
    var = np.zeros(len(ordered), dtype=float)
    daylight = clear > 1.0
    for k in range(1, len(ordered)):
        v_pred = rho * v[k - 1]
        var_pred = rho**2 * var[k - 1] + sigma**2
        if not daylight[k]:
            v[k] = 0.0
            var[k] = 0.0
            continue
        obs = np.log((solar[k] + OBS_FLOOR_W) / (clear[k] + OBS_FLOOR_W)) - np.log(
            np.clip(gain[k] / (clear[k] + OBS_FLOOR_W), 1e-9, None)
        )
        if np.isfinite(obs):
            s2 = max(obs_scale**2, 1e-9)
            kg = var_pred / (var_pred + s2)
            v[k] = v_pred + kg * (obs - v_pred)
            var[k] = (1.0 - kg) * var_pred
        else:
            v[k] = v_pred
            var[k] = var_pred
    pred = gain * np.exp(np.clip(v, -5.0, 5.0))
    return float(np.mean(np.abs(solar[mask] - pred[mask])))


def _posterior_summary(fit: Any) -> dict[str, Any]:
    """Extract the hyperparameter draws and diagnostics from an InferenceData."""
    posterior = fit.posterior
    out: dict[str, np.ndarray] = {}
    for name in (
        "rho",
        "sigma_process",
        "obs_scale",
        "nu_excess",
        "fleet_log_prof_mu",
        "fleet_log_prof_sigma",
    ):
        if name in posterior:
            out[name] = np.asarray(posterior[name].values).reshape(-1)
    log_profile = np.asarray(posterior["log_profile"].values).reshape(
        -1, posterior["log_profile"].shape[-1]
    )
    season_cos = (
        np.asarray(posterior["season_cos"].values).reshape(-1)
        if "season_cos" in posterior
        else np.zeros(1)
    )
    season_sin = (
        np.asarray(posterior["season_sin"].values).reshape(-1)
        if "season_sin" in posterior
        else np.zeros(1)
    )
    return {
        "posterior": out,
        "log_profile": log_profile,
        "season_cos": season_cos,
        "season_sin": season_sin,
    }


def _convergence_summary(fit: Any) -> dict[str, Any]:
    """R-hat / ESS / divergence summary (empty for the VI fallback)."""
    summary: dict[str, Any] = {}
    try:
        import arviz as az

        rhat_max = 0.0
        ess_min = float("inf")
        for var in ("rho", "sigma_process", "obs_scale", "log_profile"):
            if var not in fit.posterior:
                continue
            rhat = az.rhat(fit, var_names=[var])
            ess = az.ess(fit, var_names=[var])
            summary[var] = {
                "rhat_max": float(np.max(np.asarray(rhat[var].values))),
                "ess_min": float(np.min(np.asarray(ess[var].values))),
            }
            rhat_max = max(rhat_max, summary[var]["rhat_max"])
            ess_min = min(ess_min, summary[var]["ess_min"])
        summary["rhat_max"] = float(rhat_max)
        summary["ess_min"] = float(0.0 if not np.isfinite(ess_min) else ess_min)
        if "diverging" in fit.sample_stats:
            div = np.asarray(fit.sample_stats["diverging"].values)
            summary["divergences"] = int(div.sum())
            summary["divergence_rate"] = float(div.mean()) if div.size else 0.0
    except Exception as exc:  # pragma: no cover - diagnostics are best effort
        summary["error"] = f"{type(exc).__name__}: {exc}"
    return summary


def fit_pymc_clearness(
    train: pd.DataFrame,
    ctx: Any,
    *,
    tenant_id: str,
    site_id: str,
    fleet_id: str = "fleet",
    dataset_version: str = "",
    dataset_content_hash: str = "",
    validation: pd.DataFrame | None = None,
    config: PymcReferenceConfig | None = None,
) -> PymcClearnessFit:
    """Fit the PyMC reference model on the **training** split.

    The test split is never touched.  The optional ``validation`` frame is
    used *only* to select the seasonal treatment, exactly as the ops model
    does.
    """
    require_pymc()
    config = config or PymcReferenceConfig()
    config.validate()

    pytensor_info = _configure_pytensor()
    data = prepare_clearness_data(train, ctx)
    backend = "pymc_vi"
    season = config.season in ("auto", "on")
    diagnostics: dict[str, Any] = {
        "n_state_rows": int(data.n),
        "n_obs_rows": int(data.n_obs),
        "n_bins": int(data.n_bins),
        "train_rows": int(len(train)),
        "pytensor_backend": pytensor_info.get("backend"),
        "pytensor_flags": PYTENSOR_FLAGS,
        "pytensor": pytensor_info,
        "config": config.to_dict(),
    }
    idata = None
    started = time.time()

    def _run_nuts() -> Any:
        model = build_pymc_model(data, season=season, config=config)
        with model:
            return pm.sample(
                draws=config.draws,
                tune=config.tune,
                chains=config.chains,
                cores=config.cores,
                target_accept=config.target_accept,
                max_treedepth=config.max_treedepth,
                random_seed=config.random_seed,
                progressbar=config.progressbar,
                compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": False},
            )

    def _run_vi() -> Any:
        model = build_pymc_model(data, season=season, config=config)
        with model:
            approx = pm.fit(
                method="advi",
                n=config.vi_iters,
                random_seed=config.random_seed,
                progressbar=config.progressbar,
            )
            samples = approx.sample(config.vi_samples, random_seed=config.random_seed)
        return pm.to_inference_data(samples)

    if config.backend in ("nuts", "auto"):
        try:
            idata = _run_nuts()
            backend = "pymc_nuts"
        except Exception as exc:
            diagnostics["nuts_error"] = f"{type(exc).__name__}: {exc}"
            if config.backend == "nuts":
                raise
            idata = _run_vi()
            backend = "pymc_vi"
    else:
        idata = _run_vi()
        backend = "pymc_vi"

    wall_time = time.time() - started
    summary = _posterior_summary(idata)
    diagnostics["wall_time_s"] = round(wall_time, 3)
    diagnostics["backend"] = backend
    diagnostics["chains"] = int(config.chains)
    diagnostics["draws"] = int(config.draws)
    diagnostics["tune"] = int(config.tune)
    diagnostics["convergence"] = _convergence_summary(idata)
    if backend == "pymc_vi":
        diagnostics["vi_iterations"] = int(config.vi_iters)
        diagnostics["vi_note"] = (
            "VI reference — the posterior is a Gaussian approximation, not a "
            "NUTS sample.  Do not quote it as a NUTS result."
        )

    # -- seasonal selection on the validation split -------------------------
    posterior_mean = {
        "profile": np.exp(np.mean(summary["log_profile"], axis=0)).tolist(),
        "rho": float(np.mean(summary["posterior"]["rho"])),
        "sigma_process": float(np.mean(summary["posterior"]["sigma_process"])),
        "obs_scale": float(np.mean(summary["posterior"]["obs_scale"])),
        "season_cos": float(np.mean(summary["season_cos"])),
        "season_sin": float(np.mean(summary["season_sin"])),
    }
    seasonal_selected = "annual_harmonic"
    season_selection: dict[str, Any] = {"fitted": season}
    if validation is not None and len(validation) and season:
        mae_with = _seasonal_validation_mae(posterior_mean, validation, ctx, use_season=True)
        mae_without = _seasonal_validation_mae(posterior_mean, validation, ctx, use_season=False)
        if mae_without < mae_with:
            seasonal_selected = "none"
        season_selection.update(
            {
                "validation_mae_with_season": round(mae_with, 3),
                "validation_mae_without_season": round(mae_without, 3),
                "selected": seasonal_selected,
            }
        )
    elif not season:
        seasonal_selected = "none"
        season_selection["selected"] = "none"

    # -- model version / lineage -------------------------------------------
    model_version = "pymc_clearness_v3.0_{}_{}".format(
        backend.replace("pymc_", ""),
        short_hash(hash_obj({"log_profile": np.mean(summary["log_profile"], axis=0).tolist()}), 10),
    )
    diagnostics["season_selection"] = season_selection
    diagnostics["posterior_mean"] = {
        "rho": posterior_mean["rho"],
        "sigma_process": posterior_mean["sigma_process"],
        "obs_scale": posterior_mean["obs_scale"],
        "season_cos": posterior_mean["season_cos"],
        "season_sin": posterior_mean["season_sin"],
    }

    # -- origin state (filtered at the end of the training window) ----------
    from chariot.model import filter_state_series

    reference_model = build_reference_model_shell(
        posterior_mean,
        ctx,
        tenant_id=tenant_id,
        site_id=site_id,
        fleet_id=fleet_id,
        dataset_version=dataset_version,
        dataset_content_hash=dataset_content_hash,
    )
    filt = filter_state_series(reference_model, site_id, train)
    origin_state = {
        "v_mean": float(filt["filtered"][-1]),
        "v_var": float(filt["filtered_var"][-1]),
    }

    if not config.keep_state:
        # The per-step latent state is large and is not needed downstream.
        try:
            idata.posterior = idata.posterior.drop_vars(
                [v for v in ("weather_state",) if v in idata.posterior]
            )
        except Exception:  # pragma: no cover - defensive
            pass

    return PymcClearnessFit(
        backend=backend,
        model_version=model_version,
        model_name="chariot_pymc_clearness_reference",
        tenant_id=tenant_id,
        site_id=site_id,
        fleet_id=fleet_id,
        dataset_version=dataset_version,
        dataset_content_hash=dataset_content_hash,
        posterior=summary["posterior"],
        log_profile=summary["log_profile"],
        profile_edges=[float(v) for v in ELEVATION_BINS],
        season_cos=summary["season_cos"],
        season_sin=summary["season_sin"],
        seasonal_selected=seasonal_selected,
        season_selection=season_selection,
        origin_state=origin_state,
        diagnostics=diagnostics,
        n_state_rows=int(data.n),
        n_obs_rows=int(data.n_obs),
        config=config,
        extra={"idata": idata},
    )


def build_reference_model_shell(
    posterior_mean: dict[str, Any],
    ctx: Any,
    *,
    tenant_id: str,
    site_id: str,
    fleet_id: str,
    dataset_version: str,
    dataset_content_hash: str,
) -> ChariotModel:
    """A v3 :class:`ChariotModel` carrying the PyMC posterior summary.

    Used for the origin-state filter and for the lineage record.  It is *not*
    the operational model: the operational model is always
    :func:`chariot.model.fit`.
    """
    channel = ChannelFit(
        name="clearness",
        mu=0.0,
        rho=float(posterior_mean["rho"]),
        sigma=float(posterior_mean["sigma_process"]),
        gamma=1.0,
        obs_scale=float(posterior_mean["obs_scale"]),
        obs_floor=OBS_FLOOR_W,
        n_obs=0,
        profile=[float(v) for v in posterior_mean["profile"]],
        profile_edges=[float(v) for v in ELEVATION_BINS],
        season_cos=float(posterior_mean["season_cos"]),
        season_sin=float(posterior_mean["season_sin"]),
        se={"rho": 0.05, "sigma": 0.02, "obs_scale": 0.01},
    )
    site_model = SiteModel(
        tenant_id=tenant_id,
        site_id=site_id,
        solar=channel,
        load=ChannelFit(name="load"),
        load_observed=False,
    )
    fleet = FleetPosterior(fleet_id=fleet_id, site_ids=[site_id])
    return ChariotModel(
        model_name="chariot_pymc_clearness_reference",
        model_version="pymc_reference_shell",
        tenant_id=tenant_id,
        fleet_id=fleet_id,
        dataset_version=dataset_version,
        dataset_content_hash=dataset_content_hash,
        sites={site_id: site_model},
        fleet=fleet,
        site_contexts={site_id: ctx.to_dict() if hasattr(ctx, "to_dict") else {}},
    )


def build_reference_model(
    fit: PymcClearnessFit,
    ctx: Any,
) -> ChariotModel:
    """Build a v3 :class:`ChariotModel` from a reference fit (lineage)."""
    channel = fit.to_channel_fit()
    site_model = SiteModel(
        tenant_id=fit.tenant_id,
        site_id=fit.site_id,
        solar=channel,
        load=ChannelFit(name="load"),
        load_observed=False,
    )
    fleet = FleetPosterior(fleet_id=fit.fleet_id, site_ids=[fit.site_id])
    return ChariotModel(
        model_name=fit.model_name,
        model_version=fit.model_version,
        tenant_id=fit.tenant_id,
        fleet_id=fit.fleet_id,
        dataset_version=fit.dataset_version,
        dataset_content_hash=fit.dataset_content_hash,
        sites={fit.site_id: site_model},
        fleet=fleet,
        site_contexts={fit.site_id: ctx.to_dict() if hasattr(ctx, "to_dict") else {}},
    )
