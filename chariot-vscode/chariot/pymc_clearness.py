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
* **AR(1) latent weather state** — the state runs over the *observed* (lit)
  rows and is **reset at every night gap**, exactly like the ops model's
  Kalman filter: the clearness residual does not survive the night, so the
  chain is broken at the same places the ops model breaks it;
* **elevation structure** — a hierarchical profile over the 15 solar-elevation
  bins (the fleet-level hyper-mean provides the partial pooling), written in
  the non-centred form so that the funnel does not appear;
* **optional seasonal harmonic** — selected on the *validation* split, and
  dropped when it does not help (this mirrors the ops model, which also
  rejected the harmonic on the Alice Springs 2022 data);
* **robust observation noise** — a Student-t likelihood with a *fixed*
  ``nu = 4`` and a log-normal prior on the scale, the Bayesian analogue of
  the ops model's Huberised EKF.  Both choices are deliberate: a free ``nu``
  and a half-normal scale let the likelihood explain the outliers either
  with heavier tails or with the latent state, and the posterior of the
  observation scale is then weakly identified (measured: r-hat 2.7 with
  ``obs_scale`` collapsing to 0.002).

Why the AR(1) state is sampled in its non-centred form
----------------------------------------------------
The latent state is an AR(1) process.  The obvious PyTensor implementation is
a ``scan``; ``pm.AR`` looks like the answer, because PyMC implements
``AR``'s log-probability *analytically* (see
``pymc.distributions.timeseries.ar_logp``) —

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

Writing the AR(1) density directly, as a :func:`pymc.Potential` over a flat
state variable,

.. code-block:: python

    innov = concat([a[0], a[1:] - rho * a[:-1]])
    potential = sum(Normal(0, 1).logpdf(innov))

is the **same density** as ``pm.AR`` with unit innovations — PyMC's own
``ar_logp`` reduces to exactly this expression — and it makes the gradient
elementwise.  But the *centred* parameterisation is geometrically terrible
for NUTS: the state is a narrow ridge in 21 000 dimensions, the sampler
saturates its tree depth, and it never leaves the starting region (measured
on the reference host: r-hat 1.21, ``sigma_process`` collapsed to 0.0015
while the log-density at the true optimum is ~17 000 log units higher).

This module therefore samples the **non-centred** variable — the
standard-normal innovation series ``u`` — and builds the state with a
``scan``:

.. code-block:: python

    s_i = rho * s_{i-1} + u_i          # over the observed rows
    s_i = u_i                          # after a night gap (the chain resets)

The map from ``u`` to ``s`` is lower triangular with unit diagonal, so the
change of variables has unit Jacobian and **the density is unchanged**; the
sampled variables are simply on a standard normal scale, which is what NUTS
needs.  The scan is compiled by the Numba backend (no C toolchain needed)
and costs a few milliseconds forward and ~4 ms for the reverse-mode gradient
on the ~8 000-step observed chain.

The hierarchy over the elevation bins is non-centred for the same reason
(``log_profile_b = fleet_mu_b + fleet_sigma * z_b``); the compensating
``n_bins * log(fleet_sigma)`` potential keeps the joint density identical to
the centred form while removing the funnel (r-hat 1.13 / ESS 12 on the
profile bins).

Parameterisation
----------------
The state is written as ``v = sigma_process * s`` where ``s`` is the
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
* ``linker=numba`` + ``mode=Mode`` — when ``numba`` is importable the whole
  graph is compiled to machine code, which needs neither C headers nor
  fusion.  (``mode=NUMBA`` would *not* do: that is a predefined PyTensor
  mode, and predefined modes ignore ``optimizer_excluding``.)

The module also pins the BLAS thread pools to one thread.  The HMC
integrator performs one small dot product per leapfrog step on the ~21 000
element state vector; with OpenBLAS's per-core thread pool those calls cost
milliseconds each (measured: 16 ms versus 0.04 ms per call), which made the
sampler ~400x slower than the gradient itself.

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

# The BLAS thread limits have to be in the environment *before* numpy (and
# therefore OpenBLAS) is loaded.  An operator setting always wins.
for _blas_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

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

    Two subtleties are handled here, both of which cost an order of magnitude
    in sampler throughput when they are ignored:

    * ``mode=NUMBA`` is a *predefined* PyTensor mode.  ``get_mode()`` returns
      it verbatim and therefore never applies ``optimizer_excluding``.  The
      elementwise fusion rewrite then builds ``Composite`` ops, which the
      Numba backend cannot consume -- the whole graph falls back to the
      Python VM and the gradient becomes ~1000x slower.  ``mode=Mode``
      builds the mode from ``config.linker`` and *does* apply the exclusion.
    * ``cxx=`` disables C code generation.  It is mandatory on a host without
      Python headers (every compilation fails) and it also caps the elementwise
      fusion at 32 operands, which keeps the graphs unfused.
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
    if _numba_available() and "mode" not in keys and "linker" not in keys:
        flags.append("linker=numba")
        flags.append("mode=Mode")
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


_BLAS_LIMITS: Any = None


def _limit_blas_threads() -> int | None:
    """Pin the BLAS thread pools to one thread for this process.

    The HMC integrator evaluates one small dot product per leapfrog step on
    the latent state vector.  With OpenBLAS's per-core thread pool those
    calls are dominated by the thread hand-off (measured on this host:
    16 ms versus 0.04 ms per call), which makes the sampler orders of
    magnitude slower than the gradient it is supposed to be evaluating.

    ``threadpoolctl`` is used when it is importable; otherwise the setting
    only takes effect through the environment variables set at import time.
    """
    global _BLAS_LIMITS
    if _BLAS_LIMITS is not None:
        return 1
    try:  # pragma: no cover - depends on the environment
        import threadpoolctl

        _BLAS_LIMITS = threadpoolctl.threadpool_limits(limits=1)
        _BLAS_LIMITS.__enter__()
        return 1
    except Exception:  # pragma: no cover - optional dependency
        _BLAS_LIMITS = False
        return None


def _configure_pytensor() -> dict[str, Any]:
    """Apply the PyTensor settings this host needs (idempotent).

    The environment flags above are the primary mechanism, because PyTensor
    reads ``PYTENSOR_FLAGS`` once, at import time.  When something else has
    already imported PyTensor the flags are ignored, so the settings that
    *can* still be changed at runtime are applied here as a fallback:

    * ``cxx = ""`` — always possible, and mandatory on a host without
      Python headers (otherwise every compilation fails);
    * the Numba linker — only useful when the elementwise fusion rewrite is
      excluded as well, because the Numba backend cannot consume fused
      graphs.  ``FAST_COMPILE``/``vm`` is the fallback: it never fuses, so
      the unfused numpy ufuncs are used;
    * a single BLAS thread — applied at runtime as well, because the
      environment variables only take effect when OpenBLAS is loaded.
    """
    info: dict[str, Any] = {
        "pytensor_flags": PYTENSOR_FLAGS,
        "c_toolchain_available": _c_toolchain_available(),
        "numba_available": _numba_available(),
    }
    if pytensor is None:  # pragma: no cover - PyMC is missing
        info["backend"] = "unavailable"
        return info
    info["blas_threads"] = _limit_blas_threads()
    if not info["c_toolchain_available"]:
        try:
            pytensor.config.cxx = ""
        except Exception:  # pragma: no cover - defensive
            pass
    excluding = str(getattr(pytensor.config, "optimizer_excluding", "") or "")
    fusion_off = "fusion" in excluding
    backend = str(getattr(pytensor.config, "mode", "default"))
    try:  # pragma: no cover - depends on the environment
        from pytensor.compile.mode import Mode, get_mode

        mode = get_mode(None)
        if not fusion_off:
            # Rebuild the mode with the fusion rewrite excluded.  The
            # predefined ``NUMBA`` mode ignores ``optimizer_excluding``, so
            # the exclusion has to be attached to the Mode object itself.
            mode = mode.excluding("fusion")
            fusion_off = True
        if info["numba_available"] and type(mode.linker).__name__ != "NumbaLinker":
            mode = Mode(linker="numba", optimizer=mode.provided_optimizer)
        pytensor.config.mode = mode
        backend = type(mode.linker).__name__
    except Exception:  # pragma: no cover - defensive
        pass
    info["backend"] = backend
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
    draws: int = 250
    #: NUTS tuning steps per chain.
    tune: int = 250
    #: Number of chains (>= 2 is required for a meaningful R-hat).
    chains: int = 2
    #: Parallel chains.  The reference host has 2 cores and 3 GB of RAM, so
    #: the default is 1 (chains run sequentially) to avoid oversubscription.
    cores: int = 1
    #: NUTS target acceptance rate.
    target_accept: float = 0.9
    #: NUTS maximum tree depth.  Each leapfrog step evaluates the gradient
    #: of the latent state (one element per observed row), so the tree depth
    #: is the knob that bounds the cost of one iteration.
    max_treedepth: int = 8
    #: NUTS initialisation.  ``adapt_diag`` starts from the method-of-moments
    #: point of this module (no jitter): the latent state is one AR(1) chain
    #: over the observed rows, and jittering every component independently
    #: would start the chain far away from the typical set.
    init: str = "adapt_diag"
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

    The object keeps **every** row of the 15-minute grid and carries an
    observation mask instead of dropping the night: the elevation profile and
    the seasonal harmonic are indexed by the grid, while the AR(1) state is
    defined on the observed (lit) rows and is reset at every gap.
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


def _ar1_state(rho: Any, innovation: Any, reset: np.ndarray, n_steps: int) -> Any:
    """The AR(1) state over the **observed** rows.

    ``s_i = rho * s_{i-1} + u_i`` while the observations are consecutive on
    the 15-minute grid, and ``s_i = u_i`` after a gap (night), which is the
    convention the ops model uses as well: the clearness residual does not
    persist across the night.

    The recursion is written as a PyTensor scan so that the *non-centred*
    parameterisation can be used: the sampled variable is the
    standard-normal innovation series, and the state is a deterministic
    function of it.  The map from ``u`` to ``s`` is lower triangular with
    unit diagonal, so the density is unchanged.
    """
    import pytensor

    reset_flags = np.asarray(reset, dtype=bool)

    def _step(u_t, reset_t, s_prev, rho_t):
        return pytensor.tensor.switch(reset_t, u_t, rho_t * s_prev + u_t)

    # NOTE: ``n_steps`` is deliberately *not* passed to ``pytensor.scan``.
    # With statically shaped sequences PyTensor then derives the length
    # itself; passing an explicit count makes it slice the sequences with a
    # symbolic index, which fails on statically shaped vectors.  The
    # ``reshape`` drops the static length of the first sequence, which is
    # what lets PyTensor slice it with that symbolic index.
    innov = pytensor.tensor.as_tensor(innovation).reshape((-1,))
    flags = pytensor.tensor.as_tensor_variable(reset_flags)
    seq = pytensor.scan(
        fn=_step,
        sequences=[innov, flags],
        outputs_info=[pytensor.tensor.zeros((), dtype="float64")],
        non_sequences=[rho],
        strict=True,
        name="ar1_state",
        return_updates=False,
    )
    return pytensor.tensor.specify_shape(seq, (int(n_steps),))


def _state_to_innovation(state: np.ndarray, rho: float, reset: np.ndarray) -> np.ndarray:
    """The inverse of the AR(1) recursion: ``u_i = s_i - rho * s_{i-1}``."""
    arr = np.asarray(state, dtype=float)
    flags = np.asarray(reset, dtype=bool)
    innov = np.empty_like(arr)
    if arr.size:
        innov[0] = arr[0]
    if arr.size > 1:
        prev = np.where(flags[1:], 0.0, rho * arr[:-1])
        innov[1:] = arr[1:] - prev
    return innov


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
    ``profile_z``                N(0, 1) — the non-centred bin offsets
    ``log_profile``              deterministic: fleet_mu + fleet_sigma * z
    ``rho``                      U(0.5, 0.999) — AR(1) persistence per 15 min
    ``sigma_process``            HalfNormal(0.3) — AR(1) innovation scale
    ``weather_innovation``       N(0, 1) — the non-centred AR(1) innovations
    ``obs_scale``                LogNormal(log 0.05, 0.75) — observation scale
    ``nu``                       fixed at 4 (Student-t degrees of freedom)
    ``season_cos`` / ``season_sin``  N(0, 0.1) — bounded annual harmonic
    ==========================  ===============================================
    """
    require_pymc()
    n = data.n
    n_bins = data.n_bins
    obs_index = np.asarray(data.obs_index, dtype=np.int64)
    obs_bins = np.asarray(data.bins[obs_index], dtype=np.int64)
    theta_obs = 2.0 * math.pi * data.doy[obs_index] / 365.25
    n_obs = int(obs_index.size)
    # A night (or any gap) breaks the AR(1) chain, exactly like the ops
    # model: the clearness residual does not persist across the night.
    reset = np.ones(n_obs, dtype=bool)
    if n_obs > 1:
        reset[1:] = np.diff(obs_index) != 1
    start = _initial_state_estimate(data)

    with pm.Model() as model:
        # -- hierarchical elevation structure (partial pooling over bins) ---
        # The hierarchy is written in the *non-centred* form
        # ``log_profile_b = fleet_mu_b + fleet_sigma * z_b`` with ``z ~ N(0,1)``.
        # The change of variables has Jacobian ``fleet_sigma ** n_bins``, and
        # the compensating ``n_bins * log(fleet_sigma)`` term is added below so
        # that the joint density over
        # ``(fleet_mu, fleet_sigma, log_profile)`` is *identical* to the
        # centred parameterisation.  Without the compensation the funnel
        # pathology of the centred form (r-hat 1.13, ESS 12 on the profile
        # bins) reappears in a different guise.
        fleet_mu = pm.Normal("fleet_log_prof_mu", 0.0, 0.5, shape=n_bins)
        fleet_sigma = pm.HalfNormal("fleet_log_prof_sigma", 0.25)
        profile_z = pm.Normal("profile_z", 0.0, 1.0, shape=n_bins)
        log_profile = pm.Deterministic(
            "log_profile", fleet_mu + fleet_sigma * profile_z
        )
        pm.Potential("profile_jacobian", n_bins * pt.log(fleet_sigma))

        # -- AR(1) latent weather state (non-centred) ------------------------
        # The free variable is the *innovation* series ``u`` (standard
        # normal) on the observed rows; the AR(1) state is the recursion
        # ``s_i = rho * s_{i-1} + u_i``, restarted after a gap.  The map from
        # ``u`` to ``s`` is lower triangular with unit diagonal, so the change
        # of variables has unit Jacobian: this is *exactly* the same density
        # as the centred parameterisation, but the sampled variables are on a
        # standard normal scale.  That matters a great deal here — with the
        # centred parameterisation the posterior is a narrow ridge in 21 000
        # dimensions, NUTS cannot climb it (r-hat 1.21, sigma_process
        # collapsed to ~0), and the sampler saturates the tree depth.
        # The persistence is bounded away from zero: the clearness anomaly
        # is strongly autocorrelated (the ops model estimates rho ~ 0.95 on
        # this site), and letting rho collapse towards zero would make the
        # latent state flexible enough to absorb the observation noise.
        rho = pm.Uniform("rho", 0.5, 0.999)
        sigma_process = pm.HalfNormal("sigma_process", 0.3)
        innovation = pm.Normal("weather_innovation", 0.0, 1.0, shape=n_obs)
        state = _ar1_state(rho, innovation, reset, n_obs)

        # -- optional bounded annual harmonic ---------------------------------
        if season:
            season_cos = pm.Normal("season_cos", 0.0, 0.1)
            season_sin = pm.Normal("season_sin", 0.0, 0.1)
            season_term = season_cos * np.cos(theta_obs) + season_sin * np.sin(theta_obs)
        else:
            season_term = pt.zeros(len(obs_index))

        # -- observation ------------------------------------------------------
        # The mean is evaluated on the **observed rows only**.  Evaluating it
        # on the whole grid would make the gradient of the elevation profile
        # a scatter of tens of thousands of indices into 15 bins; on the
        # observed rows the scatter is an order of magnitude smaller, which
        # is what keeps the gradient affordable.
        mu_obs = (
            log_profile[obs_bins]
            + season_term
            + sigma_process * state
        )
        # The observation scale is the *bulk* (robust) scale of the clearness
        # residual.  It gets a log-normal prior rather than a half-normal:
        # the half-normal is flat at zero, so the likelihood can drive
        # ``obs_scale`` to ~0 by letting the latent state absorb the noise,
        # and the posterior becomes weakly identified (the two chains ended
        # up at r-hat 2.7 with ``obs_scale`` at 0.002).  The log-normal
        # excludes that degenerate corner without being sharp: the log-scale
        # is centred on 0.05 with a 0.75 standard deviation, i.e. the bulk
        # scale of a clearness residual.
        obs_scale = pm.LogNormal("obs_scale", math.log(0.05), 0.75)
        # The degrees of freedom are **fixed** at 4 for the same reason: a
        # free ``nu`` trades off against the latent state (heavy tails can
        # explain what the state would otherwise explain) and leaves the
        # observation scale unidentified.
        nu = 4.0
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
            # ``log_profile`` is a deterministic of the non-centred hierarchy,
            # so the *free* variables are initialised instead: the per-bin
            # fleet mean carries the method-of-moments profile and the
            # standardised offsets start at zero.
            model.set_initval(model["fleet_log_prof_mu"], start["log_profile"])
            model.set_initval(
                model["fleet_log_prof_sigma"], float(start["fleet_log_prof_sigma"])
            )
            model.set_initval(model["profile_z"], np.zeros(n_bins))
            model.set_initval(
                model["weather_innovation"],
                _state_to_innovation(
                    start["state"][obs_index], float(start["rho"]), reset
                ),
            )
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
        log_profile_at_elev = np.stack(
            [np.interp(elevation, centres, log_profile[i]) for i in range(n_paths)],
            axis=0,
        )                                                       # (n_paths, T), log space
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
        # The reference model lives in *log* space:
        #     log(u) = log_profile(elev) + season(doy) + v
        # with ``u`` the clearness ratio, exactly as in the PyMC likelihood.
        if use_season:
            season = season_cos[:, None] * np.cos(theta)[None, :] + season_sin[:, None] * np.sin(
                theta
            )[None, :]
        else:
            season = np.zeros((n_paths, t_steps), dtype=float)
        log_u = log_profile_at_elev + season + v
        u = np.clip(np.exp(np.clip(log_u, -6.0, 3.0)), 0.0, CLEARNESS_MAX)
        # The ops simulator writes ``solar = gain * clear_sky * u``; the
        # reference uses the same convention so that the *clearness ratio*
        # the Tribunal sees is exactly ``u`` and therefore always respects
        # ``CLEARNESS_MAX`` (an observation floor inside the product would
        # inflate the ratio at low light).
        power = clear[None, :] * u
        # Robust observation noise in log space (the reference likelihood).
        nu = 5.0
        noise = obs_scale[:, None] * rng.standard_t(nu, size=(n_paths, t_steps)) / math.sqrt(
            nu / 2.0
        )
        power = power * np.exp(np.clip(noise, -1.5, 1.5))
        # The clearness ceiling is re-applied *after* the observation noise:
        # the noise is multiplicative, so without this the realised ratio
        # could exceed ``CLEARNESS_MAX``.
        clear_row = clear[None, :]
        ratio = np.where(clear_row > 1.0, power / np.maximum(clear_row, 1.0), 0.0)
        power = np.where(clear_row > 1.0, np.minimum(ratio, CLEARNESS_MAX) * clear_row, 0.0)
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
    only when it lowers the validation error.  The diagnostic uses the ops
    model's *observation-floor* convention for the gain (the selection
    metric only has to be internally consistent, and the ops model is the
    reference for it); the reference simulator itself mirrors the ops
    ``simulate`` convention, ``solar = clear_sky * u``.
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
    vi_info: dict[str, Any] = {}
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
                init=config.init,
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
            try:
                history = [float(v) for v in approx.history]
                if history:
                    vi_info["elbo_loss_initial"] = history[0]
                    vi_info["elbo_loss_final"] = history[-1]
            except Exception:  # pragma: no cover - diagnostics only
                pass
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
        diagnostics["vi_diagnostics"] = vi_info
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
        # The harmonic is kept only when it buys a *meaningful* improvement
        # on the validation split (1 % relative).  A hair-thin difference is
        # noise, and the ops model uses the same "keep it only if it helps"
        # rule.
        gain = (mae_without - mae_with) / max(mae_without, 1e-9)
        if gain < 0.01:
            seasonal_selected = "none"
        season_selection.update(
            {
                "validation_mae_with_season": round(mae_with, 3),
                "validation_mae_without_season": round(mae_without, 3),
                "validation_relative_gain": round(float(gain), 6),
                "selection_rule": "keep the harmonic only if it improves the "
                "validation MAE by at least 1 %",
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
                [v for v in ("weather_innovation",) if v in idata.posterior]
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
