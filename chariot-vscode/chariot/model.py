"""CHARIOT model — hierarchical clear-sky Bayesian state-space world model.

PDF §4.7.  The predictive engine is a single hierarchical clear-sky Bayesian
state-space model anchored on the clear-sky ratio

    R_t = P_meas / max(P_clear, eps)

Latent state ``z_t`` (per site): the solar *clearness* state ``u_t`` (the
multiplier that maps clear-sky power onto measured power), the load
*scale* state ``v_t``, and — through the physics transition — the state of
charge ``s_t``.

Hierarchy: ``equipment -> site -> fleet`` partial pooling over the
hierarchical parameters ``theta`` (clearness level, AR persistence, process
noise, PV scale, load scale, observation noise).

Observation model: Student-t (robust).  Implemented as a Huberised Kalman
filter, which is the standard Gaussian approximation to a Student-t
observation model.
Transition model: physics-informed — the SOC is coupled to the calibrated
usable capacity through the deterministic flow solver.

Inference: empirical Bayes with moment-matched hierarchical posteriors, so
the model runs on CPU with NumPy only.  The interface is deliberately the
one a PyMC/NUTS implementation would expose, so the sampler can replace this
module later without touching the Tribunal or the three advisory modes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from chariot.hashing import hash_obj, short_hash

MODEL_NAME = "chariot_hcbsswm"

#: Physical upper bound of the clearness state ``u = P_meas / P_clear``.
#: The clear-sky model is itself uncertain: the measured plane-of-array
#: irradiance cannot exceed the clear-sky estimate by more than roughly a
#: third at this site (the observed maximum ratio at high irradiance is
#: ~1.23).  Bounding the state keeps the ensemble physically admissible
#: instead of letting a symmetric innovation produce impossible futures.
CLEARNESS_MAX = 1.25
MODEL_VERSION = "hcbsswm_v3.0_empirical_bayes"



def clear_sky_series(frame: pd.DataFrame) -> np.ndarray:
    """Clear-sky power series (accepts the raw or the extract column name)."""
    for col in ("clear_sky_w", "clear_sky_power_w"):
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    raise KeyError("clear-sky column not found in frame")


def solar_series(frame: pd.DataFrame) -> np.ndarray:
    """Measured solar power series."""
    for col in ("solar_w", "solar_effective_w"):
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    raise KeyError("solar column not found in frame")


# --------------------------------------------------------------------------
# Robust helpers
# --------------------------------------------------------------------------


def _huber_weights(resid: np.ndarray, scale: float, c: float = 1.345) -> np.ndarray:
    """Huber M-estimator weights (robust Student-t surrogate)."""
    if scale <= 0 or not np.isfinite(scale):
        return np.ones_like(resid)
    z = np.abs(resid) / (c * scale)
    return np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-12))


def _robust_scale(resid: np.ndarray) -> float:
    """MAD-based scale estimator (falls back to the standard deviation)."""
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return 0.0
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) * 1.4826
    if mad <= 0:
        mad = float(np.std(resid))
    return max(mad, 1e-9)


def _irls_ar1(
    u: np.ndarray,
    mask: np.ndarray,
    max_iter: int = 25,
    weights: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Robust AR(1) fit ``u_t = mu + rho (u_{t-1} - mu)``.

    Only *consecutive* pairs ``(t-1, t)`` whose two members are both inside
    ``mask`` are used, so that unobserved stretches (night) never leak into
    the regression as pseudo-pairs.  ``weights`` down-weights rows whose
    measurement is uninformative (e.g. low irradiance, where the clear-sky
    ratio is dominated by measurement noise).

    Returns ``(mu, rho, residuals)``.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask & np.isfinite(u))[0]
    if idx.size < 8:
        return float(np.nanmean(u[mask]) if mask.any() else 1.0), 0.0, np.zeros_like(u)
    keep = np.array([i for i in idx if (i - 1) >= 0 and mask[i - 1]], dtype=int)
    if keep.size < 8:
        return float(np.nanmean(u[mask])), 0.0, np.zeros_like(u)
    x = u[keep - 1]
    y = u[keep]
    w_base = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)[keep]
    w_base = np.where(np.isfinite(w_base) & (w_base > 0), w_base, 0.0)
    if w_base.sum() <= 0:
        w_base = np.ones_like(x)
    w = w_base.copy()
    mu, rho = float(np.mean(y)), 0.0
    # The level of the state is estimated as the (weighted) sample mean of
    # the series.  The textbook alternative
    # ``mu = (mean(y) - rho mean(x)) / (1 - rho)`` is ill-conditioned when
    # rho approaches one (which is the normal case for a slow weather
    # state) and would amplify rounding noise into a large level bias.
    mu = float(np.sum(w_base * y) / w_base.sum())
    for _ in range(max_iter):
        sw = w.sum()
        if sw <= 0:
            break
        xm = float(np.sum(w * x) / sw)
        ym = float(np.sum(w * y) / sw)
        sxx = float(np.sum(w * (x - xm) ** 2))
        if sxx <= 1e-12:
            rho = 0.0
        else:
            rho = float(np.sum(w * (x - xm) * (y - ym)) / sxx)
        rho = float(np.clip(rho, -0.95, 0.995))
        resid = y - (mu + rho * (x - mu))
        scale = _robust_scale(resid)
        w = w_base * _huber_weights(resid, scale)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() <= 0:
            w = w_base.copy()
    resid = np.full_like(u, np.nan)
    resid[keep] = y - (mu + rho * (x - mu))
    return mu, rho, resid


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median (falls back to the plain median)."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(np.median(v))
    idx = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(v[min(idx, v.size - 1)])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean (falls back to the plain mean)."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    w = weights[mask]
    total = float(w.sum())
    if total <= 0:
        return float(np.nanmean(values[mask]))
    return float(np.sum(w * values[mask]) / total)


def _diurnal_profile(
    state: np.ndarray,
    obs: np.ndarray,
    clear: np.ndarray,
    hours: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray, float]:
    """Clear-weighted hour-of-day profile of the latent clearness state.

    The clear-sky ratio carries a systematic geometry dependence (the
    clear-sky model's bias varies with the solar zenith angle), so the
    profile is estimated per UTC hour on the *filtered* state, weighted by
    the clear-sky level (the ratio is uninformative at low irradiance).

    The profile is returned **normalised**: its clear-weighted mean over the
    observed rows is 1, and the overall level is returned separately.  This
    makes the (profile, PV bias) pair identifiable: the profile carries the
    *shape* of the diurnal pattern, the level carries the PV bias.
    """
    clean = np.where(np.isfinite(state), state, np.nan)
    clipped = np.clip(clean, 0.0, 2.5)
    profile = np.zeros(24, dtype=float)
    for h in range(24):
        sel = obs & (hours == h) & np.isfinite(clipped)
        if sel.sum() >= 3:
            w = clear[sel]
            total_w = float(w.sum())
            if total_w > 0:
                profile[h] = float(np.sum(w * clipped[sel]) / total_w)
    known = np.flatnonzero(profile > 0)
    if known.size:
        for h in range(24):
            if profile[h] <= 0:
                profile[h] = float(np.interp(h, known, profile[known]))
    profile = np.clip(profile, 0.0, 2.5)

    level = 1.0
    if normalize:
        sel = obs & np.isfinite(clipped)
        if sel.any():
            w = clear[sel]
            total_w = float(w.sum())
            if total_w > 0:
                level = float(np.sum(w * clipped[sel]) / total_w)
        if level <= 1e-6:
            level = 1.0
        profile = profile / level
    return profile, level


def _through_origin_regression(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Weighted regression of ``y = beta * x`` through the origin.

    Returns ``(beta, standard_error)``.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        mask &= np.isfinite(weights)
    if mask.sum() < 5:
        return 1.0, float("inf")
    xm = x[mask]
    ym = y[mask]
    w = np.ones_like(xm) if weights is None else weights[mask]
    denom = float(np.sum(w * xm * xm))
    if denom <= 1e-12:
        return 1.0, float("inf")
    beta = float(np.sum(w * xm * ym) / denom)
    resid = ym - beta * xm
    dof = max(int(mask.sum()) - 1, 1)
    sigma2 = float(np.sum(w * resid * resid) / dof)
    se = math.sqrt(sigma2 / denom) if denom > 0 else float("inf")
    return beta, se


# --------------------------------------------------------------------------
# Channel fit
# --------------------------------------------------------------------------


#: Elevation bin edges (degrees) used to index the diurnal clearness profile.
ELEVATION_BINS = np.arange(0.0, 90.0 + 6.0, 6.0)


def seasonal_factor(
    season_cos: float, season_sin: float, day_of_year: np.ndarray
) -> np.ndarray:
    """Bounded annual harmonic of the clearness level.

    A single annual harmonic is used instead of a linear trend: the factor
    stays bounded (it repeats every year) so that extrapolating beyond the
    training window cannot run away the way a linear trend does.
    """
    theta = 2.0 * math.pi * np.asarray(day_of_year, dtype=float) / 365.25
    log_factor = season_cos * np.cos(theta) + season_sin * np.sin(theta)
    return np.clip(np.exp(log_factor), 0.6, 1.4)


def solar_elevation_deg(
    timestamps: pd.Series | pd.DatetimeIndex, latitude: float, longitude: float
) -> np.ndarray:
    """Solar elevation angle (degrees) for UTC timestamps.

    Standard NOAA-style approximation: declination, equation of time, hour
    angle and the resulting elevation.  Used as the *season-invariant*
    index of the diurnal clearness profile: the clear-sky model's bias
    depends on the solar zenith angle, not on the clock hour.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if len(ts) == 0:
        return np.zeros(0, dtype=float)
    day_of_year = ts.day_of_year.to_numpy().astype(float)
    hour_utc = (
        ts.hour.to_numpy().astype(float)
        + ts.minute.to_numpy().astype(float) / 60.0
        + ts.second.to_numpy().astype(float) / 3600.0
    )
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1.0 + (hour_utc - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    time_offset = eqtime + 4.0 * longitude
    true_solar_time = (hour_utc * 60.0 + time_offset) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0
    lat = math.radians(latitude)
    ha = np.radians(hour_angle)
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    return 90.0 - zenith


def elevation_bin(elevation: np.ndarray) -> np.ndarray:
    """Map solar elevations onto the profile bin index."""
    edges = ELEVATION_BINS
    idx = np.digitize(np.asarray(elevation, dtype=float), edges) - 1
    return np.clip(idx, 0, edges.size - 2)


@dataclass
class ChannelFit:
    """Posterior summary for one observed channel (solar or load)."""

    name: str
    mu: float = 0.0
    rho: float = 0.0
    sigma: float = 0.0
    gamma: float = 1.0
    obs_scale: float = 0.0
    obs_floor: float = 0.0
    n_obs: int = 0
    #: Deterministic profile of the latent state indexed by solar elevation.
    profile: list[float] = field(default_factory=list)
    #: Bin edges (degrees) of the elevation index.
    profile_edges: list[float] = field(default_factory=list)
    #: Annual harmonic of the clearness level (cosine coefficient).
    season_cos: float = 0.0
    #: Annual harmonic of the clearness level (sine coefficient).
    season_sin: float = 0.0
    #: Multiplicative calibration of the ensemble spread, fitted on the
    #: validation split so that the nominal P10-P90 band is calibrated.
    spread_factor: float = 1.0
    se: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def profile_at(self, elevation: np.ndarray) -> np.ndarray:
        """Evaluate the profile for the given solar elevations (degrees)."""
        if not self.profile:
            return np.zeros(np.asarray(elevation).size, dtype=float)
        prof = np.asarray(self.profile, dtype=float)
        edges = np.asarray(self.profile_edges, dtype=float) if self.profile_edges else ELEVATION_BINS
        centres = 0.5 * (edges[:-1] + edges[1:])
        if centres.size != prof.size:
            centres = np.arange(prof.size, dtype=float)
        return np.interp(np.asarray(elevation, dtype=float), centres, prof)


@dataclass
class SiteModel:
    """Per-site hierarchical state-space estimates."""

    tenant_id: str
    site_id: str
    solar: ChannelFit
    load: ChannelFit
    load_observed: bool
    final_state: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "site_id": self.site_id,
            "solar": self.solar.to_dict(),
            "load": self.load.to_dict(),
            "load_observed": self.load_observed,
            "final_state": dict(self.final_state),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class FleetPosterior:
    """Fleet-level hierarchical posterior with partial pooling."""

    fleet_id: str
    site_ids: list[str]
    pooled: dict[str, dict[str, float]] = field(default_factory=dict)
    tau: dict[str, float] = field(default_factory=dict)
    shrinkage: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "site_ids": list(self.site_ids),
            "pooled": {k: dict(v) for k, v in self.pooled.items()},
            "tau": dict(self.tau),
            "shrinkage": {k: dict(v) for k, v in self.shrinkage.items()},
        }


# --------------------------------------------------------------------------
# Robust Kalman filter / smoother for the clearness channel
# --------------------------------------------------------------------------


def _kalman_filter(
    u: np.ndarray,
    y: np.ndarray,
    gain: np.ndarray,
    mu: float,
    rho: float,
    sigma: float,
    obs_scale: float,
    obs_floor: float,
) -> dict[str, np.ndarray]:
    """Huberised Kalman filter for the clearness state.

    ``y_t = gain_t * exp(u_t) + e_t`` with Student-t (robust) noise, where
    ``gain_t`` carries the deterministic part of the model (clear-sky level,
    PV bias, diurnal shape and seasonal factor) and ``u_t`` is the latent
    *log* clearness anomaly.  The update is the exact EKF form for the
    multiplicative observation model.  Rows where the gain is zero (night)
    carry no information and are predict-only.
    """
    n = u.size
    a = np.zeros(n)          # predicted state
    m = np.zeros(n)          # filtered state
    p_pred = np.zeros(n)
    p_filt = np.zeros(n)
    innov = np.full(n, np.nan)
    innov_var = np.full(n, np.nan)

    u_prev = mu
    p_prev = sigma**2 / max(1.0 - rho**2, 1e-3)
    for t in range(n):
        a_t = mu + rho * (u_prev - mu)
        p_a = rho**2 * p_prev + sigma**2
        g = float(gain[t])
        r = (obs_scale * abs(g) + obs_floor) ** 2
        if np.isfinite(y[t]) and abs(g) > 1e-9:
            c = g * math.exp(max(min(a_t, 20.0), -20.0))
            e = y[t] - c
            s = c * c * p_a + r
            scale = math.sqrt(s)
            w = _huber_weights(np.array([e]), scale)[0]
            k = c * p_a / s * w
            u_t = a_t + k * e
            p_t = max((1.0 - k * c) * p_a, 1e-12)
            innov[t] = e
            innov_var[t] = s
        else:
            u_t = a_t
            p_t = max(p_a, 1e-12)
        a[t] = a_t
        m[t] = u_t
        p_pred[t] = p_a
        p_filt[t] = p_t
        u_prev = u_t
        p_prev = p_t
    return {
        "filtered": m,
        "filtered_var": p_filt,
        "predicted": a,
        "predicted_var": p_pred,
        "innovation": innov,
        "innovation_var": innov_var,
    }


def _rts_smoother(
    filtered: np.ndarray,
    filtered_var: np.ndarray,
    predicted: np.ndarray,
    predicted_var: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel backward smoother."""
    n = filtered.size
    smoothed = filtered.copy()
    smoothed_var = filtered_var.copy()
    for t in range(n - 2, -1, -1):
        if predicted_var[t + 1] <= 0:
            continue
        gain = filtered_var[t] * rho / predicted_var[t + 1]
        smoothed[t] = filtered[t] + gain * (smoothed[t + 1] - predicted[t + 1])
        smoothed_var[t] = filtered_var[t] * (1.0 - gain**2) + 1e-12
    return smoothed, smoothed_var


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _fit_clearness_channel(
    solar: np.ndarray,
    clear: np.ndarray,
    valid: np.ndarray,
    daylight: np.ndarray,
    elevation: np.ndarray,
    day_of_year: np.ndarray,
    em_iterations: int = 2,
) -> ChannelFit:
    """Fit the clearness state-space model on one site.

    Structure (multiplicative, so that every component stays physically
    interpretable)::

        ratio_t = y_t / clear_t
        u_t     = shape(elev_t) * season(doy_t) * (1 + v_t)
        ratio_t = gamma * u_t + eps_t
        v_t     = mu_v + rho (v_{t-1} - mu_v) + eta_t

    * ``shape`` — the geometry-dependent shape of the clearness, indexed by
      the solar elevation (season invariant, unlike the clock hour).
    * ``season`` — a slow seasonal factor in the clearness level (the
      clear-sky model's accuracy and the atmosphere both drift over the
      season).
    * ``v`` — the AR(1) weather state (the part that is actually
      unpredictable at a 24 h horizon).

    The estimation is deliberately *filter-free* (method of moments on the
    observed ratio) so that it is stable: an EM loop over a Huberised
    Kalman filter tends to inflate the profile and the observation noise
    together until the state collapses.
    """
    obs = valid & daylight & (clear > 1.0)
    if obs.sum() < 24:
        return ChannelFit(
            name="clearness",
            mu=0.0,
            rho=0.0,
            sigma=0.05,
            gamma=1.0,
            obs_scale=0.05,
            obs_floor=50.0,
            n_obs=int(obs.sum()),
            profile=[0.0] * int(ELEVATION_BINS.size - 1),
            profile_edges=[float(v) for v in ELEVATION_BINS],
            se={"mu": 1.0, "rho": 1.0, "sigma": 1.0, "gamma": 1.0, "obs_scale": 1.0},
            diagnostics={"status": "insufficient_data"},
        )

    ratio = np.where(clear > 1.0, solar / np.maximum(clear, 1.0), np.nan)
    weight = np.where(obs, np.maximum(clear, 1.0), 0.0)
    lit_peak = float(np.nanpercentile(clear[obs], 90)) if obs.any() else 1.0
    lit = obs & (clear > 0.2 * max(lit_peak, 1.0))
    # Log clearness: the distribution of the clearness ratio is bounded
    # above (the clear-sky model cannot be beaten by much) and unbounded
    # below (heavy cloud), i.e. strongly left skewed.  Modelling the latent
    # anomaly in log space keeps the innovation symmetric while the implied
    # clearness keeps the correct skew.
    log_ratio = np.where(obs, np.log(np.clip(ratio, 1e-3, None)), np.nan)

    # --- 1. PV bias level, elevation shape, seasonal factor ---------------
    # The published quantiles are quantiles of the forecast distribution, so
    # the level of the model is calibrated on the *median* of the clearness
    # distribution (a mean-calibrated model would place the P50 above the
    # median of a left-skewed distribution).
    log_gamma = float(np.nanmedian(log_ratio[lit])) if lit.any() else 0.0
    gamma = float(np.clip(math.exp(log_gamma), 0.3, 2.0))
    bins = elevation_bin(elevation)
    n_bins = int(ELEVATION_BINS.size - 1)
    centres = 0.5 * (ELEVATION_BINS[:-1] + ELEVATION_BINS[1:])
    profile_abs = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        sel = lit & (bins == b)
        if sel.sum() >= 3:
            profile_abs[b] = float(np.nanmedian(log_ratio[sel]))
    known = np.flatnonzero(np.isfinite(profile_abs) & (profile_abs > -20.0))
    if known.size:
        for b in range(n_bins):
            if profile_abs[b] <= -20.0:
                profile_abs[b] = float(np.interp(b, known, profile_abs[known]))
    profile = np.clip(np.exp(profile_abs - log_gamma), 0.0, 2.5)
    shape = np.interp(elevation, centres, profile)

    # Seasonal factor: annual harmonic of the log residual.  The harmonic is
    # bounded and periodic, so it can be extrapolated past the training
    # window without running away (a linear trend cannot).
    doy = np.asarray(day_of_year, dtype=float)
    theta = 2.0 * math.pi * doy / 365.25
    season_cos = 0.0
    season_sin = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_residual = log_ratio - math.log(max(gamma, 1e-9)) - np.log(np.maximum(shape, 1e-9))
    valid_season = lit & np.isfinite(log_residual)
    if valid_season.sum() > 64:
        w = weight[valid_season]
        y = log_residual[valid_season]
        x1 = np.cos(theta[valid_season])
        x2 = np.sin(theta[valid_season])
        # Weighted least squares with a ridge term (keeps the harmonic small
        # when the training window does not cover a full year).
        a11 = float(np.sum(w * x1 * x1)) + 1e-6
        a12 = float(np.sum(w * x1 * x2))
        a22 = float(np.sum(w * x2 * x2)) + 1e-6
        b1 = float(np.sum(w * x1 * y))
        b2 = float(np.sum(w * x2 * y))
        det = a11 * a22 - a12 * a12
        if abs(det) > 1e-12:
            season_cos = (a22 * b1 - a12 * b2) / det
            season_sin = (a11 * b2 - a12 * b1) / det
    # Centre the harmonic on the training window (mean factor = 1).
    if valid_season.sum() > 16:
        w = weight[valid_season]
        offset = float(
            np.sum(w * (season_cos * np.cos(theta[valid_season]) + season_sin * np.sin(theta[valid_season])))
            / max(float(w.sum()), 1e-9)
        )
        season_cos -= offset
        season_sin -= offset
    season = seasonal_factor(season_cos, season_sin, doy)

    # --- 2. anomaly AR(1) -------------------------------------------------
    anomaly = log_residual - np.log(np.maximum(season, 1e-9))
    # Centre the anomaly so that the multiplicative factor has mean one over
    # the training window (the level itself lives in gamma).
    if lit.any():
        w = weight[lit]
        anomaly = anomaly - float(np.sum(w * anomaly[lit]) / max(float(w.sum()), 1e-9))
    # The AR weights are the clear-sky levels of the *same* rows used for the
    # centring above, so that the AR's own mean and the level calibration
    # (gamma) are estimated against the same population.
    mu_v, rho, resid = _irls_ar1(anomaly, lit, weights=weight)
    mu_v = float(np.clip(mu_v, -0.3, 0.3))
    rho = float(np.clip(rho, 0.0, 0.995))

    # --- 3. split the residual into process and observation noise ---------
    r = resid[lit]
    r = r[np.isfinite(r)]
    var_r = float(np.var(r)) if r.size > 4 else 0.0
    rho_r = 0.0
    if r.size > 8:
        rho_r = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    obs_var = 0.0
    if rho > 1e-6 and rho_r < 0:
        obs_var = max(-rho_r * var_r / rho, 0.0)
    # The AR residual is the process innovation plus the (white) observation
    # noise entering both endpoints:  Var(r) = sigma^2 + s_obs^2 (1 + rho^2).
    sigma_sq = max(var_r - obs_var * (1.0 + rho**2), 1e-10)
    sigma = float(np.clip(math.sqrt(sigma_sq), 1e-4, 1.0))
    obs_scale = float(np.clip(math.sqrt(obs_var), 1e-5, 0.5))

    # --- state estimation with the final parameters ------------------------
    gain = gamma * clear * shape * season
    filt = _kalman_filter(
        u=np.zeros(solar.size),
        y=solar,
        gain=gain,
        mu=mu_v,
        rho=rho,
        sigma=sigma,
        obs_scale=obs_scale,
        obs_floor=50.0,
    )
    smoothed, _ = _rts_smoother(
        filt["filtered"],
        filt["filtered_var"],
        filt["predicted"],
        filt["predicted_var"],
        rho,
    )
    smoothed = np.clip(smoothed, -3.0, 3.0)
    u_hat = np.clip(shape * season * np.exp(smoothed), 0.0, None)

    n_obs = int(obs.sum())
    marginal = sigma / math.sqrt(max(1.0 - rho**2, 1e-6))
    se = {
        "mu": float(np.nanstd(resid[lit]) / math.sqrt(max(lit.sum(), 1))),
        "rho": float(1.0 / math.sqrt(max(lit.sum(), 1))),
        "sigma": float(sigma / math.sqrt(2.0 * max(lit.sum(), 1))),
        "gamma": float(
            np.clip(np.std(ratio[lit]) / math.sqrt(max(lit.sum(), 1)), 1e-4, 0.25)
        ),
        "obs_scale": float(obs_scale / math.sqrt(max(lit.sum(), 1))),
    }
    se["mu"] = float(np.clip(se["mu"], 1e-4, 0.5))
    se["rho"] = float(np.clip(se["rho"], 1e-3, 1.0))
    se["sigma"] = float(np.clip(se["sigma"], 1e-4, 1.0))
    se["gamma"] = float(np.clip(se["gamma"], 1e-4, 0.25))
    se["obs_scale"] = float(np.clip(se["obs_scale"], 1e-5, 0.5))

    diagnostics = {
        "status": "ok",
        "daylight_obs": n_obs,
        "lit_obs": int(lit.sum()),
        "clear_sky_ratio_median": float(np.nanmedian(ratio[obs])),
        "clear_sky_ratio_std": float(np.nanstd(ratio[obs])),
        "ar1_marginal_std_implied": float(marginal),
        "state_std_daylight": float(np.nanstd(anomaly[lit])),
        "profile_peak": float(profile.max()),
        "season_cos": season_cos,
        "season_sin": season_sin,
        "season_amplitude": float(math.hypot(season_cos, season_sin)),
        "ar_residual_autocorr": rho_r,
        "innovation_rmse_w": float(np.sqrt(np.nanmean(filt["innovation"] ** 2))),
        "one_step_rmse_w": float(
            np.sqrt(np.nanmean((solar[obs] - gain[obs] * np.exp(smoothed[obs])) ** 2))
        ),
        "one_step_mae_w": float(
            np.mean(np.abs(solar[obs] - gain[obs] * np.exp(smoothed[obs])))
        ),
        "coverage_hint": "solar P10-P90 band from the filtered predictive spread",
    }
    return ChannelFit(
        name="clearness",
        mu=mu_v,
        rho=rho,
        sigma=sigma,
        gamma=gamma,
        obs_scale=obs_scale,
        obs_floor=50.0,
        n_obs=n_obs,
        profile=[float(v) for v in profile],
        profile_edges=[float(v) for v in ELEVATION_BINS],
        season_cos=season_cos,
        season_sin=season_sin,
        se=se,
        diagnostics=diagnostics,
    )


def _fit_load_channel(
    load: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
) -> ChannelFit:
    """Fit the load-scale channel from measured load (if any)."""
    obs = valid & np.isfinite(load) & np.isfinite(reference) & (reference > 1.0)
    if obs.sum() < 24:
        return ChannelFit(
            name="load_scale",
            mu=1.0,
            rho=0.0,
            sigma=0.05,
            gamma=1.0,
            obs_scale=0.0,
            obs_floor=10.0,
            n_obs=int(obs.sum()),
            se={"mu": 1.0, "rho": 1.0, "sigma": 1.0, "gamma": 1.0, "obs_scale": 1.0},
            diagnostics={"status": "declared_no_measurements"},
        )
    ratio = load[obs] / reference[obs]
    mu = float(np.clip(np.median(ratio), 0.2, 3.0))
    rho = 0.7
    sigma = 0.05
    gamma = 1.0
    obs_scale = 0.05
    filt = _kalman_filter(
        u=np.zeros(load.size),
        y=load,
        clear=reference,
        gamma=gamma,
        mu=mu,
        rho=rho,
        sigma=sigma,
        obs_scale=obs_scale,
        obs_floor=10.0,
    )
    smoothed, _ = _rts_smoother(
        filt["filtered"], filt["filtered_var"], filt["predicted"], filt["predicted_var"], rho
    )
    smoothed = np.clip(smoothed, 0.05, None)
    gamma, se_gamma = _through_origin_regression(reference * smoothed, load)
    gamma = float(np.clip(gamma, 0.2, 3.0))
    pred = gamma * reference * smoothed
    resid = load - pred
    denom = np.maximum(np.abs(pred[obs]), 20.0)
    obs_scale = float(np.clip(_robust_scale(resid[obs] / denom), 0.005, 1.0))
    n_obs = int(obs.sum())
    return ChannelFit(
        name="load_scale",
        mu=mu,
        rho=rho,
        sigma=sigma,
        gamma=gamma,
        obs_scale=obs_scale,
        obs_floor=10.0,
        n_obs=n_obs,
        se={
            "mu": float(min(0.2 / math.sqrt(n_obs), 0.2)),
            "rho": float(1.0 / math.sqrt(n_obs)),
            "sigma": float(sigma / math.sqrt(2.0 * n_obs)),
            "gamma": float(min(se_gamma if np.isfinite(se_gamma) else 0.2, 0.5)),
            "obs_scale": float(obs_scale / math.sqrt(n_obs)),
        },
        diagnostics={"status": "ok", "n_obs": n_obs},
    )


# --------------------------------------------------------------------------
# Hierarchical pooling
# --------------------------------------------------------------------------


def pool_fleet(site_models: Sequence[SiteModel], fleet_id: str = "fleet") -> FleetPosterior:
    """Partial-pool site estimates toward a fleet-level posterior."""
    sites = list(site_models)
    pooled: dict[str, dict[str, float]] = {}
    tau: dict[str, float] = {}
    shrinkage: dict[str, dict[str, float]] = {}

    for channel in ("solar", "load"):
        params = ("mu", "rho", "sigma", "gamma", "obs_scale")
        for param in params:
            values = np.array([getattr(getattr(s, channel), param) for s in sites], dtype=float)
            ses = np.array(
                [max(getattr(getattr(s, channel), "se").get(param, 1.0), 1e-6) for s in sites],
                dtype=float,
            )
            if values.size == 1:
                pooled.setdefault(channel, {})[param] = float(values[0])
                tau.setdefault(channel, {})[param] = 0.0  # type: ignore[index]
                shrinkage.setdefault(channel, {})[param] = 1.0
                continue
            weights = 1.0 / (ses**2)
            wsum = float(weights.sum())
            mean = float(np.sum(weights * values) / wsum)
            se_mean = float(1.0 / math.sqrt(wsum))
            between = float(np.sum(weights * (values - mean) ** 2) / wsum)
            within = float(np.mean(ses**2))
            tau2 = max(between - within, 0.0)
            tau.setdefault(channel, {})[param] = tau2  # type: ignore[index]
            pooled.setdefault(channel, {})[param] = mean
            pooled[channel].setdefault(f"{param}_se", se_mean)
            for site, value, se in zip(sites, values, ses):
                w_site = tau2 / (tau2 + se**2) if (tau2 + se**2) > 0 else 0.0
                shrink = float(np.clip(w_site, 0.0, 1.0))
                shrinkage.setdefault(channel, {}).setdefault(site.site_id, {})[param] = shrink
                shrinkage[channel][site.site_id][f"{param}_pooled"] = float(
                    shrink * value + (1.0 - shrink) * mean
                )
                shrinkage[channel][site.site_id][f"{param}_se_pooled"] = float(
                    math.sqrt(1.0 / (1.0 / max(se**2, 1e-12) + 1.0 / max(tau2, 1e-12)))
                    if tau2 > 0
                    else se
                )
    return FleetPosterior(
        fleet_id=fleet_id,
        site_ids=[s.site_id for s in sites],
        pooled=pooled,
        tau=tau,
        shrinkage=shrinkage,
    )


# --------------------------------------------------------------------------
# The model artifact
# --------------------------------------------------------------------------


@dataclass
class ChariotModel:
    """Versioned CHARIOT model artifact with full lineage."""

    model_name: str
    model_version: str
    tenant_id: str
    fleet_id: str
    dataset_version: str
    dataset_content_hash: str
    sites: dict[str, SiteModel]
    fleet: FleetPosterior
    site_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "tenant_id": self.tenant_id,
            "fleet_id": self.fleet_id,
            "dataset_version": self.dataset_version,
            "dataset_content_hash": self.dataset_content_hash,
            "sites": {k: v.to_dict() for k, v in self.sites.items()},
            "fleet": self.fleet.to_dict(),
            "site_contexts": self.site_contexts,
            "diagnostics": self.diagnostics,
        }


def _load_reference_profile(ctx: Any, timestamps: pd.Series) -> np.ndarray:
    """Declared (or measured) load reference profile in W for each timestamp."""
    profile = (getattr(ctx, "metadata", {}) or {}).get("load_profile_w") or {}
    if profile:
        hours = pd.to_datetime(timestamps, utc=True).dt.hour.to_numpy()
        return np.array([float(profile.get(str(int(h)), profile.get(int(h), 0.0))) for h in hours])
    return np.zeros(len(timestamps))


def load_reference_profile(ctx: Any, timestamps: pd.Series) -> np.ndarray:
    """Public alias of :func:`_load_reference_profile`.

    The offline PyMC reference path reuses the *declared* load profile so
    that both models are compared on the same load assumption.  The DKASC
    export has no load meter, so this is an engineering declaration and not
    something either model learns.
    """
    return _load_reference_profile(ctx, timestamps)


def build_load_ensemble(
    load_profile: np.ndarray,
    *,
    n_scenarios: int,
    t_steps: int,
    declared_load_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the declared load ensemble (W) and the per-scenario site bias.

    Load channel: declared profile x scenario bias x AR(1) deviation.  The
    construction is shared with the offline PyMC reference so that the
    Tribunal comparison is about the solar channel, not about two different
    load assumptions.
    """
    profile = np.nan_to_num(np.asarray(load_profile, dtype=float), nan=0.0)
    load_sigma = float(declared_load_sigma)
    site_bias = np.clip(rng.lognormal(0.0, load_sigma, size=n_scenarios), 0.2, 5.0)
    load_dev = np.zeros((n_scenarios, t_steps), dtype=float)
    load_phi = 0.6
    for k in range(1, t_steps):
        load_dev[:, k] = load_phi * load_dev[:, k - 1] + rng.normal(0.0, 0.05, size=n_scenarios)
    load = np.maximum(profile[None, :] * site_bias[:, None] * (1.0 + load_dev), 0.0)
    return load, site_bias


def _seasonal_validation_mae(
    ch: ChannelFit,
    latitude: float,
    longitude: float,
    frame: pd.DataFrame,
    season_cos: float,
    season_sin: float,
) -> float:
    """One-step-ahead MAE (W) of the solar channel for a given seasonal term."""
    if frame is None or len(frame) == 0:
        return float("inf")
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    solar = solar_series(frame)
    clear = np.nan_to_num(clear_sky_series(frame), nan=0.0)
    elevation = solar_elevation_deg(frame["timestamp_utc"], latitude, longitude)
    doy = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.day_of_year.to_numpy()
    shape = ch.profile_at(elevation)
    season = seasonal_factor(season_cos, season_sin, doy)
    gain = ch.gamma * clear * shape * season
    filt = _kalman_filter(
        u=np.zeros(len(frame)),
        y=solar,
        gain=gain,
        mu=ch.mu,
        rho=ch.rho,
        sigma=ch.sigma,
        obs_scale=ch.obs_scale,
        obs_floor=ch.obs_floor,
    )
    pred = gain * np.exp(filt["predicted"])
    mask = (clear > 1.0) & np.isfinite(solar)
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(solar[mask] - pred[mask])))


def fit(
    train: pd.DataFrame,
    ctx: Any,
    *,
    tenant_id: str,
    site_id: str,
    fleet_id: str = "fleet",
    dataset_version: str = "",
    dataset_content_hash: str = "",
    validation: pd.DataFrame | None = None,
) -> ChariotModel:
    """Fit the CHARIOT world model on the training split of one site."""
    frame = train.sort_values("timestamp_utc").reset_index(drop=True)
    solar = pd.to_numeric(frame["solar_w"], errors="coerce").to_numpy(dtype=float)
    clear = clear_sky_series(frame)
    valid = np.isfinite(solar) & np.isfinite(clear)
    daylight = clear > 1.0

    elevation = solar_elevation_deg(
        frame["timestamp_utc"], float(ctx.latitude or 0.0), float(ctx.longitude or 0.0)
    )
    day_of_year = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.day_of_year.to_numpy()
    solar_channel = _fit_clearness_channel(
        solar, clear, valid, daylight, elevation, day_of_year
    )

    load_observed = "load_w" in frame.columns and frame["load_w"].notna().any()
    if load_observed:
        load_values = pd.to_numeric(frame["load_w"], errors="coerce").to_numpy(dtype=float)
        load_ref = load_values
    else:
        load_values = np.full(len(frame), np.nan)
        load_ref = _load_reference_profile(ctx, frame["timestamp_utc"])
    load_channel = _fit_load_channel(load_values, load_ref, np.isfinite(load_values))

    # ---- model selection: keep the seasonal term only if it helps --------
    if validation is not None and len(validation) and (
        abs(solar_channel.season_cos) > 1e-9 or abs(solar_channel.season_sin) > 1e-9
    ):
        lat = float(ctx.latitude or 0.0)
        lon = float(ctx.longitude or 0.0)
        mae_with = _seasonal_validation_mae(
            solar_channel,
            lat,
            lon,
            validation,
            solar_channel.season_cos,
            solar_channel.season_sin,
        )
        mae_without = _seasonal_validation_mae(solar_channel, lat, lon, validation, 0.0, 0.0)
        selected = "annual_harmonic"
        if mae_without < mae_with:
            solar_channel.season_cos = 0.0
            solar_channel.season_sin = 0.0
            selected = "none"
        solar_channel.diagnostics["seasonality_selection"] = {
            "selected": selected,
            "validation_mae_with_season": round(mae_with, 3),
            "validation_mae_without_season": round(mae_without, 3),
            "fitted_amplitude": solar_channel.diagnostics.get("season_amplitude"),
        }

    site_model = SiteModel(
        tenant_id=tenant_id,
        site_id=site_id,
        solar=solar_channel,
        load=load_channel,
        load_observed=bool(load_observed),
    )

    fleet = pool_fleet([site_model], fleet_id=fleet_id)
    model_version = (
        f"{MODEL_VERSION}_{short_hash(hash_obj(site_model.to_dict()), 10)}"
    )
    return ChariotModel(
        model_name=MODEL_NAME,
        model_version=model_version,
        tenant_id=tenant_id,
        fleet_id=fleet_id,
        dataset_version=dataset_version,
        dataset_content_hash=dataset_content_hash,
        sites={site_id: site_model},
        fleet=fleet,
        site_contexts={site_id: ctx.to_dict() if hasattr(ctx, "to_dict") else dict(ctx)},
        diagnostics={
            "n_train_rows": int(len(frame)),
            "train_time_start": str(frame["timestamp_utc"].min()),
            "train_time_end": str(frame["timestamp_utc"].max()),
            "load_observed": bool(load_observed),
            "clear_sky_rows": int(daylight.sum()),
        },
    )


# --------------------------------------------------------------------------
# Filtering (state estimation for rolling origins)
# --------------------------------------------------------------------------


def filter_state_series(model: ChariotModel, site_id: str, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Run the clearness filter over an observed history.

    Used to condition rolling origins on data available *at* the origin.
    Only rows up to the origin may be passed in — no future leakage.
    """
    site = model.sites[site_id]
    ch = site.solar
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    solar = solar_series(ordered)
    clear = clear_sky_series(ordered)
    gain = observation_gain(ch, model.site_contexts.get(site_id, {}), ordered)
    filt = _kalman_filter(
        u=np.zeros(len(ordered)),
        y=solar,
        gain=gain,
        mu=ch.mu,
        rho=ch.rho,
        sigma=ch.sigma,
        obs_scale=ch.obs_scale,
        obs_floor=ch.obs_floor,
    )
    return filt


def observation_gain(
    ch: ChannelFit, site_ctx: dict[str, Any], frame: pd.DataFrame
) -> np.ndarray:
    """Deterministic part of the solar observation model.

    ``gain_t = gamma * clear_t * shape(elev_t) * season(doy_t)`` — the
    expected solar output when the latent clearness anomaly is zero.
    """
    frame = frame.sort_values("timestamp_utc")
    clear = np.nan_to_num(clear_sky_series(frame), nan=0.0)
    elevation = solar_elevation_deg(
        frame["timestamp_utc"],
        float((site_ctx or {}).get("latitude") or 0.0),
        float((site_ctx or {}).get("longitude") or 0.0),
    )
    doy = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.day_of_year.to_numpy()
    season = seasonal_factor(ch.season_cos, ch.season_sin, doy)
    shape = ch.profile_at(elevation)
    return ch.gamma * clear * shape * season


def one_step_prediction(model: ChariotModel, site_id: str, frame: pd.DataFrame) -> np.ndarray:
    """One-step-ahead solar prediction (W) for every row of ``frame``.

    The prediction uses only information available *before* the row, so it is
    safe to use as a regression target for the residual chamber.
    """
    site = model.sites[site_id]
    ch = site.solar
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    filt = filter_state_series(model, site_id, ordered)
    gain = observation_gain(ch, model.site_contexts.get(site_id, {}), ordered)
    return gain * np.exp(filt["predicted"])


# --------------------------------------------------------------------------
# Scenario simulation
# --------------------------------------------------------------------------


def _sample_channel_params(
    ch: ChannelFit,
    pooled: dict[str, float],
    shrinkage: dict[str, float],
    n: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw hierarchical parameters for ``n`` scenarios."""
    out: dict[str, np.ndarray] = {}
    for param in ("mu", "rho", "sigma", "gamma", "obs_scale"):
        site_mean = getattr(ch, param)
        site_se = max(ch.se.get(param, 1e-3), 1e-6)
        fleet_mean = pooled.get(param, site_mean)
        fleet_se = pooled.get(f"{param}_se", site_se)
        shrink = float(shrinkage.get(param, 1.0))
        mean = shrink * site_mean + (1.0 - shrink) * fleet_mean
        se = float(np.sqrt((shrink * site_se) ** 2 + ((1.0 - shrink) * fleet_se) ** 2))
        se = float(np.clip(se, 1e-5, 0.5))
        out[param] = rng.normal(mean, se, size=n)
    out["rho"] = np.clip(out["rho"], -0.95, 0.98)
    # The spread factor is a calibration of the *process* noise: it widens
    # or narrows the whole ensemble (it is fitted on the validation split).
    out["sigma"] = np.clip(out["sigma"] * float(ch.spread_factor), 1e-4, 2.0)
    out["gamma"] = np.clip(out["gamma"], 0.2, 1.2)
    out["obs_scale"] = np.clip(out["obs_scale"], 0.005, 1.0)
    return out


def simulate(
    model: ChariotModel,
    site_id: str,
    future: pd.DataFrame,
    *,
    n_scenarios: int = 500,
    seed: int = 7,
    state: dict[str, float] | None = None,
    residual: dict[str, np.ndarray] | None = None,
    residual_weight: float = 0.0,
    load_profile: np.ndarray | None = None,
    declared_load_sigma: float = 0.0,
    dt_h: float = 0.25,
) -> dict[str, np.ndarray]:
    """Simulate a scenario ensemble for the future window.

    The latent clearness state is propagated with a heavy-tailed (Student-t,
    df=4) process innovation so that the ensemble tails are conservative
    rather than Gaussian-thin; the physics Tribunal then certifies or rejects
    each path.
    """
    rng = np.random.default_rng(seed)
    site = model.sites[site_id]
    ch = site.solar
    pooled = model.fleet.pooled.get("solar", {})
    shrink = model.fleet.shrinkage.get("solar", {}).get(site_id, {})

    frame = future.sort_values("timestamp_utc").reset_index(drop=True)
    t_steps = len(frame)
    clear = clear_sky_series(frame)
    clear = np.nan_to_num(clear, nan=0.0)
    if load_profile is None:
        site_ctx = model.site_contexts.get(site_id, {}) or {}
        meta = site_ctx.get("metadata", {}) or {}
        load_profile = _load_reference_profile(
            type("Ctx", (), {"metadata": meta})(), frame["timestamp_utc"]
        )
    load_profile = np.nan_to_num(np.asarray(load_profile, dtype=float), nan=0.0)

    params = _sample_channel_params(ch, pooled, shrink, n_scenarios, rng)

    ctx = model.site_contexts.get(site_id, {})
    elevation = solar_elevation_deg(
        frame["timestamp_utc"],
        float(ctx.get("latitude") or 0.0),
        float(ctx.get("longitude") or 0.0),
    )
    profile = ch.profile_at(elevation)
    doy = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.day_of_year.to_numpy()
    base = profile * seasonal_factor(ch.season_cos, ch.season_sin, doy)
    daylight = clear > 1.0

    # The latent weather state v follows the AR(1); the deterministic diurnal
    # profile is added back to obtain the clearness u = profile + v.  The
    # state is reset during the night: the clear-sky ratio is undefined when
    # the sun is below the horizon, so no information survives a night.
    if state is None:
        v0 = ch.mu
        var0 = (ch.sigma**2) / max(1.0 - ch.rho**2, 1e-3)
    else:
        v0 = float(state.get("v_mean", ch.mu))
        var0 = float(state.get("v_var", (ch.sigma**2) / max(1.0 - ch.rho**2, 1e-3)))

    df = 5.0
    scale = math.sqrt(df / 2.0)
    v = np.zeros((n_scenarios, t_steps), dtype=float)
    first = np.zeros(n_scenarios, dtype=bool)
    for k in range(t_steps):
        if k == 0:
            prev = np.full(n_scenarios, v0, dtype=float)
        else:
            prev = v[:, k - 1]
        shock = rng.standard_t(df, size=n_scenarios) / scale
        nxt = params["mu"] + params["rho"] * (prev - params["mu"]) + params["sigma"] * shock
        if k == 0:
            nxt = nxt + rng.normal(0.0, math.sqrt(max(var0, 1e-9)), size=n_scenarios)
        night = not daylight[k]
        v[:, k] = np.where(night, 0.0, nxt)
        first |= daylight[k] & ~first

    u = np.clip(base[None, :] * np.exp(v), 0.0, CLEARNESS_MAX)

    gamma = params["gamma"]
    obs = rng.standard_t(df, size=(n_scenarios, t_steps)) / scale
    obs_scale = params["obs_scale"][:, None]
    solar = gamma[:, None] * clear[None, :] * u
    noise = obs_scale * np.maximum(np.abs(solar), 50.0) * obs
    solar = np.maximum(solar + noise, 0.0)

    # Load channel: declared profile x scenario bias x AR(1) deviation.
    load, site_bias = build_load_ensemble(
        load_profile,
        n_scenarios=n_scenarios,
        t_steps=t_steps,
        declared_load_sigma=declared_load_sigma,
        rng=rng,
    )

    if residual is not None and residual_weight > 0.0:
        w = float(residual_weight)
        if "solar_w" in residual:
            solar = np.maximum(solar + w * residual["solar_w"], 0.0)
        if "load_w" in residual:
            load = np.maximum(load + w * residual["load_w"], 0.0)

    return {
        "solar_w": solar,
        "load_w": load,
        "clearness_u": u,
        "gamma": gamma,
        "load_bias": site_bias,
        "dt_h": np.array([dt_h]),
    }
