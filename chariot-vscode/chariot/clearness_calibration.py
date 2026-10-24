"""The clearness calibration (W3).

The v3 clearness model carries a systematic, *conditional* bias: its expected
clearness is too high in the afternoon and too low around the shoulders, so
the scenario P50 sits above the measurement (about +4.5 kW on the 2022
validation window).  The bias is not a constant offset, so a single global
correction does not help; it depends on the hour of the day and on the solar
elevation.

What this module does
----------------------
It estimates a **bin-wise offset on the log clearness ratio** and applies it
to the generated scenario paths:

``log(observed ratio) - log(modelled P50 ratio)``

is averaged inside (hour, elevation) bins on the **training** split only, the
bin means are shrunk towards the global mean, and the resulting multiplier
``exp(offset)`` is applied to the scenario power before the Tribunal sees it.

The discipline
--------------
* The offsets are fitted on the **training** split, using the same rolling
  origins the operational run uses, so the correction is fitted against the
  *forecast* geometry (a 96-step fan), not against a one-step proxy.
* The shrinkage is selected on the **validation** split.  Nothing is ever
  fitted or selected on the report-only split.
* The clearness cap survives: the calibrated power is re-capped at
  ``CLEARNESS_MAX x clear``, so a calibrated scenario can never claim more
  than the clear-sky reference allows.
* The correction is a *level* correction on a physical ratio.  It never
  touches the Tribunal, the confidence engine or the advisory contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

#: Bumped whenever the recipe changes.
CALIBRATION_VERSION = "chariot_clearness_calibration_v1.0"

#: The candidate shrinkings.  0.0 means "no calibration" and is always
#: available as the fallback candidate.
DEFAULT_SHRINKAGES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Elevation bin edges (degrees) used for the correction.
ELEVATION_EDGES: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0)

#: The log-ratio residuals are clipped to this range before the bin means
#: are taken (a factor of two either way).  It keeps a single extreme hour
#: from moving the correction while preserving the shape of the bias.
RESIDUAL_CLIP = 0.7

#: A bin with fewer rows than this is not used: it falls back to the global
#: correction.
MIN_BIN_COUNT = 20


@dataclass(frozen=True)
class ClearnessCalibration:
    """A bin-wise multiplicative correction on the clearness ratio."""

    version: str = CALIBRATION_VERSION
    hour_edges: tuple[int, ...] = tuple(range(25))
    elevation_edges: tuple[float, ...] = ELEVATION_EDGES
    #: (n_hours, n_elevations) offsets on the log ratio.
    offsets: tuple[tuple[float, ...], ...] = ()
    global_offset: float = 0.0
    shrinkage: float = 0.0
    n_rows: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.shrinkage > 0.0 and bool(self.offsets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "hour_edges": list(self.hour_edges),
            "elevation_edges": list(self.elevation_edges),
            "offsets": [list(row) for row in self.offsets],
            "global_offset": self.global_offset,
            "shrinkage": self.shrinkage,
            "n_rows": self.n_rows,
            "active": self.active,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClearnessCalibration":
        return cls(
            version=payload.get("version", CALIBRATION_VERSION),
            hour_edges=tuple(payload.get("hour_edges", tuple(range(25)))),
            elevation_edges=tuple(payload.get("elevation_edges", ELEVATION_EDGES)),
            offsets=tuple(tuple(float(v) for v in row) for row in payload.get("offsets", ())),
            global_offset=float(payload.get("global_offset", 0.0)),
            shrinkage=float(payload.get("shrinkage", 0.0)),
            n_rows=int(payload.get("n_rows", 0)),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


def _bin_index(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Index of each value in a set of bin edges (clipped into range)."""
    edges_arr = np.asarray(edges, dtype=float)
    idx = np.digitize(values, edges_arr[1:-1], right=False)
    return np.clip(idx, 0, len(edges_arr) - 2).astype(int)


def calibration_multiplier(
    calibration: ClearnessCalibration,
    frame: pd.DataFrame,
    elevation: np.ndarray,
) -> np.ndarray:
    """The per-row multiplicative correction ``exp(offset)``."""
    if not calibration.active:
        return np.ones(len(frame), dtype=float)
    hours = pd.DatetimeIndex(frame["timestamp_utc"]).hour.to_numpy()
    offsets = np.asarray(calibration.offsets, dtype=float)
    h_idx = np.clip(hours, 0, offsets.shape[0] - 1)
    e_idx = _bin_index(np.asarray(elevation, dtype=float), calibration.elevation_edges)
    e_idx = np.clip(e_idx, 0, offsets.shape[1] - 1)
    return np.exp(offsets[h_idx, e_idx])


def apply_calibration_to_paths(
    solar_paths: np.ndarray,
    multiplier: np.ndarray,
    clear_sky_w: np.ndarray,
    *,
    clearness_max: float,
) -> np.ndarray:
    """Apply the correction to a scenario ensemble and re-cap the clearness.

    ``solar_paths`` has shape (n_scenarios, n_steps).  The correction is a
    level shift on the clearness ratio, so the ratio is re-capped at
    ``clearness_max`` afterwards: a calibrated scenario can never claim more
    than the clear-sky reference allows.
    """
    mult = np.asarray(multiplier, dtype=float)[None, :]
    corrected = np.asarray(solar_paths, dtype=float) * mult
    clear = np.asarray(clear_sky_w, dtype=float)[None, :]
    cap = clearness_max * clear
    corrected = np.where(clear > 1.0, np.minimum(corrected, cap), corrected)
    return np.maximum(corrected, 0.0)


def _window_residuals(
    model: Any,
    site_id: str,
    ctx: Any,
    windows: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    *,
    n_scenarios: int,
    seed: int,
    state_fn: Any,
    elevation_fn: Any,
) -> pd.DataFrame:
    """Per-row log-ratio residuals of the scenario P50 over the windows."""
    from chariot.model import clear_sky_series, simulate, solar_series

    frames: list[pd.DataFrame] = []
    for idx, (origin, future, actual) in enumerate(windows):
        state = state_fn(model, actual, future, ctx)
        sim = simulate(
            model,
            site_id,
            future,
            n_scenarios=n_scenarios,
            seed=seed + idx,
            state=state,
        )
        p50 = np.median(sim["solar_w"], axis=0)
        clear = np.nan_to_num(clear_sky_series(future), nan=0.0)
        truth = solar_series(future)
        modelled_ratio = p50 / np.maximum(clear, 1.0)
        observed_ratio = truth / np.maximum(clear, 1.0)
        # Rows where the plant delivered (almost) nothing are *not*
        # clearness observations -- they are outages or shutdowns, and they
        # would otherwise dominate a ratio fit with an infinite residual.
        mask = (
            (clear > 1000.0)
            & np.isfinite(truth)
            & np.isfinite(p50)
            & (truth > 0.02 * clear)
        )
        if not mask.any():
            continue
        elevation = elevation_fn(future, ctx)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp_utc": future["timestamp_utc"].to_numpy()[mask],
                    "hour": pd.DatetimeIndex(future["timestamp_utc"]).hour.to_numpy()[mask],
                    "elevation": np.asarray(elevation, dtype=float)[mask],
                    "log_residual": np.log(
                        np.maximum(observed_ratio[mask], 1e-6)
                        / np.maximum(modelled_ratio[mask], 1e-6)
                    ),
                    "error_w": (p50 - truth)[mask],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "hour", "elevation", "log_residual", "error_w"])
    return pd.concat(frames, ignore_index=True)


def fit_clearness_calibration(
    model: Any,
    site_id: str,
    ctx: Any,
    windows: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    *,
    n_scenarios: int = 120,
    seed: int = 7,
    shrinkage: float = 1.0,
    state_fn: Any = None,
    elevation_fn: Any = None,
) -> tuple[ClearnessCalibration, pd.DataFrame]:
    """Fit the bin-wise log-ratio offsets on the training windows."""
    if state_fn is None or elevation_fn is None:
        raise ValueError("fit_clearness_calibration needs state_fn and elevation_fn")
    residuals = _window_residuals(
        model,
        site_id,
        ctx,
        windows,
        n_scenarios=n_scenarios,
        seed=seed,
        state_fn=state_fn,
        elevation_fn=elevation_fn,
    )
    if residuals.empty:
        return ClearnessCalibration(shrinkage=0.0), residuals

    n_hours = 24
    n_elev = len(ELEVATION_EDGES) - 1
    # The estimator is a *trimmed mean*: the mean carries the shape of the
    # bias (which is what the correction has to remove), but the residuals
    # are clipped so that a single extreme hour cannot move it.
    log_res = np.clip(
        residuals["log_residual"].to_numpy(dtype=float), -RESIDUAL_CLIP, RESIDUAL_CLIP
    )
    global_offset = float(np.mean(log_res))
    h_idx = np.clip(residuals["hour"].to_numpy(), 0, n_hours - 1)
    e_idx = _bin_index(residuals["elevation"].to_numpy(), ELEVATION_EDGES)

    bin_mean = np.full((n_hours, n_elev), global_offset, dtype=float)
    counts = np.zeros((n_hours, n_elev), dtype=float)
    for h in range(n_hours):
        for e in range(n_elev):
            sel = log_res[(h_idx == h) & (e_idx == e)]
            if sel.size:
                counts[h, e] = sel.size
                bin_mean[h, e] = float(np.mean(sel))

    # Hierarchical shrinkage towards the global offset, with the bin count
    # as the weight: an empty or thin bin falls back to the global
    # correction.  A bin below MIN_BIN_COUNT is not used at all -- a handful
    # of rows is not evidence about a bin.
    k = 50.0
    shrunk = (bin_mean * counts + k * global_offset) / (counts + k)
    shrunk = np.where(counts >= MIN_BIN_COUNT, shrunk, global_offset)
    offsets = global_offset + (shrunk - global_offset) * float(shrinkage)

    calibration = ClearnessCalibration(
        offsets=tuple(tuple(float(v) for v in row) for row in offsets),
        global_offset=global_offset,
        shrinkage=float(shrinkage),
        n_rows=int(len(residuals)),
        diagnostics={
            "global_log_offset": global_offset,
            "global_log_offset_mean": float(np.mean(log_res)),
            "residual_clip": RESIDUAL_CLIP,
            "global_bias_w": float(residuals["error_w"].mean()),
            "n_bins": int((counts > 0).sum()),
            "min_bin_count": int(counts[counts > 0].min()) if (counts > 0).any() else 0,
            "shrinkage_prior_rows": k,
            "min_bin_count_required": MIN_BIN_COUNT,
        },
    )
    return calibration, residuals


def apply_shrinkage(calibration: ClearnessCalibration, shrinkage: float) -> ClearnessCalibration:
    """Return the same calibration at a different shrinkage."""
    if not calibration.offsets:
        return ClearnessCalibration(shrinkage=0.0, n_rows=calibration.n_rows)
    offsets = np.asarray(calibration.offsets, dtype=float)
    global_offset = calibration.global_offset
    shrunk = global_offset + (offsets - global_offset) * float(shrinkage)
    return ClearnessCalibration(
        version=calibration.version,
        hour_edges=calibration.hour_edges,
        elevation_edges=calibration.elevation_edges,
        offsets=tuple(tuple(float(v) for v in row) for row in shrunk),
        global_offset=global_offset,
        shrinkage=float(shrinkage),
        n_rows=calibration.n_rows,
        diagnostics=dict(calibration.diagnostics),
    )


def score_calibrations(
    model: Any,
    site_id: str,
    ctx: Any,
    windows: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    candidates: Sequence[ClearnessCalibration],
    *,
    n_scenarios: int,
    seed: int,
    state_fn: Any,
    elevation_fn: Any,
) -> list[dict[str, float]]:
    """Score every candidate calibration on the validation windows.

    The scenarios are simulated **once** per window and every candidate is
    scored against the same ensemble, so the selection is cheap and exactly
    comparable.  The score is the scenario P50 error against the truth: MAE,
    RMSE and the bias, on the same windows the operational run uses.
    """
    from chariot.model import clear_sky_series, simulate, solar_series

    errors: list[list[np.ndarray]] = [[] for _ in candidates]
    for idx, (origin, future, actual) in enumerate(windows):
        state = state_fn(model, actual, future, ctx)
        sim = simulate(
            model,
            site_id,
            future,
            n_scenarios=n_scenarios,
            seed=seed + idx,
            state=state,
        )
        p50 = np.median(sim["solar_w"], axis=0)
        clear = np.nan_to_num(clear_sky_series(future), nan=0.0)
        truth = solar_series(future)
        # Rows where the plant delivered (almost) nothing are *not*
        # clearness observations -- they are outages or shutdowns, and they
        # would otherwise dominate a ratio fit with an infinite residual.
        mask = (
            (clear > 1000.0)
            & np.isfinite(truth)
            & np.isfinite(p50)
            & (truth > 0.02 * clear)
        )
        if not mask.any():
            continue
        elevation = elevation_fn(future, ctx)
        for pos, candidate in enumerate(candidates):
            if candidate.active:
                mult = calibration_multiplier(candidate, future, elevation)
                corrected = apply_calibration_to_paths(
                    p50[None, :], mult, clear, clearness_max=_clearness_max()
                )[0]
            else:
                corrected = p50
            errors[pos].append(corrected[mask] - truth[mask])

    out: list[dict[str, float]] = []
    for candidate, err_list in zip(candidates, errors):
        if not err_list:
            out.append({"mae": math.inf, "rmse": math.inf, "bias": math.inf, "n": 0})
            continue
        err = np.concatenate(err_list)
        out.append(
            {
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "n": int(err.size),
            }
        )
    return out


def _clearness_max() -> float:
    from chariot.model import CLEARNESS_MAX

    return float(CLEARNESS_MAX)
