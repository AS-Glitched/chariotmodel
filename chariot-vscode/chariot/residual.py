"""Residual Proposal Chamber — the bounded shadow corrector (PDF §4.8).

    r_t   = y_t - mu_CHARIOT_t
    y_hat = mu_CHARIOT + w * r_hat,      0 <= w <= w_max   (default 0.20)

The chamber is a *shadow*: it never becomes the energy-accounting authority.
Every residual-augmented trajectory is still handed to the Reality Tribunal.

Families: ``ridge`` (closed-form, no extra dependency).  Targets: ``solar_w``
and ``load_w``; the SOC residual is policy-gated and off by default.

Feature discipline
------------------
The master reference lists ``R_t`` (the clear-sky ratio) among the features.
Using the *realised* ``R_t`` would leak the target into the correction
(``r_t`` contains ``y_t``), so the realised ratio is **not** used.  The
chamber uses the *predictable* clear-sky ratio implied by the model
(``mu_CHARIOT / P_clear``) plus strictly leakage-safe lags (observations from
before the forecast origin), calendar terms, calibration factors and regime
tags.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chariot.config import RESIDUAL_POLICY, ResidualPolicy
from chariot.hashing import hash_obj, short_hash


@dataclass
class ResidualArtifact:
    """A versioned residual corrector for one target."""

    residual_version: str
    family: str
    target: str
    feature_names: list[str]
    coef: list[float]
    intercept: float
    mean: list[float]
    scale: list[float]
    alpha: float
    model_version: str
    dataset_version: str
    w_max: float = RESIDUAL_POLICY.w_max
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def load(path: str | Path) -> "ResidualArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ResidualArtifact(**data)

    # -- inference ---------------------------------------------------------
    def design(self, features: np.ndarray) -> np.ndarray:
        z = (features - np.asarray(self.mean)) / np.asarray(self.scale)
        return z

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = self.design(features)
        return z @ np.asarray(self.coef) + self.intercept


def build_features(
    frame: pd.DataFrame,
    *,
    mu_solar: np.ndarray,
    mu_load: np.ndarray,
    clear_sky: np.ndarray,
    expected_ratio: np.ndarray,
    last_solar: float = 0.0,
    last_load: float = 0.0,
    solar_lags: np.ndarray | None = None,
    load_lags: np.ndarray | None = None,
    previous_correction: np.ndarray | None = None,
    calibration: dict[str, float] | None = None,
    site_encoding: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """Build the leakage-safe design matrix.

    ``mu_solar``/``mu_load`` are the CHARIOT means for the same rows.  They
    may be 1-D (shape ``(T,)``) for a single path or 2-D (shape ``(n, T)``)
    for a whole scenario ensemble; in the 2-D case the rows are emitted
    scenario-major (``i * T + t``).  All other inputs must be knowable at
    forecast time.
    """
    ts = pd.to_datetime(frame["timestamp_utc"], utc=True)
    hour = ts.dt.hour.to_numpy(dtype=float)
    doy = ts.dt.day_of_year.to_numpy(dtype=float)
    n = len(frame)

    mu_solar = np.asarray(mu_solar, dtype=float)
    mu_load = np.asarray(mu_load, dtype=float)
    if mu_solar.ndim == 1:
        mu_solar = mu_solar[None, :]
    if mu_load.ndim == 1:
        mu_load = mu_load[None, :]
    n_scen = max(mu_solar.shape[0], 1)
    if mu_solar.shape[1] != n:
        raise ValueError("mu_solar length must match the frame length")

    clear_sky = np.asarray(clear_sky, dtype=float)
    expected_ratio = np.asarray(expected_ratio, dtype=float)
    if expected_ratio.ndim == 1:
        expected_ratio = expected_ratio[None, :]
    expected_ratio_flat = expected_ratio.reshape(-1)

    if solar_lags is None:
        solar_lags = np.concatenate([[last_solar], mu_solar[0, :-1]]) if n else np.zeros(0)
    if load_lags is None:
        load_lags = np.concatenate([[last_load], mu_load[0, :-1]]) if n else np.zeros(0)
    solar_lags = np.asarray(solar_lags, dtype=float)
    load_lags = np.asarray(load_lags, dtype=float)

    calibration = calibration or {}
    cal_arr = np.array(
        [
            float(calibration.get("pv_correction_factor", 1.0)),
            float(calibration.get("capacity_factor", 1.0)),
            float(calibration.get("charge_loss_factor", 0.0)),
            float(calibration.get("discharge_loss_factor", 0.0)),
        ]
    )

    regime_day = (clear_sky > 1.0).astype(float)
    regime_cloud = np.clip(expected_ratio_flat, 0.0, 1.5)

    # The residual's own lag: the correction proposed for the previous step.
    # It is the model's own output (never the realised residual), so it is
    # knowable at forecast time and it cannot leak the target.
    if previous_correction is None:
        prev_corr = np.zeros(n, dtype=float)
    else:
        prev_corr = np.asarray(previous_correction, dtype=float).reshape(-1)
        if prev_corr.size != n:
            prev_corr = np.zeros(n, dtype=float)

    def tiled(values: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray(values, dtype=float), n_scen)

    scenario_mu_solar = mu_solar.reshape(-1)
    scenario_mu_load = mu_load.reshape(-1)

    columns = [
        scenario_mu_solar,
        scenario_mu_load,
        tiled(clear_sky),
        expected_ratio_flat,
        np.log1p(np.maximum(scenario_mu_solar, 0.0)),
        np.log1p(np.maximum(scenario_mu_load, 0.0)),
        tiled(solar_lags),
        tiled(load_lags),
        tiled(np.sin(2 * np.pi * hour / 24.0)),
        tiled(np.cos(2 * np.pi * hour / 24.0)),
        tiled(np.sin(2 * np.pi * doy / 365.25)),
        tiled(np.cos(2 * np.pi * doy / 365.25)),
        tiled(regime_day),
        regime_cloud,
        tiled(prev_corr),
        np.full(n * n_scen, site_encoding),
        np.full(n * n_scen, cal_arr[0]),
        np.full(n * n_scen, cal_arr[1]),
        np.full(n * n_scen, cal_arr[2]),
        np.full(n * n_scen, cal_arr[3]),
    ]
    names = [
        "mu_solar",
        "mu_load",
        "clear_sky",
        "expected_ratio",
        "log_mu_solar",
        "log_mu_load",
        "solar_lag1",
        "load_lag1",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "is_daylight",
        "regime_ratio",
        "previous_correction",
        "site_encoding",
        "cal_pv_correction",
        "cal_capacity_factor",
        "cal_charge_loss",
        "cal_discharge_loss",
    ]
    X = np.column_stack([np.asarray(c, dtype=float) for c in columns])
    X[~np.isfinite(X)] = 0.0
    return X, names


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 10.0,
    feature_names: list[str] | None = None,
    target: str = "solar_w",
    family: str = "ridge",
    model_version: str = "",
    dataset_version: str = "",
    w_max: float = RESIDUAL_POLICY.w_max,
) -> ResidualArtifact:
    """Fit a (standardised) ridge corrector with a closed-form solution."""
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    Xv = X[mask]
    yv = y[mask]
    if Xv.shape[0] < 20:
        raise ValueError("insufficient rows to fit the residual chamber")

    mean = Xv.mean(axis=0)
    scale = Xv.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Z = (Xv - mean) / scale
    ybar = float(yv.mean())
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    b = Z.T @ (yv - ybar)
    coef = np.linalg.solve(A, b)
    intercept = ybar

    pred = Z @ coef + intercept
    rmse = float(np.sqrt(np.mean((pred - yv) ** 2)))
    base_rmse = float(np.sqrt(np.mean((yv - ybar) ** 2)))
    version = f"resid_{target}_{short_hash(hash_obj([list(coef), intercept, alpha]), 10)}"
    return ResidualArtifact(
        residual_version=version,
        family=family,
        target=target,
        feature_names=feature_names or [f"f{i}" for i in range(X.shape[1])],
        coef=[float(c) for c in coef],
        intercept=float(intercept),
        mean=[float(m) for m in mean],
        scale=[float(s) for s in scale],
        alpha=float(alpha),
        model_version=model_version,
        dataset_version=dataset_version,
        w_max=float(w_max),
        diagnostics={
            "n_rows": int(Xv.shape[0]),
            "rmse": rmse,
            "baseline_rmse": base_rmse,
            "r2_vs_mean": float(1.0 - (rmse**2) / max(base_rmse**2, 1e-12)),
        },
    )


def select_weight(
    artifact: ResidualArtifact,
    X: np.ndarray,
    *,
    y_true: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray | None = None,
    candidates: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select the effective residual weight ``w`` on a validation split.

    The weight is chosen from ``[0, w_max]`` to minimise the validation MAE
    of the corrected forecast ``mu + w * r_hat`` against the *realised*
    measurement ``y_true``.  ``mask`` selects the rows of ``X`` that belong
    to the validation split.
    """
    if candidates is None:
        candidates = np.linspace(0.0, artifact.w_max, 21)
    r_hat = artifact.predict(X)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        r_hat = r_hat[mask]
        y_true = np.asarray(y_true)[mask]
        mu = np.asarray(mu)[mask]
    scores = []
    for w in candidates:
        pred = mu + float(w) * r_hat
        mask = np.isfinite(pred) & np.isfinite(y_true)
        mae = float(np.mean(np.abs(pred[mask] - y_true[mask]))) if mask.any() else float("inf")
        scores.append({"w": float(w), "mae": mae})
    best = min(scores, key=lambda s: s["mae"])
    return {"best_w": float(best["w"]), "scores": scores}


def propose_paths(
    artifact: ResidualArtifact,
    X: np.ndarray,
) -> np.ndarray:
    """Residual correction for every scenario row of a path."""
    return artifact.predict(X)


def register(store: Any, artifact: ResidualArtifact, status: str = "draft") -> None:
    """Persist a residual version in the store."""
    store.upsert(
        "residual_versions",
        {
            "residual_version": artifact.residual_version,
            "family": artifact.family,
            "target": artifact.target,
            "model_version": artifact.model_version,
            "dataset_version": artifact.dataset_version,
            "w_max": artifact.w_max,
            "params_json": json.dumps(artifact.to_dict(), sort_keys=True),
            "status": status,
            "created_at_utc": _now(),
        },
    )


def _now() -> str:
    from chariot.store import utc_now

    return utc_now()


def policy_targets(policy: ResidualPolicy = RESIDUAL_POLICY) -> tuple[str, ...]:
    return policy.allowed_targets
