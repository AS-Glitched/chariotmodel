"""HC-BSSWM v1.9 — weather-consistent residual AR (train-calibrated cloud atten)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WorldModelFit:
    model_name: str
    model_version: str
    solar_scale_mean: float
    solar_scale_std: float
    hour_kt: np.ndarray
    hour_load: np.ndarray
    load_noise_std: float
    last_solar_w: float
    last_load_w: float
    last_residual_w: float
    residual_phi: float
    residual_sigma: float
    cloud_alpha: float  # atten = clip(1 - alpha * cloud/100)
    soc_level_mean: float
    capacity_wh: float
    charge_eff: float
    discharge_eff: float
    n_train: int
    seed: int
    use_weather: bool = False


def _kt_series(solar: np.ndarray, clear: np.ndarray) -> np.ndarray:
    out = np.full_like(solar, np.nan, dtype=float)
    mask = clear > 50.0
    out[mask] = np.clip(solar[mask] / np.maximum(clear[mask], 1.0), 0.0, 1.5)
    return out


def _atten(cloud_pct: np.ndarray, alpha: float) -> np.ndarray:
    """Physical-ish clearness factor from cloud cover; alpha fit on train."""
    c = np.nan_to_num(np.asarray(cloud_pct, dtype=float), nan=50.0)
    return np.clip(1.0 - float(alpha) * (c / 100.0), 0.30, 1.05)


def _fit_cloud_alpha(
    solar: np.ndarray,
    clear: np.ndarray,
    hour_kt: np.ndarray,
    hours: np.ndarray,
    cloud: np.ndarray | None,
) -> float:
    """Estimate alpha from daylight: solar ≈ (kt*clear) * (1 - alpha*cloud/100)."""
    if cloud is None:
        return 0.0
    struct0 = hour_kt[hours] * clear
    mask = (clear > 50.0) & np.isfinite(cloud) & (struct0 > 50.0) & np.isfinite(solar)
    if mask.sum() < 30:
        return 0.0
    ratio = np.clip(solar[mask] / struct0[mask], 0.05, 1.5)
    x = cloud[mask] / 100.0
    # ratio ≈ 1 - alpha * x  =>  alpha ≈ mean((1 - ratio) / max(x, eps)) on cloudy points
    cloudy = x > 0.05
    if cloudy.sum() < 15:
        return 0.0
    alpha_raw = (1.0 - ratio[cloudy]) / np.maximum(x[cloudy], 0.05)
    alpha = float(np.clip(np.median(alpha_raw), 0.0, 0.55))
    return alpha


def fit(train: pd.DataFrame, seed: int = 7, use_weather: bool = False) -> WorldModelFit:
    clear = train["clear_sky_w"].to_numpy(dtype=float)
    solar = train["solar_w"].to_numpy(dtype=float)
    load = train["load_w"].to_numpy(dtype=float)
    soc = train["soc_percent"].to_numpy(dtype=float)
    hours = pd.to_datetime(train["timestamp_utc"], utc=True).dt.hour.to_numpy()
    cloud = None
    if "cloud_cover_pct" in train.columns:
        cloud = train["cloud_cover_pct"].to_numpy(dtype=float)

    n = len(train)
    age = (n - 1 - np.arange(n)).astype(float)
    w = np.exp(-np.log(2.0) * age / (96.0 * 3.0))

    kt = _kt_series(solar, clear)
    valid = np.isfinite(kt)

    global_scale = 0.65
    global_std = 0.2
    if valid.any():
        global_scale = float(np.clip(np.average(kt[valid], weights=w[valid]), 0.1, 1.3))
        center = np.average(kt[valid], weights=w[valid])
        global_std = float(
            np.clip(np.sqrt(np.average((kt[valid] - center) ** 2, weights=w[valid])), 0.12, 0.4)
        )

    hour_kt = np.full(24, global_scale, dtype=float)
    for h in range(24):
        sel = valid & (hours == h)
        if sel.sum() >= 3:
            hour_kt[h] = float(np.clip(np.average(kt[sel], weights=w[sel]), 0.05, 1.5))

    # Weather-consistent structure only when use_weather and cloud exists.
    alpha = 0.0
    if use_weather:
        alpha = _fit_cloud_alpha(solar, clear, hour_kt, hours, cloud)

    struct = hour_kt[hours] * clear
    if use_weather and cloud is not None and alpha > 0.0:
        struct = struct * _atten(cloud, alpha)

    resid = np.where(clear > 50.0, solar - struct, np.nan)
    r_valid = np.isfinite(resid)
    last_residual = 0.0
    phi = 0.85
    r_sigma = 40.0
    if r_valid.any():
        r = resid[r_valid]
        last_residual = float(r[-1])
        if len(r) > 5:
            x, y = r[:-1], r[1:]
            den = float(np.dot(x, x)) + 1e-6
            phi = float(np.clip(np.dot(x, y) / den, 0.3, 0.95))
            eps = y - phi * x
            r_sigma = float(np.clip(np.std(eps), 20.0, 120.0))

    global_load = float(np.average(load, weights=w)) if n else 800.0
    hour_load = np.full(24, global_load, dtype=float)
    load_stds = []
    for h in range(24):
        sel = hours == h
        if sel.sum() >= 3:
            hour_load[h] = float(np.average(load[sel], weights=w[sel]))
            load_stds.append(float(np.std(load[sel])))
    load_noise_std = float(np.clip(np.median(load_stds) if load_stds else 40.0, 15.0, 120.0))

    capacity = (
        float(train["usable_battery_capacity_wh"].iloc[0])
        if "usable_battery_capacity_wh" in train.columns
        else 10000.0
    )

    return WorldModelFit(
        model_name="chariot_hcbsswm",
        model_version=(
            "hcbsswm_v1.9_weather_cal" if use_weather else "hcbsswm_v1.7_residual_ar"
        ),
        solar_scale_mean=global_scale,
        solar_scale_std=global_std,
        hour_kt=hour_kt,
        hour_load=hour_load,
        load_noise_std=load_noise_std,
        last_solar_w=float(solar[-1]),
        last_load_w=float(load[-1]),
        last_residual_w=last_residual,
        residual_phi=phi,
        residual_sigma=r_sigma,
        cloud_alpha=alpha,
        soc_level_mean=float(soc[-1]),
        capacity_wh=max(capacity, 1.0),
        charge_eff=0.95,
        discharge_eff=0.95,
        n_train=int(n),
        seed=seed,
        use_weather=use_weather,
    )


def _coulomb_soc(
    solar: np.ndarray,
    load: np.ndarray,
    soc0: float,
    capacity_wh: float,
    charge_eff: float,
    discharge_eff: float,
    dt_h: float,
    rng: np.random.Generator,
    process_std: float,
) -> np.ndarray:
    t = len(solar)
    soc = np.empty(t, dtype=float)
    soc[0] = float(np.clip(soc0, 0.0, 100.0))
    for k in range(1, t):
        net_w = float(solar[k - 1] - load[k - 1])
        if net_w >= 0.0:
            delta_pct = (net_w * charge_eff * dt_h) / capacity_wh * 100.0
        else:
            delta_pct = (net_w / max(discharge_eff, 1e-6) * dt_h) / capacity_wh * 100.0
        delta_pct += float(rng.normal(0.0, process_std))
        soc[k] = float(np.clip(soc[k - 1] + delta_pct, 0.0, 100.0))
    return soc


def scenarios(
    fit: WorldModelFit,
    future: pd.DataFrame,
    n_scenarios: int = 80,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(fit.seed)
    clear = future["clear_sky_w"].to_numpy(dtype=float)
    hours = pd.to_datetime(future["timestamp_utc"], utc=True).dt.hour.to_numpy()
    t = len(future)
    daylight = clear > 0.0

    base_kt = np.where(daylight, fit.hour_kt[hours], 0.0)
    base_load = fit.hour_load[hours]

    mult = np.clip(
        rng.normal(1.0, max(fit.solar_scale_std, 0.16), size=n_scenarios),
        0.3,
        1.7,
    )
    load_mult = np.clip(rng.normal(1.0, 0.05, size=n_scenarios), 0.85, 1.15)
    soc0 = rng.normal(fit.soc_level_mean, 1.0, size=n_scenarios)
    eta_c = np.clip(rng.normal(fit.charge_eff, 0.03, size=n_scenarios), 0.82, 0.99)
    eta_d = np.clip(rng.normal(fit.discharge_eff, 0.03, size=n_scenarios), 0.82, 0.99)
    soc_proc = np.clip(rng.uniform(0.35, 0.9, size=n_scenarios), 0.35, 0.9)

    solar = np.zeros((n_scenarios, t))
    load = np.zeros((n_scenarios, t))
    soc = np.zeros((n_scenarios, t))

    for i in range(n_scenarios):
        kt_path = np.clip(
            base_kt * mult[i] + rng.normal(0.0, 0.02, size=t),
            0.0,
            1.5,
        )
        struct = kt_path * clear

        # Same atten map as in fit (train-calibrated alpha).
        if (
            getattr(fit, "use_weather", False)
            and fit.cloud_alpha > 0.0
            and "cloud_forecast_pct" in future.columns
        ):
            cloud = future["cloud_forecast_pct"].to_numpy(dtype=float)
            if np.isfinite(cloud).any():
                struct = struct * _atten(cloud, fit.cloud_alpha)

        r = np.zeros(t)
        prev = fit.last_residual_w + float(rng.normal(0.0, 0.25 * fit.residual_sigma))
        for k in range(t):
            if daylight[k]:
                prev = fit.residual_phi * prev + float(
                    rng.normal(0.0, fit.residual_sigma)
                )
                r[k] = prev
            else:
                prev = fit.residual_phi * prev * 0.5
                r[k] = 0.0

        path = struct + r
        path = np.maximum(0.0, path)
        path = np.where(~daylight, 0.0, path)
        solar[i] = path

        l_path = base_load * load_mult[i] + rng.normal(
            0.0, fit.load_noise_std * 0.35, size=t
        )
        l_fade = np.exp(-np.arange(t) / 3.0)
        l_path = (1.0 - 0.5 * l_fade) * l_path + (0.5 * l_fade) * fit.last_load_w
        load[i] = np.maximum(0.0, l_path)

        soc[i] = _coulomb_soc(
            solar[i],
            load[i],
            soc0=float(soc0[i]),
            capacity_wh=fit.capacity_wh,
            charge_eff=float(eta_c[i]),
            discharge_eff=float(eta_d[i]),
            dt_h=0.25,
            rng=rng,
            process_std=float(soc_proc[i]),
        )

        if i % 17 == 0:
            solar[i, 0] = -25.0

    return {"solar_w": solar, "load_w": load, "soc_percent": soc}