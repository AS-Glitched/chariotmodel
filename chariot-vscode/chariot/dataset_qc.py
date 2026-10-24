"""Dataset quality control for the CHARIOT telemetry slices.

The checks are deliberately *descriptive*, not advisory: they describe what the
data actually contains so that a later stage can be judged against it.  None
of them can change a model, a split or a prediction.

Checks
------
``cadence``
    The timestamps are strictly increasing on a 15-minute grid with no
    duplicates.  A gap in the *measurement* is allowed (the source marks it
    ``is_gap``); a gap in the *clock* is not.
``night``
    The night solar is zero.  "Night" is every row whose clear-sky power is
    below the lit threshold, because that is the model's own definition of
    night — the same threshold the Tribunal and the generator use.
``clear_sky_distinct``
    ``clear_sky_w`` is a *different series* from ``solar_w`` (it is not a
    copy of the measurement).  A copy would silently make the clearness
    ratio identically 1.
``kt_over_one``
    The share of daylight rows whose clearness ratio exceeds 1.05 and 1.20.
    A high share means the clear-sky reference under-binds the measurement
    (the known W2 defect); the number is the acceptance metric for the
    clear-sky upgrade, measured on the *validation* year.
``energy``
    The energy scale: the maximum measured power against the site rating.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

#: A row counts as "lit" above this clear-sky level (watts).  Mirrors the
#: model's own lit mask (``LIT_FRACTION`` × the 90th percentile of the
#: clear-sky level), but with an absolute floor so that the QC works on a
#: single window as well as on a year.
LIT_CLEAR_SKY_W = 1.0

#: Cadence of the DKASC export, minutes.
CADENCE_MIN = 15


def _night_solar_stats(solar: np.ndarray, clear: np.ndarray) -> dict[str, Any]:
    night = clear <= LIT_CLEAR_SKY_W
    if not night.any():
        return {"n_night_rows": 0, "max_night_solar_w": 0.0, "night_nonzero_rows": 0}
    vals = solar[night]
    finite = vals[np.isfinite(vals)]
    return {
        "n_night_rows": int(night.sum()),
        "max_night_solar_w": float(np.max(finite)) if finite.size else 0.0,
        "night_nonzero_rows": int((finite > 0.5).sum()),
    }


def qc_frame(
    frame: pd.DataFrame,
    *,
    rated_pv_w: float = 220_000.0,
    label: str = "",
) -> dict[str, Any]:
    """Run every QC check on one telemetry frame.

    The frame must carry ``timestamp_utc``, ``solar_w`` and ``clear_sky_w``.
    Missing columns are reported as failures rather than raising, so the QC
    can be run on a partially prepared slice.
    """
    out: dict[str, Any] = {"label": label, "rows": int(len(frame))}
    if frame.empty:
        out["status"] = "empty"
        return out

    ts = pd.to_datetime(frame["timestamp_utc"], utc=True)
    solar = pd.to_numeric(frame.get("solar_w", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    clear = (
        pd.to_numeric(frame.get("clear_sky_w", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    )

    # -- cadence ----------------------------------------------------------
    ordered = ts.sort_values()
    diffs = ordered.diff().dropna().dt.total_seconds().to_numpy() / 60.0
    unique_steps = sorted({float(v) for v in diffs}) if diffs.size else []
    non_cadence = int((np.abs(diffs - CADENCE_MIN) > 1e-6).sum()) if diffs.size else 0
    out["cadence"] = {
        "duplicates": int(ts.duplicated().sum()),
        "monotonic": bool(ts.is_monotonic_increasing),
        "non_15min_steps": non_cadence,
        "distinct_step_minutes": unique_steps[:8],
        "cadence_ok": bool(ts.is_monotonic_increasing and non_cadence == 0 and not ts.duplicated().any()),
    }

    # -- night ------------------------------------------------------------
    out["night"] = _night_solar_stats(solar, clear)
    out["night"]["ok"] = bool(
        out["night"]["max_night_solar_w"] <= 1.0 and out["night"]["night_nonzero_rows"] == 0
    )

    # -- clear sky is not the measurement ---------------------------------
    finite = np.isfinite(solar) & np.isfinite(clear)
    if finite.sum() > 1:
        identical = int(np.allclose(solar[finite], clear[finite]))
        max_abs_diff = float(np.max(np.abs(solar[finite] - clear[finite])))
        corr = float(np.corrcoef(solar[finite], clear[finite])[0, 1])
    else:  # pragma: no cover - degenerate input
        identical, max_abs_diff, corr = 0, 0.0, float("nan")
    out["clear_sky_distinct"] = {
        "identical_to_solar": bool(identical),
        "max_abs_difference_w": max_abs_diff,
        "correlation": corr,
        "ok": bool(not identical and max_abs_diff > 1.0),
    }

    # -- clearness ratio --------------------------------------------------
    day = (clear > LIT_CLEAR_SKY_W) & np.isfinite(solar) & np.isfinite(clear)
    ratio = solar[day] / np.maximum(clear[day], 1e-9)
    if ratio.size:
        out["kt_over_one"] = {
            "n_daylight_rows": int(ratio.size),
            "pct_gt_1_05": float(100.0 * np.mean(ratio > 1.05)),
            "pct_gt_1_20": float(100.0 * np.mean(ratio > 1.20)),
            "max_ratio": float(np.max(ratio)),
            "p50_ratio": float(np.median(ratio)),
            "p05_ratio": float(np.percentile(ratio, 5)),
            "p95_ratio": float(np.percentile(ratio, 95)),
        }
    else:  # pragma: no cover - degenerate input
        out["kt_over_one"] = {
            "n_daylight_rows": 0,
            "pct_gt_1_05": float("nan"),
            "pct_gt_1_20": float("nan"),
            "max_ratio": float("nan"),
            "p50_ratio": float("nan"),
            "p05_ratio": float("nan"),
            "p95_ratio": float("nan"),
        }

    # -- energy scale -----------------------------------------------------
    finite_solar = solar[np.isfinite(solar)]
    max_solar = float(np.max(finite_solar)) if finite_solar.size else 0.0
    out["energy"] = {
        "max_solar_w": max_solar,
        "max_solar_x_rated": max_solar / rated_pv_w if rated_pv_w else float("nan"),
        "max_clear_sky_w": float(np.max(clear[np.isfinite(clear)])) if np.isfinite(clear).any() else 0.0,
        "ok": bool(rated_pv_w <= 0 or max_solar <= rated_pv_w * 1.05),
    }

    # -- gaps -------------------------------------------------------------
    out["gaps"] = {
        "is_gap_rows": int(frame["is_gap"].sum())
        if "is_gap" in frame.columns
        else 0,
        "nan_solar_rows": int((~np.isfinite(solar)).sum()),
    }

    out["status"] = (
        "ok"
        if out["cadence"]["cadence_ok"]
        and out["night"]["ok"]
        and out["clear_sky_distinct"]["ok"]
        else "attention"
    )
    return out


def render_qc_markdown(
    qc: dict[str, Any],
    *,
    title: str = "Dataset QC",
    extra: str = "",
) -> str:
    """Render the QC report as markdown."""
    lines: list[str] = [f"# {title}", ""]
    if extra:
        lines.extend([extra, ""])

    lines.append("| slice | rows | status | cadence | night solar max | clear-sky distinct | daylight kt>1.05 | daylight kt>1.20 | max solar |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, block in qc.get("slices", {}).items():
        cad = block["cadence"]
        night = block["night"]
        csd = block["clear_sky_distinct"]
        kt = block["kt_over_one"]
        en = block["energy"]
        lines.append(
            "| {k} | {rows} | {status} | {cad} | {night:.3f} W | {csd} | {p105:.2f} % | {p120:.2f} % | {mx:.0f} W ({mxr:.3f}× rated) |".format(
                k=key,
                rows=block["rows"],
                status=block.get("status", "?"),
                cad="ok" if cad["cadence_ok"] else f"FAIL ({cad['non_15min_steps']} bad steps)",
                night=night["max_night_solar_w"],
                csd="ok" if csd["ok"] else "FAIL (copy of solar_w)",
                p105=kt["pct_gt_1_05"],
                p120=kt["pct_gt_1_20"],
                mx=en["max_solar_w"],
                mxr=en["max_solar_x_rated"],
            )
        )
    lines.append("")

    for key, block in qc.get("slices", {}).items():
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"- rows: **{block['rows']}**, gaps flagged: {block['gaps']['is_gap_rows']}, NaN solar: {block['gaps']['nan_solar_rows']}")
        lines.append(
            f"- cadence: duplicates {block['cadence']['duplicates']}, non-15-min steps {block['cadence']['non_15min_steps']}, monotonic {block['cadence']['monotonic']}"
        )
        lines.append(
            f"- night: {block['night']['n_night_rows']} rows, max solar {block['night']['max_night_solar_w']:.4f} W, non-zero rows {block['night']['night_nonzero_rows']}"
        )
        lines.append(
            f"- clear sky distinct from solar: identical={block['clear_sky_distinct']['identical_to_solar']}, max |Δ| {block['clear_sky_distinct']['max_abs_difference_w']:.1f} W, corr {block['clear_sky_distinct']['correlation']:.4f}"
        )
        kt = block["kt_over_one"]
        lines.append(
            f"- clearness ratio on {kt['n_daylight_rows']} daylight rows: p05 {kt['p05_ratio']:.4f}, p50 {kt['p50_ratio']:.4f}, p95 {kt['p95_ratio']:.4f}, max {kt['max_ratio']:.4f}"
        )
        lines.append(
            f"- daylight solar above clear sky: **{kt['pct_gt_1_05']:.2f} %** above 1.05×, **{kt['pct_gt_1_20']:.2f} %** above 1.20×"
        )
        en = block["energy"]
        lines.append(
            f"- energy: max measured {en['max_solar_w']:.0f} W ({en['max_solar_x_rated']:.3f}× rated), max clear sky {en['max_clear_sky_w']:.0f} W"
        )
        lines.append("")

    if qc.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for note in qc["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def summarise(qc: dict[str, Any]) -> str:
    """One-line status summary used by the stage reports."""
    bits = []
    for key, block in qc.get("slices", {}).items():
        kt = block["kt_over_one"]
        bits.append(
            f"{key}: {block['rows']} rows, kt>1.05 {kt['pct_gt_1_05']:.2f} %, "
            f"kt>1.20 {kt['pct_gt_1_20']:.2f} %, night max {block['night']['max_night_solar_w']:.3f} W"
        )
    return "; ".join(bits) if bits else "no slices"


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")
