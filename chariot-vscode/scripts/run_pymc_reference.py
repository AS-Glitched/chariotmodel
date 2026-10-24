#!/usr/bin/env python3
"""Compare the operational NumPy clearness model with the offline PyMC reference.

Both models are fitted on the **same** training split of the **same** Alice
Springs 2022 training extract, both are rolled forward over the **same**
rolling-origin windows on the validation and test splits, and both are
certified by the **same** Tribunal.  The comparison is therefore about the
inference, not about the plumbing.

Rules honoured by this script
-----------------------------
* **The test split is report-only.**  Every choice (the seasonal harmonic,
  the ensemble spread factor, the number of chains, the draws) is made on the
  training or validation split.  Nothing is fitted on the test split and no
  test target ever reaches a fit.
* **The residual chamber is OFF** for this comparison, for both models, so
  the numbers describe the clearness model and not the residual corrector.
  The v3 operational run keeps its own residual policy; nothing here changes
  it.
* **The Tribunal is not loosened.**  Reference scenarios are generated
  through the documented adapter seam in :func:`chariot.genesis.generate_batch`
  and then go through exactly the same physics gate, the same content hash
  and the same lineage as operational scenarios.
* **The ops model stays the default.**  This script never writes an active
  model version and never changes the operational configuration.

Usage::

    python scripts/run_pymc_reference.py \
        --source /path/to/dka_alice_springs_solar_only_15min_2022.csv \
        --out runs/pymc_reference \
        --draws 500 --tune 500 --chains 2

Outputs (in ``--out``):

* ``comparison.json`` — the full machine-readable comparison;
* ``summary.md`` — the side-by-side table (small, safe to commit);
* ``stdout`` — the same table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chariot.admissibility import evaluate_frame  # noqa: E402
from chariot.advisory import build_all  # noqa: E402
from chariot.calibration import approve_version, create_version  # noqa: E402
from chariot.calibration import resolve as resolve_calibration  # noqa: E402
from chariot.capsule import seal_capsule  # noqa: E402
from chariot.config import (  # noqa: E402
    ADMISSIBILITY_POLICY,
    CONFIDENCE_POLICY,
    TRIBUNAL_POLICY,
)
from chariot.confidence import evaluate as evaluate_confidence  # noqa: E402
from chariot.genesis import generate_batch  # noqa: E402
from chariot.metrics import baseline_scoreboard, point_metrics, score_window  # noqa: E402
from chariot.model import clear_sky_series  # noqa: E402
from chariot.model import fit as fit_model  # noqa: E402
from chariot.model import solar_series  # noqa: E402
from chariot.pipeline import (  # noqa: E402
    COVERAGE_TARGET,
    SPREAD_FACTOR_BOUNDS,
    _calibrate_spread,
    _dump_json,
    _origin_state,
    alice_springs_site,
)
from chariot.river_flow import (  # noqa: E402
    build_horizontal_slice,
    build_vertical_slice,
    issue_replay_certificate,
    verify_replay,
)
from chariot.site_master import resolve_context, upsert_site  # noqa: E402
from chariot.store import Store  # noqa: E402
from chariot.training_extract import build_training_extract, load_dataset  # noqa: E402
from chariot.tribunal import validate_batch  # noqa: E402

DEFAULT_SOURCE = "/tmp/arena-coding-uploads/dka_alice_springs_solar_only_15min_2022.csv"
DEFAULT_OUT = ROOT / "runs" / "pymc_reference"
DEFAULT_STORE = ROOT / "var" / "pymc_reference.db"


# --------------------------------------------------------------------------
# The spine up to the training extract
# --------------------------------------------------------------------------


def build_dataset(
    *,
    store: Store,
    source: Path,
    start: str,
    end: str,
) -> tuple[Any, dict[str, float], pd.DataFrame, dict[str, Any]]:
    """Run the spine down to the training extract and return the dataset.

    The exact same sequence of layers the operational pipeline uses, so the
    ops model and the reference model are fitted on byte-identical rows.
    """
    site = alice_springs_site()
    upsert_site(
        store,
        site["tenant_id"],
        site["site_id"],
        **{k: v for k, v in site.items() if k not in ("tenant_id", "site_id")},
    )
    ctx = resolve_context(store, site["tenant_id"], site["site_id"])

    raw = pd.read_csv(source)
    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    mask = (raw["timestamp_utc"] >= pd.Timestamp(start, tz="UTC")) & (
        raw["timestamp_utc"] < pd.Timestamp(end, tz="UTC")
    )
    raw = raw.loc[mask].sort_values("timestamp_utc").reset_index(drop=True)

    decisions = evaluate_frame(
        raw, ctx=ctx, policy=ADMISSIBILITY_POLICY, feature_set="solar_only_v1"
    )
    admitted_mask = decisions["decision"].isin(
        ["ADMISSIBLE", "ADMISSIBLE_WITH_WARNINGS"]
    ).to_numpy()
    admitted = raw.loc[admitted_mask].copy().reset_index(drop=True)
    admitted["decision"] = decisions.loc[admitted_mask, "decision"].to_numpy()
    admitted["quality_flags"] = decisions.loc[admitted_mask, "soft_warnings"].apply(
        list
    ).to_numpy()
    admitted["input_fingerprint"] = decisions.loc[admitted_mask, "input_fingerprint"].to_numpy()

    vertical = build_vertical_slice(
        store, admitted, tenant_id=site["tenant_id"], site_id=site["site_id"]
    )
    horizontal = {
        param: build_horizontal_slice(
            store, admitted, param, tenant_id=site["tenant_id"], site_id=site["site_id"]
        )
        for param in ("solar_w", "clear_sky_w")
    }
    slice_ids = [vertical["slice_id"]] + [h["slice_id"] for h in horizontal.values()]
    slice_hashes = [vertical["content_hash"]] + [h["content_hash"] for h in horizontal.values()]
    replay = issue_replay_certificate(
        store, slice_ids, tenant_id=site["tenant_id"], site_id=site["site_id"]
    )
    verify_replay(store, replay["replay_id"])

    calib_version = create_version(
        store,
        scope_type="site",
        scope_id=site["site_id"],
        capacity_factor=1.0,
        charge_loss_factor=float(ctx.metadata.get("charge_loss_factor", 0.0)),
        discharge_loss_factor=float(ctx.metadata.get("discharge_loss_factor", 0.0)),
        pv_correction_factor=1.0,
        method="engineering_declaration",
        notes="site-level operating losses and PV correction for the DKASC reference site",
        tenant_id=site["tenant_id"],
        site_id=site["site_id"],
    )
    approve_version(store, calib_version, approved_by="pymc_reference")
    factors = resolve_calibration(
        store,
        tenant_id=site["tenant_id"],
        site_id=site["site_id"],
        fleet_id=ctx.fleet_id,
    )

    capsule = seal_capsule(
        store,
        admitted,
        tenant_id=site["tenant_id"],
        site_id=site["site_id"],
        calibration_version=factors.calibration_version,
        admissibility_versions=[ADMISSIBILITY_POLICY.rule_set_version],
        slice_ids=slice_ids,
        slice_hashes=slice_hashes,
        ctx=ctx,
        created_by="pymc_reference",
    )
    manifest = build_training_extract(
        store,
        admitted,
        tenant_id=site["tenant_id"],
        site_id=site["site_id"],
        capsule_ids=[capsule["capsule_id"]],
        calibration_version=factors.calibration_version,
        calibration_factors=factors.to_dict(),
        admissibility_versions=[ADMISSIBILITY_POLICY.rule_set_version],
        slice_ids=slice_ids,
        ctx=ctx,
        train_frac=0.60,
        validate_frac=0.20,
        test_frac=0.20,
        created_by="pymc_reference",
    )
    dataset = load_dataset(store, manifest["dataset_version"])
    return ctx, factors.to_dict(), dataset, manifest


# --------------------------------------------------------------------------
# Window helpers
# --------------------------------------------------------------------------


def plan_windows(
    frame: pd.DataFrame, horizon_steps: int, max_windows: int | None
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Tile a split into rolling-origin windows (same rule as the pipeline)."""
    windows: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    n = len(frame)
    step = max(1, horizon_steps)
    start = 0
    while start < n:
        end = min(start + step, n)
        future = frame.iloc[start:end].reset_index(drop=True)
        actual = frame.iloc[start:end].reset_index(drop=True)
        label = str(future["timestamp_utc"].iloc[0])
        windows.append((label, future, actual))
        start = end
    if max_windows is not None:
        windows = windows[:max_windows]
    return windows


def _band_metrics(actual: pd.DataFrame, solar: np.ndarray) -> dict[str, Any]:
    """MAE / RMSE / bias / P10-P90 coverage of a raw scenario ensemble."""
    truth = solar_series(actual).astype(float)
    n = min(truth.size, solar.shape[1])
    truth = truth[:n]
    solar = np.asarray(solar, dtype=float)[:, :n]
    p10 = np.percentile(solar, 10, axis=0)
    p50 = np.percentile(solar, 50, axis=0)
    p90 = np.percentile(solar, 90, axis=0)
    mask = np.isfinite(truth)
    out = point_metrics(truth[mask], p50[mask])
    inside = (truth[mask] >= p10[mask]) & (truth[mask] <= p90[mask])
    out["coverage"] = float(np.mean(inside)) if inside.size else float("nan")
    out["coverage_p10_p90"] = out["coverage"]
    out["n_steps"] = int(mask.sum())
    return out


def _calibrate_pymc_spread(
    pymc_fit: Any,
    validate: pd.DataFrame,
    ctx: Any,
    *,
    horizon_steps: int,
    target_coverage: float,
    n_windows: int,
    n_scenarios: int,
    seed: int,
) -> dict[str, Any]:
    """Validation-fitted spread calibration for the reference ensemble.

    The same bisection the operational pipeline uses, with the same bounds,
    so the two bands are compared on equal terms.
    """
    from chariot.pymc_clearness import build_reference_model

    if validate is None or len(validate) < horizon_steps * 2:
        return {"calibrated": False, "reason": "insufficient_validation_rows"}
    history = validate.sort_values("timestamp_utc").reset_index(drop=True)
    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    stride = max(1, len(history) // max(n_windows, 1))
    for i in range(n_windows):
        start = i * stride
        future = history.iloc[start : start + horizon_steps]
        if len(future) < horizon_steps:
            break
        windows.append((future.reset_index(drop=True), future.reset_index(drop=True)))
    if not windows:
        return {"calibrated": False, "reason": "no_validation_windows"}

    ref_model = build_reference_model(pymc_fit, ctx)

    def coverage_for(factor: float) -> tuple[float, int]:
        hits = 0
        total = 0
        for idx, (future, actual) in enumerate(windows):
            state = _origin_state(ref_model, history, future, ctx)
            paths = pymc_fit.simulate_paths(
                future,
                ctx,
                n_paths=n_scenarios,
                seed=seed + idx,
                spread_factor=factor,
                origin_state=state,
            )
            solar = paths["solar_w"]
            p10 = np.percentile(solar, 10, axis=0)
            p90 = np.percentile(solar, 90, axis=0)
            truth = solar_series(actual)
            mask = np.isfinite(truth)
            if mask.any():
                hits += int(np.sum((truth[mask] >= p10[mask]) & (truth[mask] <= p90[mask])))
                total += int(mask.sum())
        return (hits / total if total else 0.0), total

    lo, hi = SPREAD_FACTOR_BOUNDS[0], SPREAD_FACTOR_BOUNDS[1]
    best_factor, best_cov = 1.0, 0.0
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        cov, _n = coverage_for(mid)
        best_factor, best_cov = mid, cov
        if cov < target_coverage:
            lo = mid
        else:
            hi = mid
    return {
        "calibrated": True,
        "target_coverage": target_coverage,
        "achieved_validation_coverage": round(float(best_cov), 4),
        "spread_factor": round(float(np.clip(best_factor, *SPREAD_FACTOR_BOUNDS)), 4),
        "n_windows": len(windows),
        "n_scenarios": n_scenarios,
    }


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    from chariot.pymc_clearness import (
        PymcReferenceConfig,
        build_reference_model,
        fit_pymc_clearness,
        pymc_available,
    )

    if not pymc_available():
        print(
            "PyMC is not importable — the reference path cannot run.\n"
            "Install it with:  pip install 'pymc>=5' 'pytensor>=2.20' numba",
            file=sys.stderr,
        )
        return {"status": "pymc_unavailable"}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.store)

    print("[1/5] building the training extract (same spine as the ops run)...")
    ctx, factors, dataset, manifest = build_dataset(
        store=store, source=Path(args.source), start=args.start, end=args.end
    )
    train = dataset.loc[(dataset["split"] == "train") & dataset["is_trainable"]].reset_index(
        drop=True
    )
    validate = dataset.loc[
        (dataset["split"] == "validate") & dataset["is_trainable"]
    ].reset_index(drop=True)
    test = dataset.loc[(dataset["split"] == "test") & dataset["is_trainable"]].reset_index(
        drop=True
    )
    print(
        f"      rows: train={len(train)} validate={len(validate)} test={len(test)} "
        f"| dataset={manifest['dataset_version']}"
    )

    # ------------------------------------------------------------ ops model
    print("[2/5] fitting the operational NumPy model (train split)...")
    ops_model = fit_model(
        train,
        ctx,
        tenant_id=ctx.tenant_id,
        site_id=ctx.site_id,
        fleet_id=ctx.fleet_id or "fleet",
        dataset_version=manifest["dataset_version"],
        dataset_content_hash=manifest["content_hash"],
        validation=validate,
    )
    ops_spread = _calibrate_spread(
        ops_model,
        validate,
        ctx,
        ctx.site_id,
        horizon_steps=args.horizon,
        target_coverage=COVERAGE_TARGET,
        n_windows=args.spread_windows,
        n_scenarios=args.spread_scenarios,
        seed=args.seed,
    )
    print(f"      spread calibration: {ops_spread}")

    # --------------------------------------------------------- PyMC reference
    print("[3/5] fitting the offline PyMC reference (train split)...")
    config = PymcReferenceConfig(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        backend=args.backend,
        season=args.season,
        random_seed=args.seed,
    )
    pymc_fit = fit_pymc_clearness(
        train,
        ctx,
        tenant_id=ctx.tenant_id,
        site_id=ctx.site_id,
        fleet_id=ctx.fleet_id or "fleet",
        dataset_version=manifest["dataset_version"],
        dataset_content_hash=manifest["content_hash"],
        validation=validate,
        config=config,
    )
    print(
        f"      backend={pymc_fit.backend} wall={pymc_fit.diagnostics.get('wall_time_s')}s "
        f"posterior={pymc_fit.diagnostics.get('posterior_mean')}"
    )

    # The same validation-fitted spread calibration is applied to the
    # reference ensemble, so the two bands are compared on equal terms.
    pymc_spread = _calibrate_pymc_spread(
        pymc_fit,
        validate,
        ctx,
        horizon_steps=args.horizon,
        target_coverage=COVERAGE_TARGET,
        n_windows=args.spread_windows,
        n_scenarios=args.spread_scenarios,
        seed=args.seed,
    )
    print(f"      reference spread calibration: {pymc_spread}")

    ref_model = build_reference_model(pymc_fit, ctx)

    # -------------------------------------------------------------- windows
    print("[4/5] rolling the two models over the validation and test windows...")
    history = dataset.sort_values("timestamp_utc").reset_index(drop=True)
    results: dict[str, Any] = {"ops": {}, "pymc": {}}
    per_window: list[dict[str, Any]] = []

    for split_name, split_frame in (("validate", validate), ("test", test)):
        windows = plan_windows(split_frame, args.horizon, args.windows)
        for model_key, model, spread in (
            ("ops", ops_model, float(ops_model.sites[ctx.site_id].solar.spread_factor)),
            ("pymc", ref_model, float(pymc_spread.get("spread_factor", 1.0))),
        ):
            rows: list[dict[str, Any]] = []
            for idx, (origin, future, actual) in enumerate(windows):
                state = _origin_state(model, history, future, ctx)
                paths = None
                if model_key == "pymc":
                    paths = pymc_fit.simulate_paths(
                        future,
                        ctx,
                        n_paths=args.scenarios,
                        seed=args.seed + idx,
                        spread_factor=spread,
                        origin_state=state,
                    )
                batch = generate_batch(
                    model,
                    ctx.site_id,
                    future,
                    tenant_id=ctx.tenant_id,
                    n_scenarios=args.scenarios,
                    seed=args.seed + idx,
                    ctx=ctx,
                    calibration=factors,
                    residual_version=None,
                    residual_weight=0.0,
                    residual_artifact=None,
                    state=state,
                    dt_h=0.25,
                    paths=paths,
                )
                tribunal = validate_batch(
                    batch, ctx=ctx, calibration=factors, policy=TRIBUNAL_POLICY, store=store
                )
                report = evaluate_confidence(
                    batch,
                    tribunal.retained_idx,
                    survival_rate=tribunal.survival_rate,
                    policy=CONFIDENCE_POLICY,
                )
                # The training extract carries the clear-sky power under the
                # canonical name ``clear_sky_power_w``; the baseline
                # scoreboard expects ``clear_sky_w`` so that the clear-sky
                # naive baseline is on the table next to the persistence one.
                actual_for_baselines = actual
                if "clear_sky_w" not in actual.columns and "clear_sky_power_w" in actual.columns:
                    actual_for_baselines = actual.rename(
                        columns={"clear_sky_power_w": "clear_sky_w"}
                    )
                baselines = baseline_scoreboard(
                    actual_for_baselines,
                    last_solar=state["last_solar"],
                    last_load=state["last_load"],
                    last_soc=state["soc_percent"],
                )
                advisories = build_all(
                    batch,
                    tribunal.retained_idx,
                    report,
                    lineage={
                        "model_version": model.model_version,
                        "model_name": model.model_name,
                        "dataset_version": manifest["dataset_version"],
                        "dataset_content_hash": manifest["content_hash"],
                        "comparison": "ops_vs_pymc_reference",
                        "residual_version": None,
                        "residual_weight": 0.0,
                    },
                    ctx=ctx,
                    baselines=baselines,
                    rejection_sample=tribunal.rejections,
                )
                metrics = score_window(actual, advisories["grid_engineer"], dt_h=0.25)
                row = {
                    "split": split_name,
                    "model": model_key,
                    "origin": origin,
                    "survival_rate": tribunal.survival_rate,
                    "n_retained": tribunal.n_retained,
                    "n_rejected": tribunal.n_rejected,
                    "constraint_counts": tribunal.constraint_counts,
                    "confidence": report.confidence,
                    "uncertainty": report.uncertainty,
                    "status": report.status,
                    "metrics": metrics,
                    "solar_metrics": _band_metrics(actual, batch.solar_w),
                    "baselines": baselines,
                }
                rows.append(row)
                per_window.append(row)
            results[model_key][split_name] = _aggregate(rows)

    # ---------------------------------------------------------- the summary
    print("[5/5] writing the comparison...")
    comparison = {
        "comparison_version": "pymc_reference_comparison_v1.0",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": {
            "version": manifest["dataset_version"],
            "content_hash": manifest["content_hash"],
            "source": str(args.source),
            "window": [args.start, args.end],
            "rows": {
                "train": int(len(train)),
                "validate": int(len(validate)),
                "test": int(len(test)),
            },
            "split_policy": "time_ordered_60_20_20",
        },
        "protocol": {
            "residual_chamber": "off (both models, pure clearness comparison)",
            "horizon_steps": int(args.horizon),
            "scenarios_per_window": int(args.scenarios),
            "windows_per_split": int(args.windows),
            "seed": int(args.seed),
            "coverage_target": COVERAGE_TARGET,
            "spread_factor_bounds": list(SPREAD_FACTOR_BOUNDS),
            "selection_split": "validation",
            "test_split_usage": "report only",
            "tribunal_policy_version": TRIBUNAL_POLICY.policy_version,
            "confidence_policy_version": CONFIDENCE_POLICY.policy_version,
        },
        "ops_model": {
            "model_version": ops_model.model_version,
            "params": ops_model.sites[ctx.site_id].solar.to_dict(),
            "spread_calibration": ops_spread,
        },
        "pymc_reference": {
            **pymc_fit.manifest(),
            "fit": pymc_fit.to_dict(),
            "spread_calibration": pymc_spread,
        },
        "results": results,
        "windows": per_window,
    }
    (out_dir / "comparison.json").write_text(_dump_json(comparison), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(comparison), encoding="utf-8")
    print(render_markdown(comparison))
    return comparison


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-window rows of one (model, split) pair."""
    if not rows:
        return {}
    out: dict[str, Any] = {"n_windows": len(rows)}
    for key in ("survival_rate", "confidence", "uncertainty"):
        out[key] = {
            "mean": float(np.mean([r[key] for r in rows])),
            "min": float(np.min([r[key] for r in rows])),
            "median": float(np.median([r[key] for r in rows])),
            "max": float(np.max([r[key] for r in rows])),
        }
    for metric in ("mae", "rmse", "bias", "coverage", "coverage_p10_p90"):
        vals = [r["solar_metrics"].get(metric, float("nan")) for r in rows]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        out[f"solar_{metric}"] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "median": float(np.median(vals)) if vals else float("nan"),
            "min": float(np.min(vals)) if vals else float("nan"),
            "max": float(np.max(vals)) if vals else float("nan"),
        }
    rejects: dict[str, int] = {}
    for r in rows:
        for name, count in (r.get("constraint_counts") or {}).items():
            rejects[name] = rejects.get(name, 0) + int(count)
    out["constraint_counts"] = rejects
    out["n_rejected_total"] = int(sum(r["n_rejected"] for r in rows))
    out["n_retained_total"] = int(sum(r["n_retained"] for r in rows))

    # Baselines (the same definition for both models).
    base: dict[str, Any] = {}
    for r in rows:
        for b in r.get("baselines") or []:
            key = f"{b['target']}:{b['model']}"
            slot = base.setdefault(key, {"mae": [], "rmse": [], "bias": []})
            for m in ("mae", "rmse", "bias"):
                if m in b and b[m] is not None and np.isfinite(b[m]):
                    slot[m].append(float(b[m]))
    out["baselines"] = {
        k: {m: (float(np.mean(v)) if v else float("nan")) for m, v in slot.items()}
        for k, slot in base.items()
    }
    return out


def render_markdown(comparison: dict[str, Any]) -> str:
    """The side-by-side table (ops vs PyMC)."""
    if comparison.get("status") == "pymc_unavailable":
        return "PyMC unavailable — no comparison produced.\n"
    res = comparison["results"]
    lines: list[str] = []
    lines.append("# Ops (NumPy) vs PyMC reference — Alice Springs 2022")
    lines.append("")
    ds = comparison["dataset"]
    lines.append(
        f"Dataset `{ds['version']}` (content hash `{ds['content_hash'][:16]}…`), "
        f"time-ordered 60/20/20 split; train={ds['rows']['train']} "
        f"validate={ds['rows']['validate']} test={ds['rows']['test']}."
    )
    lines.append("")
    proto = comparison["protocol"]
    lines.append(
        f"Protocol: horizon {proto['horizon_steps']} steps, {proto['scenarios_per_window']} "
        f"scenarios/window, residual chamber {proto['residual_chamber']}, "
        f"selection on the {proto['selection_split']} split, test split {proto['test_split_usage']}."
    )
    lines.append("")
    ref = comparison["pymc_reference"]
    lines.append(
        f"Reference fit: backend **{ref['inference_backend']}**, chains {ref['chains']}, "
        f"draws {ref['draws']}, divergences {ref['divergences']}, "
        f"rhat_max {ref.get('rhat_max')}, ess_min {ref.get('ess_min')}, "
        f"wall {ref.get('wall_time_s')} s."
    )
    lines.append("")
    lines.append(
        "| split | model | survival | solar MAE (W) | solar RMSE (W) | solar bias (W) "
        "| P10-P90 coverage | confidence |"
    )
    lines.append("|" + "---|" * 8)
    for split in ("validate", "test"):
        for key, label in (("ops", "ops (NumPy)"), ("pymc", "PyMC reference")):
            r = res.get(key, {}).get(split) or {}
            if not r:
                continue
            lines.append(
                "| {split} | {label} | {surv:.4f} | {mae:,.0f} | {rmse:,.0f} | {bias:,.0f} "
                "| {cov:.4f} | {conf:.4f} |".format(
                    split=split,
                    label=label,
                    surv=r["survival_rate"]["mean"],
                    mae=r["solar_mae"]["mean"],
                    rmse=r["solar_rmse"]["mean"],
                    bias=r["solar_bias"]["mean"],
                    cov=r["solar_coverage_p10_p90"]["mean"],
                    conf=r["confidence"]["mean"],
                )
            )
    lines.append("")
    lines.append("### Baselines (same windows)")
    lines.append("")
    lines.append("| split | baseline | MAE (W) | RMSE (W) | bias (W) |")
    lines.append("|---|---|---|---|---|")
    for split in ("validate", "test"):
        r = res.get("ops", {}).get(split) or {}
        for key, vals in sorted((r.get("baselines") or {}).items()):
            if not key.startswith("solar_w:"):
                continue
            name = key.split(":", 1)[1]
            lines.append(
                f"| {split} | {name} | {vals.get('mae', float('nan')):,.0f} | "
                f"{vals.get('rmse', float('nan')):,.0f} | {vals.get('bias', float('nan')):,.0f} |"
            )
    lines.append("")
    lines.append(
        "Persistence baseline = the solar level measured at the forecast origin, held "
        "constant over the window.  Clear-sky naive = the clear-sky model of the site, "
        "used directly as the forecast."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="source CSV export")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2023-01-01")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--horizon", type=int, default=96, help="window length in 15-min steps")
    parser.add_argument("--windows", type=int, default=20, help="windows per split")
    parser.add_argument("--scenarios", type=int, default=200, help="scenarios per window")
    parser.add_argument("--spread-windows", type=int, default=6)
    parser.add_argument("--spread-scenarios", type=int, default=120)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--backend", default="auto", choices=["auto", "nuts", "vi"])
    parser.add_argument("--season", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    comparison = run_comparison(args)
    return 0 if comparison.get("status") != "pymc_unavailable" else 3


if __name__ == "__main__":
    raise SystemExit(main())
