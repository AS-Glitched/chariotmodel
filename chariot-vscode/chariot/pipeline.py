"""End-to-end CHARIOT prototype spine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chariot.admissibility import apply_gate
from chariot.advisory import build as build_advisory
from chariot.baselines import scoreboard
from chariot.extract import seal, time_split
from chariot.hcbsswm import fit, scenarios
from chariot.synthetic import make_site_frame
from chariot.tribunal import filter_scenarios

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "site_1_30d_15min.csv"


def load_raw(csv_path: str | Path | None = None, days: int = 7, seed: int = 7) -> tuple[pd.DataFrame, str]:
    """Load 30-day CSV if present; otherwise synthetic fallback."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if path.exists():
        df = pd.read_csv(path)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        if "is_gap" in df.columns:
            df["is_gap"] = df["is_gap"].astype(str).str.lower().isin(["1", "true", "yes"])
        else:
            df["is_gap"] = False
        return df, f"csv:{path.name}"
    # Fallback demo path
    raw = make_site_frame(days=days, seed=seed)
    raw.loc[raw.index[3], "solar_w"] = -50.0
    return raw, "synthetic:7d"


def run(
    out_dir: str | Path = "runs/latest",
    days: int = 7,
    n_scenarios: int = 80,
    seed: int = 7,
    csv_path: str | Path | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw, source = load_raw(csv_path=csv_path, days=days, seed=seed)
    clean = apply_gate(raw)
    meta = seal(clean, dataset_version="lab_extract_30d_v1" if source.startswith("csv:") else "lab_extract_v1")
    train, test = time_split(clean, train_frac=0.7)
    future = test.iloc[: min(96, len(test))].copy()

    board = scoreboard(test)
    model = fit(train, seed=seed)
    draws = scenarios(model, future, n_scenarios=n_scenarios)
    gate = filter_scenarios(draws, rated_pv_w=meta["rated_pv_w"])
    lineage = {
        **meta,
        "data_source": source,
        "model_name": model.model_name,
        "model_version": model.model_version,
        "split_policy": "time_ordered_v0",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "solar_scale_mean": round(model.solar_scale_mean, 4),
    }
    modes = build_advisory(future, gate, board.to_dict(orient="records"), lineage)

    manifest = {
        "status": modes["grid_engineer"]["status"],
        "survival_rate": modes["grid_engineer"]["survival_rate"],
        "lineage": lineage,
        "baselines": board.to_dict(orient="records"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "grid_engineer.json").write_text(json.dumps(modes["grid_engineer"], indent=2), encoding="utf-8")
    (out / "researcher.json").write_text(json.dumps(modes["researcher"], indent=2), encoding="utf-8")
    (out / "investor_planner.json").write_text(json.dumps(modes["investor_planner"], indent=2), encoding="utf-8")
    clean.to_parquet(out / "sealed_extract.parquet", index=False)
    return manifest
    