"""Scale-aware hard/soft physics pre-gate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_row(row: pd.Series) -> dict:
    hard: list[str] = []
    soft: list[str] = []
    rated = float(row["rated_pv_w"]) if pd.notna(row.get("rated_pv_w")) else None

    for col in ("timestamp_utc", "solar_w", "load_w", "soc_percent"):
        if pd.isna(row[col]):
            hard.append("MANDATORY_FIELDS")
            break
    if pd.notna(row["solar_w"]) and float(row["solar_w"]) < 0:
        hard.append("SOLAR_NON_NEGATIVE")
    if pd.notna(row["soc_percent"]) and not (0.0 <= float(row["soc_percent"]) <= 100.0):
        hard.append("SOC_BOUNDS")
    for col in ("solar_w", "load_w"):
        if pd.notna(row[col]) and not np.isfinite(float(row[col])):
            hard.append("POWER_FINITE")
            break
    if rated is not None and pd.notna(row["solar_w"]) and float(row["solar_w"]) > 1.2 * rated:
        hard.append("SOLAR_SCALE_CEILING")

    ts = pd.Timestamp(row["timestamp_utc"])
    night = ts.hour < 5 or ts.hour > 20
    if night and rated is not None and pd.notna(row["solar_w"]):
        if float(row["solar_w"]) > max(10.0, 0.001 * rated):
            soft.append("NIGHT_SOLAR")

    if "MANDATORY_FIELDS" in hard and bool(row.get("is_gap", False)):
        decision = "INSUFFICIENT_DATA"
    elif hard:
        decision = "REJECTED"
    elif soft:
        decision = "ADMISSIBLE_WITH_WARNINGS"
    else:
        decision = "ADMISSIBLE"
    return {"decision": decision, "hard_failures": hard, "soft_warnings": soft}


def apply_gate(frame: pd.DataFrame) -> pd.DataFrame:
    reports = [evaluate_row(row) for _, row in frame.iterrows()]
    out = pd.concat([frame.reset_index(drop=True), pd.DataFrame(reports)], axis=1)
    keep = out["decision"].isin(["ADMISSIBLE", "ADMISSIBLE_WITH_WARNINGS"])
    clean = out.loc[keep].copy()
    clean["quality_flags"] = clean["soft_warnings"].apply(lambda x: list(x))
    return clean.reset_index(drop=True)
