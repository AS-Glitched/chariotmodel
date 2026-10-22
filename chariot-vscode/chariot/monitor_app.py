"""CHARIOT Monitor API — advisory only, lab limits declared in every response."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chariot.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "latest"

LIMITS = {
    "advisory_only": True,
    "not_a_controller": True,
    "ops_model": "hcbsswm_v1.7_residual_ar",
    "weather_default": False,
    "primary_lab_metric": "lab_dsm_proxy_v0_daylight",
    "primary_lab_metric_note": (
        "Daylight-only nameplate proxy: 100*|P50-actual|/rated_pv_w "
        "where clear_sky_w>50; not CERC settlement-grade"
    ),
    "allowed_claims": [
        "Beats naive clear-sky on multi-horizon energy error (lab H2/H3)",
        "Reports scenario uncertainty with Tribunal survival",
        "Fail-closed physics gate on scenarios",
        "Lab daylight DSM-proxy metric available for comparison",
        "Advisory outputs only; human oversight required",
    ],
    "forbidden_claims": [
        "Best or production-ready point forecast",
        "DSM-ready or CERC schedule compliance",
        "Beats 15-minute persistence in general",
        "Weather skill on synthetic lab data",
        "Closed-loop control authority",
    ],
    "evaluation": "docs/METRICS.md; runs/DSM_lab_proxy_v0_daylight; runs/E1b_E2_multi_horizon",
}

app = FastAPI(
    title="CHARIOT Monitor API",
    version="0.1.0-lab",
    description="Physics-governed advisory monitor. Not for closed-loop control.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "chariot-monitor"}


@app.get("/ready")
def ready():
    return {
        "status": "ready",
        "default_csv_exists": (ROOT / "data" / "site_1_30d_15min.csv").exists(),
        "last_run_dir_exists": RUNS.exists(),
    }


@app.post("/api/v1/monitor/run")
def monitor_run(n_scenarios: int = 80, seed: int = 7):
    """Run full spine on configured data (30d CSV if present) and return advisories."""
    manifest = run(
        out_dir=RUNS,
        n_scenarios=n_scenarios,
        seed=seed,
    )
    payload = {
        "limits": LIMITS,
        "manifest": manifest,
        "grid_engineer": _read_json(RUNS / "grid_engineer.json"),
        "researcher": _read_json(RUNS / "researcher.json"),
        "investor_planner": _read_json(RUNS / "investor_planner.json"),
    }
    return payload


@app.get("/api/v1/monitor/latest")
def monitor_latest():
    """Return last written advisories without re-running the model."""
    if not (RUNS / "manifest.json").exists():
        return {
            "limits": LIMITS,
            "error": "no_run_yet",
            "hint": "POST /api/v1/monitor/run first, or run: py scripts/run_prototype.py",
        }
    return {
        "limits": LIMITS,
        "manifest": _read_json(RUNS / "manifest.json"),
        "grid_engineer": _read_json(RUNS / "grid_engineer.json"),
        "researcher": _read_json(RUNS / "researcher.json"),
        "investor_planner": _read_json(RUNS / "investor_planner.json"),
    }
@app.get("/api/v1/monitor/metrics")
def monitor_metrics():
    """Return last daylight DSM-proxy summary if present (no model re-run)."""
    path = RUNS / "score_dsm_blocks.json"
    if not path.exists():
        return {
            "limits": LIMITS,
            "error": "no_metrics_yet",
            "hint": "Run: py scripts/score_dsm_blocks.py",
        }
    data = _read_json(path)
    summary = {k: v for k, v in data.items() if k != "origins"}
    return {"limits": LIMITS, "metrics": summary}

def _read_json(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))