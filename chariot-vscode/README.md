# CHARIOT v3

**Physics-governed hierarchical Bayesian energy intelligence platform** for solar PV +
battery + load systems, from rooftop scale to grid scale.

This repository implements `CHARIOT-3.pdf` — the *Final Design Freeze* master reference —
end to end, on the real DKASC **Alice Springs 2022** dataset.

> **Advisory only.** CHARIOT never controls a device. It publishes physics-certified
> forecasts, confidence, and event probabilities for human review. There is no
> dispatch, setpoint, or closed-loop control endpoint — by design and by test.

```text
Telemetry
  → Site Master          (scale context: ratings, battery, limits)
  → Admissibility Pre-Gate (fail-closed physics rules on every row)
  → River Flow           (content-addressed vertical/horizontal slices + replay)
  → Calibration          (versioned, scoped, audited effective parameters)
  → Capsule              (immutable, content-addressed seal)
  → Training Extract     (materialised Parquet + manifest + content hash)
  → CHARIOT model        (hierarchical clear-sky Bayesian state-space world model)
  → Residual Chamber     (shadow corrector, hard ceiling w ≤ w_max)
  → Scenario Genesis     (trajectory fan, no pre-dropping)
  → Reality Tribunal     (per-timestep physics, fail-closed) + Rejection Echo
  → Confidence Engine    (survival × sharpness)
  → Advisory API         (Research · Grid Engineer · Investor/Government)
```

---

## 1. Install

```bash
cd chariot-vscode
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

Pure NumPy + pandas + pyarrow on the model side (no PyMC, no GPU). FastAPI is only
needed for the HTTP surface.

## 2. Run on the Alice Springs dataset

The source export (`dka_alice_springs_solar_only_15min_2022.csv`, 16 MB, 35 002 rows)
is **not** committed. Ingest it from wherever it lives:

```bash
# 1. materialise the full-year slice + deterministic 60/20/20 split
python scripts/prepare_dataset.py \
    --source /path/to/dka_alice_springs_solar_only_15min_2022.csv

# 2. run the full v3 spine
python scripts/run_chariot3.py \
    --source var/data/dka_alice_springs_2022_15min.parquet \
    --out    runs/chariot3_alice_springs_2022 \
    --scenarios 500

# 3. diagnostics for the clearness channel
python scripts/diagnose_model.py var/data/dka_alice_springs_2022_15min.parquet
```

`prepare_dataset.py` writes the slice **and** a `.manifest.json` recording the content
hash and the exact train/validate/test timestamp boundaries. The slice itself lands in
`var/` (git-ignored) — the repo carries code and small JSON manifests only.

## 3. Measured result (full year, 40 rolling windows, 200 scenarios/window)

| Metric | Value |
|---|---|
| Physics survival rate | **0.906** (min 0.840, max 0.955) |
| Confidence `C` | **0.417** (28 eligible · 12 reduced-confidence · 0 abstained) |
| Solar MAE / RMSE | **25.5 kW** / 33.0 kW |
| Solar bias | **+14.8 kW** (intrinsic — P50 is a median of a left-skewed distribution) |
| P10–P90 coverage | **0.765** (validation-fitted spread factor 1.295) |
| Baseline (persistence) MAE | 67.8 kW |

Tribunal rejections across the year (754 of 8 000):

| Constraint | Count |
|---|---|
| `SOLAR_RAMP_LIMIT` | 465 |
| `SOLAR_OVERSCALE` | 293 |
| `GRID_EXPORT_LIMIT` | 11 |
| `GRID_IMPORT_LIMIT` | 2 |

Every rejection is written to the append-only Rejection Echo — never dropped.

## 4. Repository layout

```text
chariot-vscode/
  chariot/            the v3 package (20 modules, see docs/ARCHITECTURE.md)
    config.py         v3 identity, versioned policies, horizons, SiteContext
    hashing.py        canonical JSON + SHA-256 lineage helpers
    store.py          SQLite lineage/audit store (stdlib only)
    site_master.py    site registry + effective-context resolution
    admissibility.py  hard/soft fail-closed physics rule engine
    river_flow.py     content-addressed slices + replay certificates
    calibration.py    versioned, scoped, audited calibration
    capsule.py        immutable capsule sealing/verification
    training_extract.py  materialised dataset + clear-sky gate + 60/20/20 split
    physics.py        bus-balance solver (the Tribunal's energy accounting)
    model.py          hierarchical clear-sky Bayesian SSM
    residual.py       Residual Proposal Chamber (ridge, capped, versioned)
    genesis.py        Scenario Genesis Engine
    tribunal.py       Reality Tribunal + Rejection Echo Memory
    governor.py       Residual Stability Governor (sole runtime authority for w)
    feedback.py       Dual Feedback Cortex
    confidence.py     Physics-Survival Confidence Engine
    advisory.py       role modes, quantiles, events, abstention
    metrics.py        evaluation metrics + baseline scoreboard
    pipeline.py       the end-to-end v3 spine
    api.py            FastAPI surface (46 operations / 44 paths, advisory-only)
  scripts/            prepare_dataset.py · run_chariot3.py · diagnose_model.py
  tests/              conftest.py · test_spine.py (16) · test_api.py (3)
  docs/ARCHITECTURE.md
  runs/               small JSON manifests/metrics from real runs
  var/                runtime: store, capsules, slices, parquet (git-ignored)
```

## 5. HTTP surface

```bash
uvicorn chariot.api:app --reload --port 8000
```

`GET /health` returns the live policy snapshot and `advisory_only: true`.

| Group | Routes |
|---|---|
| Data layer | `/sites`, `/sites/{id}/context`, `/admissibility/{evaluate,decisions,rule-sets}`, `/flow/ingest`, `/slices/{vertical,horizontal}`, `/replay/{id}`, `/calibration/versions` (+`approve`/`retire`/`resolve`), `/capsules/seal`, `/training-extract/build` |
| Model core | `/model/chariot/{train,infer,scenarios,versions,promote}`, `/model/residual/{train,propose,versions}`, `/model/scenarios/generate` |
| Tribunal | `/tribunal/validate-batch`, `/tribunal/batches/{id}/retained`, `/tribunal/batches/{id}/rejections`, `/tribunal/rejections`, `/tribunal/rejections/{id}`, `/tribunal/policy` (GET + PUT) |
| Feedback / governor | `/feedback/cortex/run`, `/feedback/jobs`, `/residual/governor/{status,evaluate,quarantine,set-weight}` |
| Confidence | `/confidence/evaluate`, `/confidence/reports/{id}`, `/confidence/policy` (GET + PUT) |
| Advisory | `/advisory/forecast`, `/advisory/events`, `/advisory/confidence` |

**Explicitly absent:** any control, dispatch, or setpoint endpoint.
`tests/test_api.py::test_no_control_endpoints_exist` enforces this.

A `PUT` on a policy **mints a new version** — the previous one is never edited in
place, and the full version history is kept in `policy_versions`:

```text
tribunal_v3.0  →  tribunal_v3.0.1  →  tribunal_v3.0.2  …
```

## 6. Hard invariants (enforced by tests)

1. The Tribunal is the **sole** energy-accounting authority for retained futures.
2. Every advisory scenario passes strict physics validation at **every** timestep.
3. The residual cannot bypass the Tribunal and cannot exceed its hard ceiling.
4. Rejections from Admissibility *and* Tribunal are logged and never dropped.
5. River Flow payloads are content-addressed and replay-certifiable.
6. Calibration is versioned, scoped, and audited.
7. Capsules **earn** their immutability — no store-first-validate-later.
8. All outputs are advisory-only.
9. Scale awareness comes from explicit Site Master ratings, not separate engines.
10. Full cryptographic lineage, unbroken, end to end.

`tests/test_spine.py` asserts each of these against a synthetic reference site;
`tests/test_api.py` walks the whole HTTP spine.

## 7. Deviations from the PDF (and why)

| PDF | This implementation | Reason |
|---|---|---|
| Bayesian inference via NUTS / Pathfinder / ADVI | Hierarchical SSM fitted with a robust log-space EKF + method-of-moments shape/seasonal estimation, pure NumPy | The environment has no PyMC/Stan and no GPU; the PDF's *contract* (hierarchical clear-sky SSM, Student-t-robust observation, physics-informed transition, multi-horizon predictive) is preserved. |
| Residual features include realised `R_t` | Uses the **model-implied** expected ratio `μ / P_clear` | `r_t = y_t − μ_t` contains `y_t`; using realised `R_t` leaks the target. Documented, not silently swapped. |
| Batch sizes 500–1000 scenarios | 200 scenarios × 40 windows in the reference run | CPU budget; the engine is size-agnostic (`--scenarios`). |
| The dataset is a solar-only export | Battery/load are **declared engineering assumptions** | DKASC Alice Springs solar-only export has no `load_w`/`soc_percent`; the site card declares them and the calibration version is the single source of truth for η. |

## 8. What this repository is not

- Not a controller. No setpoints, no dispatch, no closed loop.
- Not a financial guarantee. Investor/Government mode is a probabilistic planning view.
- Not a multi-tenant SaaS. Tenant isolation and RBAC are modelled in the store
  schema and policy layer, but there is no identity provider or billing.
- Not the v1 lab prototype. The v1 modules (`hcbsswm.py`, `extract.py`, `monitor_app.py`,
  `baselines.py`, `synthetic.py`) and their run artefacts were superseded by v3 and
  removed; the history remains in git.

## 9. License / attribution

DKASC Alice Springs data © Desert Knowledge Australia / DKEG, used under the
DKASC data licence. See the upstream dataset for terms.
