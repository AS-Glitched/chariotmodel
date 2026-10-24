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
    model.py          hierarchical clear-sky Bayesian SSM (the OPS model)
    pymc_clearness.py offline PyMC reference model (NUTS) + adapter
    residual.py       Residual Proposal Chamber (ridge, capped, versioned)
    genesis.py        Scenario Genesis Engine
    tribunal.py       Reality Tribunal + Rejection Echo Memory
    governor.py       Residual Stability Governor (sole runtime authority for w)
    feedback.py       Dual Feedback Cortex
    confidence.py     Physics-Survival Confidence Engine
    advisory.py       role modes, quantiles, events, abstention
    metrics.py        evaluation metrics + baseline scoreboard
    pipeline.py       the end-to-end v3 spine
    api.py            FastAPI surface (50 operations / 46 paths, advisory-only)
  scripts/            prepare_dataset.py · run_chariot3.py · diagnose_model.py
                      run_pymc_reference.py (ops vs PyMC reference comparison)
  tests/              conftest.py · test_spine.py (16) · test_api.py (6)
                      test_pymc_reference.py (11, PyMC optional marker)
  docs/ARCHITECTURE.md
  docs/REPORT.md          the engineering report (formulas, tests, results)
  docs/CHARIOT_FULL_REPORT.txt   everything in ONE file: the report + the full
                          source of every module/script/test + the run artefacts
                          (regenerate: python scripts/build_full_report.py)
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
| Bayesian inference via NUTS / Pathfinder / ADVI | **Ops**: hierarchical SSM fitted with a robust log-space EKF + method-of-moments shape/seasonal estimation, pure NumPy. **Reference**: the same contract re-implemented in PyMC with NUTS as the primary backend (`chariot/pymc_clearness.py`, offline, §8) | The ops model must stay dependency-free and deterministic; the PDF's *contract* (hierarchical clear-sky SSM, Student-t-robust observation, physics-informed transition, multi-horizon predictive) is preserved in both, and the PyMC path now provides the Bayesian posterior the PDF asks for — without changing the operational default. |
| Residual features include realised `R_t` | Uses the **model-implied** expected ratio `μ / P_clear` | `r_t = y_t − μ_t` contains `y_t`; using realised `R_t` leaks the target. Documented, not silently swapped. |
| Batch sizes 500–1000 scenarios | 200 scenarios × 40 windows in the reference run | CPU budget; the engine is size-agnostic (`--scenarios`). |
| The dataset is a solar-only export | Battery/load are **declared engineering assumptions** | DKASC Alice Springs solar-only export has no `load_w`/`soc_percent`; the site card declares them and the calibration version is the single source of truth for η. |

## 8. The offline PyMC reference model

There are **two** implementations of the clearness contract in this repository,
and they have different jobs.

| | Ops model (operational) | PyMC reference (offline) |
|---|---|---|
| Module | `chariot/model.py` | `chariot/pymc_clearness.py` |
| Inference | robust log-space EKF + method-of-moments shape/seasonal estimation, pure NumPy | NUTS (≥ 2 chains) on a hierarchical, non-centred state-space model; VI fallback, always labelled `pymc_vi` |
| Used by | the v3 spine (default) | `scripts/run_pymc_reference.py` only |
| Role | the operational model | an independent check on the operational model |

The reference model keeps the same anchor `R_t = P_meas / max(P_clear, ε)` in log
space, the same 15-bin hierarchical elevation profile, an AR(1) latent weather
state that is **reset at every night gap** (like the ops model), a Student-t
observation noise (`nu = 4`, log-normal scale prior) and an optional bounded
annual harmonic that is selected on the **validation** split only.

The posterior is projected onto the v3 `ChannelFit` by `to_channel_fit()` — the
**adapter** — and the reference scenarios are produced through the *same* Genesis
seam (`generate_batch(..., paths=...)`) and pass through the *same* Tribunal as the
operational scenarios. The Tribunal is **not** loosened for the reference, and the
operational default is **not** changed: nothing in the v3 spine imports PyMC.

**CLI only, on purpose.** There is deliberately **no** `POST /model/chariot/train_pymc`
endpoint. An HTTP route that trains a different model would be one `promote` call
away from replacing the operational model; keeping the reference behind a script
means promoting it is an explicit, reviewable act. The reference fit is therefore
written to `runs/pymc_reference/` and never to the model store.

### Reproducing the comparison

```bash
pip install -r requirements.txt            # pymc is an *extra*, not required
python scripts/run_pymc_reference.py \
    --source /path/to/dka_alice_springs_solar_only_15min_2022.csv \
    --out runs/pymc_reference \
    --draws 250 --tune 250 --chains 2 --windows 20 --scenarios 200
```

Both models are fitted on the **same** rows of the **same** training extract,
rolled forward over the same rolling-origin windows, and certified by the same
Tribunal. Outputs: `runs/pymc_reference/comparison.json` (machine readable) and
`runs/pymc_reference/summary.md` (the side-by-side table).

Rules the script obeys:

* the **test split is report-only** — every choice is made on train/validate;
* the **residual chamber is OFF** for both models, so the numbers describe the
  clearness model and not the residual corrector;
* the reference scenarios go through the **same physics gate** as the ops ones;
* the ops model stays the operational default — the script never writes an
  active model version.

### When to trust which

* Trust the **ops model** for anything operational. It is fast, deterministic,
  dependency-free and the one the spine certifies.
* Trust the **PyMC reference** as a *check*: it gives a posterior over the
  clearness parameters (with r-hat/ESS/divergences), which the NumPy model does
  not. Where the two disagree, the disagreement is information, not a bug.
* The reference is **not** a weather-forecast skill claim and **not** a
  DSM/CERC readiness claim. It models the *clearness* channel only, on a
  solar-only site whose load and battery are declared engineering assumptions.

## 9. The 2022 → 2023 evaluation protocol (W1)

The operational question CHARIOT has to answer is *does the model still work on a
year it has never seen?* The original v3 evaluation answered it with a
chronological 60/20/20 split **inside 2022**, so the "test" year was only ten
weeks long and the training window started in December 2021 (a summer month in
the southern hemisphere — a seasonal artefact in the fit).

The `year2022_2023` protocol replaces it:

| split | window | use |
| --- | --- | --- |
| `train_2022` | 2022-01-01 → 2022-08-07 (first 60 % of 2022) | fitting only |
| `validate_2022` | 2022-08-07 → 2022-12-31 (remaining 40 % of 2022) | selection only |
| `test_2023` | 2023-01-01 → 2023-12-31 (whole year) | **report only** — never fitted, never selected on |

```bash
# 1. build the slices, the manifest (content hashes + exact boundaries) and the QC
python scripts/prepare_dataset.py --protocol year2022_2023 \
    --source  <dka_alice_springs_solar_only_15min_2022.csv> \
    --source-test <dka_alice_springs_solar_only_15min_2023.csv>

# 2. run the ops spine on the protocol (the model is fitted on 2022 train only)
python scripts/run_chariot3.py \
    --source var/data/dka_alice_springs_2022_2023.parquet \
    --start 2022-01-01 --end 2024-01-01 \
    --honour-split --out runs/w1_ops_v3_protocol_2022_2023

# 3. the 2022 selection metrics (the gate every later stage is judged on)
python scripts/run_chariot3.py --source var/data/dka_alice_springs_2022_2023.parquet \
    --start 2022-01-01 --end 2024-01-01 --honour-split --evaluate-split validate \
    --out runs/w1_ops_v3_val_2022

# 4. the stage report
python scripts/build_w1_report.py
```

`--honour-split` tells the training extract to keep the split that was assigned
upstream instead of re-splitting chronologically; `--evaluate-split` chooses
which split the rolling-origin windows are drawn from. Both default to the v3
behaviour, so the existing spine and its tests are untouched.

**Measured QC (both years).** Strict 15-minute cadence, 0 duplicate timestamps,
night solar exactly 0 W, clear-sky ≠ measurement in every slice. Daylight
clearness above 1.05×: **27.68 %** (train_2022), **22.45 %** (validate_2022),
**18.85 %** (test_2023); above 1.20×: 5.20 %, 4.97 %, 1.29 %. The 2022 and 2023
exports do **not** share a clear-sky geometry (15° vs 17.5° tilt), so the
year-over-year clearness shift is a *reference* artefact — W2's entry point.

**Ops baseline on the protocol** (v3 spine unchanged, 40 rolling windows,
200 scenarios/window):

| window | MAE (W) | RMSE (W) | bias (W) | coverage | survival |
| --- | --- | --- | --- | --- | --- |
| 2022 validate (selection) | 21,143 | 28,568 | +4,403 | 0.806 | 0.944 |
| 2023 test (report only) | 17,163 | 23,223 | +3,218 | 0.888 | 0.906 |

Baselines on the same windows: clear-sky naive 22,757 W (2022 validate) and
20,304 W (2023 test); persistence-at-origin 69,054 W and 68,234 W.

## 10. The clear-sky reference (W2)

The two DKASC exports do **not** share a clear-sky reference: the 2022 export
used a 15° fixed plane, the 2023 export a 17.5° one. Worse, the exported
reference is *below* the measured plane-of-array irradiance on clear days, so
the plant "beat its own clear sky" on a quarter of the 2022 daylight rows
(clearness above 1.05, maximum above 3) — physically impossible, and the
reason a quarter of the scenarios were rejected as overscale.

`chariot/clearsky.py` rebuilds the reference from first principles:

1. **Geometry from the data.** The array layout is not in the export, so no
   tracker mix is invented. The best fixed plane is inferred from the
   *measured* POA of the 2022 train slice: **15° north**, RMSE 39.6 W/m²,
   against 174.3 W/m² for a single-axis tracker on the same rows (the fixed
   plane wins by 4.4×).
2. **A published clear-sky model.** Ineichen/Perez with the static pvlib
   Linke-turbidity climatology, transposed with Hay-Davies. The climatology
   carries no year-specific information, so applying it to the report-only
   year leaks nothing.
3. **A published DC conversion.** `clear_sky_w = poa_w_m2 × rated_pv_w / 1176`
   — the same nominal conversion the export used, so the two references are
   directly comparable.

The reference is a **data-preparation** step (pvlib); the operational model
path stays pure NumPy.

| 2022 validation window | exported reference | clear-sky v2 |
| --- | --- | --- |
| daylight clearness > 1.05 | 22.45 % | **1.78 %** |
| daylight clearness > 1.20 | 4.97 % | **1.63 %** |
| clear-sky naive baseline (MAE) | 22,757 W | 41,560 W |
| ops MAE | 21,143 W | 21,312 W |
| ops survival (mean / min) | 0.944 / 0.855 | **0.956 / 0.930** |
| SOLAR_OVERSCALE rejections | 101 | **21** |

The naive baseline got *worse* — the old reference was rigged low — while the
model's margin over it grew from 1,614 W to 20,248 W, and the physically
impossible overscale rejections collapsed. The Tribunal was not touched.

## 11. What this repository is not

- Not a controller. No setpoints, no dispatch, no closed loop.
- Not a financial guarantee. Investor/Government mode is a probabilistic planning view.
- Not a multi-tenant SaaS. Tenant isolation and RBAC are modelled in the store
  schema and policy layer, but there is no identity provider or billing.
- Not the v1 lab prototype. The v1 modules (`hcbsswm.py`, `extract.py`, `monitor_app.py`,
  `baselines.py`, `synthetic.py`) and their run artefacts were superseded by v3 and
  removed; the history remains in git.

## 12. Changelog (improvement programme W1 → W7)

The operational default is the **versioned NumPy clearness SSM**
(`chariot/model.py`). Every model change bumps the model version, and a stage
is only promoted if it passes its gate on the **2022 validation** window. The
2023 window is report-only.

| stage | what landed | model version | gate |
| --- | --- | --- | --- |
| W1 | `year2022_2023` protocol (train 2022 / validate 2022 / test 2023), `chariot/dataset_qc.py`, `--honour-split` + `--evaluate-split` in the ops CLI | unchanged (`hcbsswm_v3.0_empirical_bayes_*`) | PASS — QC clean, boundaries exact, no model change |
| W2 | clear-sky upgrade: one documented reference for both years (`chariot/clearsky.py`, `chariot_clearsky_v2.0_fixed_plane`) | unchanged (`hcbsswm_v3.0_empirical_bayes_*`), reference versioned separately | PASS — daylight kt>1.05 on 2022 val 22.45 % → 1.78 %, validation MAE +0.8 %, survival 0.944 → 0.956 |
| W3 | clearness calibration (`chariot/clearness_calibration.py`) | unchanged — the calibration is off by default | NOT PROMOTED — the validation selection kept the shrinkage at 0.0 |
| W4 | residual chamber: the previous-correction feature + night forcing | `resid_solar_w_*` (the selected weight is 0) | NOT PROMOTED — 0.0 % validation MAE improvement, so the weight stays at 0 |
| W5 | baselines and metric honesty (`chariot/metrics.py`) | unchanged | PASS — both persistences, the clear-sky naive with its energy error, the daylight % of rated |
| W6 | solar-only advisory mode (`chariot/advisory.py`) | unchanged | PASS — the SOC/grid event probabilities are nulled and flagged `hybrid_assumed` |
| W7 | low-dimensional PyMC reference (offline) | — | see `docs/PYMC_LOW_DIMENSIONAL_PLAN.md` |

Stage reports live in `runs/<stage_id>/{summary.md,comparison.json}`.

## 13. 2023 holdout results + what we claim

The 2023 window was **never fitted and never selected on**. The model is
fitted on the first 60 % of 2022, the calibration and the residual weight are
selected on the remaining 40 % of 2022, and 2023 is reported once, at the
end.

| | 2022 validation (selection) | 2023 test (report only) |
| --- | --- | --- |
| MAE | 21,312 W | 17,260 W |
| RMSE | 29,151 W | 23,713 W |
| bias | +4,558 W | +5,606 W |
| daylight MAE as % of rated | 9.82 % | 8.27 % |
| energy error | 360,247 Wh | 283,309 Wh |
| P10–P90 coverage | 0.843 | 0.879 |
| Tribunal survival (mean / min) | 0.956 / 0.930 | 0.936 / 0.415 |
| scenarios proposed / rejected | 8,000 / 351 | 8,000 / 514 |
| baseline: persistence at the horizon | 69,867 W | 68,866 W |
| baseline: persistence at 15 minutes (single step) | 9,323 W | 14,116 W |
| baseline: clear-sky naive | 41,560 W | 38,902 W |

40 rolling origins per window, 200 scenarios per origin, the v3 inference
code unchanged.

**What we claim**

* CHARIOT forecasts the **solar** channel of a solar-only site on a year it
  has never seen: 8.3 % of the rated DC power as a daylight MAE, with a
  +5.6 kW bias and a 0.879 P10–P90 coverage, and it beats the
  horizon-matched persistence by 4.0× and the clear-sky naive baseline by
  2.3×.
* The scenarios are **physics-certified by the Tribunal**: 94 % of the
  proposed paths survive, and the rejections are recorded, not dropped.
* The platform is **advisory only**. It publishes forecasts, confidence and
  event probabilities for human review. There is no dispatch, setpoint or
  control endpoint.
* The clear-sky reference is **documented and versioned**
  (`chariot_clearsky_v2.0_fixed_plane`): a fixed 15° north plane inferred
  from the measured POA of the training year, the Ineichen/Perez model with
  the static Linke climatology, and the export's own DC conversion.

**What we do not claim**

* No forecast-skill claim on the load, the battery, the state of charge or
  the grid: this site has **no meter** for any of them. The hybrid event
  probabilities are nulled in the advisory payload and marked
  `hybrid_assumed`.
* No weather-forecast skill. Nothing in this repository consumes an issued
  forecast; the ERA5 covariates are labelled covariates, not forecasts.
* No DSM/CERC readiness claim. The "% of rated" figure is a descriptive
  daylight error, not a compliance metric.
* No claim that the 2023 numbers are anything but **report-only**.
* The 2022 test tail of the old 60/20/20 protocol is **discarded**: under
  the 2022 → 2023 protocol no 2022 row is a test target.

**What the improvement programme changed**

| stage | outcome |
| --- | --- |
| W1 — protocol + QC | landed (the 2022 → 2023 protocol, the QC module, the manifests) |
| W2 — clear-sky upgrade | **landed**: the daylight clearness above 1.05 fell from 22.45 % to 1.78 % on the validation window |
| W3 — clearness calibration | **not promoted**: the selection kept the shrinkage at 0 (the bias is a tail effect, not a level error) |
| W4 — residual chamber | **not promoted**: the selected weight is 0 (the clearness innovation is not predictable at 15 minutes) |
| W5 — baselines and metrics | landed (both persistences, the clear-sky naive with its energy error, the daylight % of rated) |
| W6 — solar-only advisory | landed (the hybrid events are nulled and flagged) |
| W7 — low-dimensional PyMC reference | see `docs/PYMC_LOW_DIMENSIONAL_PLAN.md` |

## 14. License / attribution

DKASC Alice Springs data © Desert Knowledge Australia / DKEG, used under the
DKASC data licence. See the upstream dataset for terms.
