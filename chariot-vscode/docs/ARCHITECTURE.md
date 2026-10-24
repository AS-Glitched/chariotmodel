# CHARIOT v3 — Architecture

Implementation of `CHARIOT-3.pdf` (Final Design Freeze). This document maps every
PDF layer to the module that implements it, states the contracts between layers, and
records where the implementation deliberately departs from the document.

---

## 1. Layer map

| PDF layer (design freeze index) | Module | Contract |
|---|---|---|
| Site Master | `chariot/site_master.py` | `resolve_context(tenant, site) → SiteContext` with `rated_pv_w`, battery Wh, charge/discharge W, timezone, fleet, `source`, `missing_rating`. Resolution order: request override → active site row → missing (scale ceiling skipped, everything else still runs). |
| Admissibility Pre-Gate | `chariot/admissibility.py` | `evaluate_record` / `evaluate_frame` → `Decision` ∈ {ADMISSIBLE, ADMISSIBLE_WITH_WARNINGS, REJECTED, INSUFFICIENT_DATA}. Hard: MANDATORY_FIELDS, SOLAR_NON_NEGATIVE, SOC_BOUNDS, POWER_FINITE, SOLAR_SCALE_CEILING (κ=1.2). Soft: ENERGY_BALANCE, NIGHT_SOLAR, SOC_RATE_LIMIT. Every evaluation is logged with `input_fingerprint` + `rule_set_version`. |
| River Flow Extraction Fabric | `chariot/river_flow.py` | `build_vertical_slice` / `build_horizontal_slice` / `issue_replay_certificate` / `verify_replay`. Content-addressed Parquet, idempotent, flags and gaps preserved, no physics rejection (that is upstream's job). |
| Calibration Engine | `chariot/calibration.py` | Four effective parameters: `c_f` (0,1.2], `ℓ_c` [0,0.5), `ℓ_d` [0,0.5), `γ_pv` (0,1.2]. Scope priority equipment → site → fleet → identity. Lifecycle draft → approved → retired, every transition audited. `resolve` always succeeds (identity default). |
| Immutable Capsule Sealing | `chariot/capsule.py` | `seal_capsule` → `capsule_id = sha256(payload ‖ canonical_header)`. Seal-time checks: non-empty, valid range, monotonic timestamps, schema, admission purity, calibration approved-or-identity, site known. Payload is never updated — a correction is a new capsule. `verify_capsule` rebuilds the header from the stored manifest and re-hashes. |
| Training Extract | `chariot/training_extract.py` | Materialised Parquet + manifest + content hash. Clear-sky **required** for trainable rows (`MISSING_CLEAR_SKY` otherwise); non-trainable rows are kept with `is_trainable=false`. Identity: tenant, site, timestamp, capsule, dataset_version. Manifest: dataset_version, feature_set_id (`physics_rich_v1`), capsule_ids, calibration/admissibility versions, trainable_row_count, content_hash, time range. Split: **time-ordered 60 / 20 / 20** with boundaries recorded in the manifest. |
| CHARIOT model | `chariot/model.py` | Hierarchical clear-sky Bayesian state-space world model. Anchor `R_t = P_meas / max(P_clear, ε)`. |
| Residual Proposal Chamber | `chariot/residual.py` | `ŷ = μ_CHARIOT + w·r̂`, `0 ≤ w ≤ w_max` (0.20). Ridge family (LightGBM/XGBoost permitted by the PDF; not installed here). Versioned artefacts, leakage-safe features, fixed target model version. |
| Scenario Genesis Engine | `chariot/genesis.py` | Multivariate trajectory ensembles in Tribunal format. No physics rejection inside Genesis, no pre-dropping of "ugly" scenarios, no user-facing quantiles. Reads the effective `w` from the Governor. |
| Reality Tribunal + Rejection Echo | `chariot/tribunal.py`, `chariot/physics.py` | Per-timestep, fail-closed validation of every scenario. Energy-balance residual ≤ 0.1 Wh, SOC within `[SOC_min, SOC_max]`, power/equipment limits from Site Master + policy, optional ramp limits. Any single timestep failure rejects the whole scenario. Echo is append-only and its write is required for batch success. |
| Dual Feedback Cortex | `chariot/feedback.py` | Path A (model): re-weighting near recurrent failures, physics-informed penalties, dataset window proposals → new train jobs → new model version. Path B (residual): quarantine requests to the Governor, w maintenance, attenuation, recovery. |
| Residual Stability Governor | `chariot/governor.py` | **Sole runtime authority** for `w`. State machine `active ↔ attenuated ↔ quarantined`. Signals: `c = E[|r̂|]/E[|ŷ|]`, `ρ = corr(|r̂|, reject_indicator)`, per-regime reject rates, residual-on vs residual-off survival delta. Genesis always calls `get_effective_w()`; the Cortex cannot bypass it. |
| Physics-Survival Confidence Engine | `chariot/confidence.py` | `s = N_retained/N_proposed`, `u` = normalised P90−P10 width **from retained scenarios only**, `C = g(1−u)·s`. Policy: `s < s_min` → abstained; `C < C_low` → reduced confidence; `u` high with `s < 1` → reduced confidence; else eligible for human review. Survival dominates sharpness. |
| Advisory API | `chariot/advisory.py`, `chariot/api.py` | Role modes (Research / Grid Engineer / Investor-Government), horizons short / operational / long (never exceeding the certified trajectory length), P10/P50/P90 from retained scenarios only, empirical event probabilities, abstention with explicit reasons and HTTP 200. `advisory_only: true` on every response. |

Supporting modules:

| Module | Role |
|---|---|
| `chariot/config.py` | v3 identity, versioned policy dataclasses, horizon table, `SiteContext`, `policy_snapshot()` |
| `chariot/hashing.py` | canonical JSON, SHA-256, frame/file content hashes |
| `chariot/store.py` | SQLite lineage + audit store (stdlib only). Tables: sites, admissibility_decisions, river_slices, calibration_versions, calibration_applications, capsules, datasets, model_versions, residual_versions, scenario_batches, tribunal_results, rejections, feedback_jobs, governor_events, confidence_reports, advisories, policy_versions, audit_log. JSON columns auto-decode on read. |
| `chariot/metrics.py` | point metrics, band coverage, energy error, window scoring, baseline scoreboard |
| `chariot/pipeline.py` | the end-to-end v3 spine + spread calibration + reference site card |

---

## 2. The spine, step by step

`chariot/pipeline.py::run` executes the following, in order, for every rolling
forecast window:

1. **Load** the time window from the prepared Parquet (full year by default).
2. **Admissibility** — evaluate every row against the hard/soft rule engine; only
   `ADMISSIBLE` and `ADMISSIBLE_WITH_WARNINGS` survive. Counts land in the manifest.
3. **River Flow** — build one vertical slice (full state over time) and horizontal
   slices (one parameter vs. time), then issue a **Replay Certificate**
   (`slice_ids`, `content_hashes`, `root_hash`, `schema_version`).
4. **Calibration** — resolve the effective factors for the site scope, create a
   calibration version, and stamp the factors onto the seal.
5. **Capsule** — seal the admitted payload; the capsule ID is the content hash of
   payload + canonical header.
6. **Training Extract** — materialise the dataset, apply the clear-sky gate, assign
   the deterministic time-ordered 60/20/20 split, write the manifest with the
   content hash and split boundaries.
7. **CHARIOT model** — fit the hierarchical clearness SSM on the **train** split.
8. **Residual** — fit the Ridge chamber on train, select `w` on **validate**, cap it
   at `w_max`.
9. **Spread calibration** — bisect a spread factor so that the **validation**
   P10–P90 coverage hits the target (0.78), bounded to `[0.70, 1.30]`.
10. **Scenarios** — Genesis fans out S trajectories from the origin state.
11. **Tribunal** — validate every scenario at every timestep; retain survivors, write
    every rejection to the Echo.
12. **Confidence** — compute `s`, `u`, `C` and the status from the retained set.
13. **Advisory** — publish the three role panels from the retained set only.

The **test** split is never touched during fitting or calibration — it is the
held-out measurement window used by `chariot/metrics.py`.

---

## 3. The CHARIOT model (`chariot/model.py`)

The model is a hierarchical clear-sky Bayesian state-space world model. The clearness
channel is the load-bearing part, and it is deliberately **multiplicative in log
space** because the clear-sky ratio is right-skewed and its upper tail is bounded:

```text
observation   y_t = gain_t · exp(u_t) · ε_t        (robust, Huberised)
process       u_t = ρ · u_{t-1} + η_t              (weather AR(1) state)
gain_t        = μ_shape(elevation_bin_t) · seasonal_factor(day_of_year_t)
```

Fitting is a two-stage procedure:

* **Stage 1 — method of moments.** A diurnal/elevation profile `μ_shape` is estimated
  by a weighted through-origin regression of the observed power on clear-sky power
  inside each elevation bin. An annual harmonic seasonal factor
  `1 + a·sin(2πd/365) + b·cos(2πd/365)` is then fitted and **selected against the
  validation split** (harmonics 0–3 are compared on validation MAE). The AR(1)
  coefficient `ρ` and scale `σ` are estimated by a robust, Huber-weighted IRLS fit of
  the residual ratio against its *same-solar-elevation, consecutive-daylight* lag
  partner, with a stable level estimator (weighted sample mean, not the
  ill-conditioned `(m_y − ρm_x)/(1 − ρ)`).
* **Stage 2 — one filter pass.** A Kalman filter on `log(ratio)` with the fitted
  process/measurement scales produces the filtered state and its variance, which the
  simulator seeds.

Clearness is hard-capped at `CLEARNESS_MAX = 1.25` (above the admissibility ceiling
of 1.2 × rated, but bounded to keep the simulator in the physical envelope).

**Simulation** draws `u_t` forward as an AR(1) around the *seasonally and elevation
shaped* mean, applies the observation gain, multiplies by clear-sky power, and
produces the solar ensemble. The ensemble spread is then multiplied by the
validation-fitted `spread_factor` (1.295 on the reference run).

`one_step_prediction` and `observation_gain` are exposed for diagnostics
(`scripts/diagnose_model.py`).

### Why not the PDF's NUTS/Pathfinder/ADVI **in the ops path**?

The operational path must be fast, deterministic and dependency-free, so the
clearness channel is estimated with a deterministic, CPU-only, pure-NumPy estimator
(the PDF's *contract* — hierarchical partial pooling, robust Student-t-style
observation, physics-informed transition, multi-horizon predictive — is preserved
in the *implementation* of the ops model).

The PDF's *sampler* is nevertheless implemented, offline, as an independent
reference: see **§3.1 The offline PyMC reference model** below. The ops default is
unchanged; the reference exists so that the ops model can be *checked* against a
genuine Bayesian posterior, not so that it can be replaced by one.

---

## 3.1 The offline PyMC reference model (`chariot/pymc_clearness.py`)

An additive, **offline** second implementation of the same clearness contract.
It is never imported by the v3 spine, never writes an active model version and
never changes an operational default.

### The model

```text
observation   log R_i = log_profile[b(i)] + season(doy_i) + σ_proc · s_i + ε_i
ε_i           StudentT(ν = 4, scale = obs_scale)
state         s_i = ρ · s_{i-1} + u_i            (AR(1), reset at every night gap)
u_i           ~ Normal(0, 1)                     (free, non-centred)
profile       log_profile_b = fleet_mu_b + fleet_sigma · z_b,  z_b ~ Normal(0, 1)
season        season_cos · cos θ + season_sin · sin θ          (optional)
```

Five deliberate choices, each of which was necessary to make NUTS work at all:

1. **Non-centred AR(1).** The free variable is the innovation series `u`, not the
   state. The map `u → s` is lower-triangular with unit diagonal, so the density is
   *identical* (unit Jacobian) but the sampled variables are on a standard-normal
   scale. With the centred form the posterior is a narrow ridge in 8 000
   dimensions: NUTS stalled in it (r-hat 1.21, `sigma_process` collapsed to 0.0015)
   even though the optimum was ~17 000 log units away.
2. **The chain is reset at every night gap**, exactly as in the ops model — the
   clearness residual does not survive the night. Without the reset the sampler
   drove `ρ` to its 0.999 boundary and the state became a within-day random walk.
3. **Non-centred hierarchy with the Jacobian term.** `n_bins · log(fleet_sigma)` is
   added as a `pm.Potential`, which makes the joint density over
   `(fleet_mu, fleet_sigma, log_profile)` identical to the centred form while
   removing the funnel (the centred form gave r-hat 1.13 / ESS 12 on the profile
   bins).
4. **Single-threaded BLAS.** OpenBLAS's per-core pool made every small dot in the
   HMC integrator cost ~16 ms; the NUTS iteration time fell from 24–98 s to
   0.6–0.75 s once the pool was pinned to one thread (`threadpoolctl` plus the
   `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1` environment variables, set before
   NumPy is imported).
5. **An identifiable observation model.** `obs_scale` has a *log-normal* prior
   (log 0.05, sd 0.75) and the Student-t degrees of freedom are **fixed** at 4.
   With a half-normal scale and a free `nu`, the likelihood can explain the
   same outliers either with heavier tails or with the latent state; the
   observation scale was then weakly identified (the two chains ended at
   r-hat 2.7 with `obs_scale` collapsed to 0.002). `rho` is bounded below by
   0.5 for the same reason — an unrestricted persistence lets the state
   become flexible enough to absorb the noise.

### The backends

| Backend | When | Labelled as |
|---|---|---|
| `pymc_nuts` | default, ≥ 2 chains, `adapt_diag` init, `max_treedepth=8` | `inference_backend: pymc_nuts` |
| `pymc_vi` | fallback when NUTS cannot be used (or `--backend vi`) | `inference_backend: pymc_vi` — **never** presented as NUTS; the manifest carries `vi_diagnostics` instead of r-hat/ESS |

### The adapter (`to_channel_fit`)

The posterior is projected onto the v3 `ChannelFit`:

| PyMC | v3 `ChannelFit` | Note |
|---|---|---|
| `exp(log_profile)` | `profile` | the ops model stores a *shape*, the reference stores an absolute log ratio → `gamma = 1.0` |
| `rho` | `rho` | identical meaning |
| `sigma_process` | `sigma` | both are the **innovation** scale of the AR(1) |
| `obs_scale` | `obs_scale` | both in log units |
| `season_cos/sin` | `season_cos/sin` | the ops model applies `exp(·)` and clips to [0.6, 1.4]; the reference applies the raw additive term |

The reference scenarios are then produced by `simulate_paths()` (a forward
simulation from the **posterior draws**, not from a Gaussian summary) and pushed
through the **same** Genesis seam, `generate_batch(..., paths=...)`, so they go
through the same Tribunal, the same content hash and the same lineage as the
operational scenarios.

### Why there is no `POST /model/chariot/train_pymc`

The reference is reachable **only** through `scripts/run_pymc_reference.py`. An
HTTP route that trains a *different* model would be one `promote` call away from
replacing the operational model, and the API is the surface where "an operator
did something" has to be auditable. Keeping the reference behind a script means
that promoting it stays an explicit, reviewable act; the reference fit is
written to `runs/pymc_reference/` and never into the model store. This is the
"CLI-only acceptable if safer" option in the brief.

### The comparison protocol (`scripts/run_pymc_reference.py`)

1. The **same** spine runs down to the training extract, so both models see
   byte-identical rows.
2. The ops model is fitted on the **train** split; the spread factor is
   calibrated on **validate** (target coverage 0.78, bounded to [0.70, 1.30]).
3. The PyMC reference is fitted on the **train** split, with the seasonal
   harmonic selected on **validate** (kept only if it improves the validation
   MAE by ≥ 1 %).
4. The same spread calibration is applied to the reference ensemble.
5. Both models are rolled forward over the same rolling-origin windows on
   **validate** and **test**, with the residual chamber **off** for both.
6. Every batch goes through the same Tribunal; the metrics are the same.
7. The **test split is report-only** — nothing is fitted or selected on it.

The outputs are `runs/pymc_reference/comparison.json` (machine readable) and
`runs/pymc_reference/summary.md` (the side-by-side table).

### Invariants the reference must respect

* night rows and sub-threshold clear-sky rows produce **exactly zero** solar;
* the clearness ratio is capped at `CLEARNESS_MAX = 1.25` (no uncapped ratios);
* `clear_sky_w` is never used as `solar_w`;
* the Tribunal is identical and is not loosened for the reference;
* nothing is ever fitted on the test split.

---

## 4. Physics (`chariot/physics.py`)

The bus-balance solver is the Tribunal's energy accounting. Per timestep:

```text
solar + discharge + import  ==  load + charge + export + losses
```

with the battery constrained to `[SOC_min, SOC_max]` (5 %–95 %) and to its rated
charge/discharge power. The solver **caps charge and discharge by the remaining SOC
energy**, so it cannot charge a full battery or over-drain an empty one — an
unconstrained solver produces impossible trajectories and the Tribunal (correctly)
rejects them.

---

## 5. Site card (reference)

The DKASC Alice Springs export is solar-only (no `load_w`, no `soc_percent`), so the
battery and load are **declared engineering assumptions** carried in the site card and
resolved through the calibration version:

| Field | Value |
|---|---|
| `rated_pv_w` | 220 000 W |
| `usable_battery_capacity_wh` | 220 000 Wh |
| `max_charge_w` / `max_discharge_w` | 110 000 / 110 000 W |
| grid import / export limit | 60 000 / 250 000 W |
| solar ramp limit | 110 000 W per 15-min step |
| SOC band | 5 %–95 % (initial 50 %) |
| charge / discharge loss | 3 % / 3 % (η = 0.97, from the calibration version) |
| declared load profile | 16–47 kW, `load_relative_sigma` 0.18 |
| latitude / longitude | Alice Springs, NT (−23.698, 133.881) |

**The calibration version is the single source of truth for η.** The Genesis engine
and the Tribunal both read it (`genesis._battery_from_context`), so the generator and
the judge solve the *same* energy balance. A mismatch here produced a 100 %
`ENERGY_BALANCE` rejection rate during bring-up — the bug that fix now guards against.

Measured physics envelope of the dataset itself: max solar 0.921 × rated; |Δsolar| per
15 min p50 346 W, p90 15 962 W, p99 0.261 × rated, p99.9 0.405 × rated; **0 %** of
steps above 0.5 × rated.

---

## 6. Multi-scale, multi-tenancy

One rule family, parameterised by the Site Master. A 5 kW rooftop and a 5 MW plant
differ only in `rated_pv_w`, `usable_battery_capacity_wh`, and the power limits.
Tenant isolation is enforced by `tenant_id` on every table and every object key;
the same rule family applies across tenants.

---

## 7. What CHARIOT will not do (and how that is enforced)

| Prohibited | Enforcement |
|---|---|
| Automatic closed-loop device control | No control/dispatch/setpoint route exists — `test_no_control_endpoints_exist` |
| Residual or NN as energy-accounting authority | The Tribunal's deterministic solver is the only authority; residual output is a *proposal* that must pass it |
| Silent drop of rejections | Every rejection is persisted to the Echo; batch success requires the echo write |
| Store-first immutability | Capsules are sealed only after admission + calibration + seal-time checks |
| Unversioned calibration or policy | Every policy dataclass carries a `policy_version`; calibration versions are immutable and lifecycle-audited |
| Advisory numbers without certified scenarios when abstained | Abstention returns an explicit reason with no fabricated P50 |

---

## 8. Lineage chain

```text
telemetry
 → admissibility_decision (rule_set_version, input_fingerprint)
 → river_slice (content hash)
 → calibration_version (4 factors)
 → capsule_id (sha256 payload ‖ header)
 → dataset_version (content hash, split boundaries)
 → chariot_model_version
 → residual_version? (+ effective w from the Governor)
 → scenario batch (content_hash, lineage)
 → tribunal result + rejection_ids
 → confidence_report
 → advisory_id
```

Every link is stored in `chariot/store.py` and reproduced verbatim in the run
manifest (`runs/<run>/manifest.json`).

---

## 9. Test coverage

`tests/test_spine.py` (16 tests) asserts the invariants against a synthetic reference
site: admissibility decisions and fail-closed behaviour, content addressing and
replay verification, calibration scoping and lifecycle, capsule immutability,
clear-sky gating and split determinism, residual ceiling, Genesis → Tribunal
survival, echo completeness, governor authority, confidence status, abstention,
and lineage completeness.

`tests/test_api.py` (6 tests) walks the whole HTTP spine — ingest → admit → slice →
calibrate → seal → extract → train → scenarios → confidence → advisory →
feedback, asserts `advisory_only: true`, asserts the absence of control endpoints,
asserts that abstention returns explicit reasons rather than a fabricated P50,
checks that the Rejection Echo is individually addressable, and checks that a
policy change mints a **new version** rather than editing the old one in place.

`tests/test_pymc_reference.py` (13 tests) covers the **offline reference path**.
Most of them need no PyMC at all: the operational spine must run with PyMC
*absent* (a subprocess import check), the prepared data must mirror the ops lit
mask, the adapter must produce a well-formed `ChannelFit`, the manifest must
carry every required field, the reference ensemble must be night-zero, finite and
capped, and the Genesis seam must accept the reference paths **without changing
the operational default**. Two tests are marked `pymc` and are skipped when the
dependency is missing: a real 1-chain tiny NUTS fit on the synthetic site, and
the night-reset property of the AR(1) scan. Run them with::

    python -m pytest tests/ -q -m pymc

---

## 10. HTTP surface

50 operations across 46 paths, matching the PDF §6 map:

| Group | Operations |
|---|---|
| Sites | `GET/POST /sites`, `GET /sites/{id}/context` |
| Admissibility | `POST /admissibility/evaluate`, `GET /admissibility/decisions`, `GET /admissibility/rule-sets` |
| River Flow | `POST /flow/ingest`, `GET /slices/vertical`, `GET /slices/horizontal`, `GET /replay/{id}` |
| Calibration | `GET/POST /calibration/versions`, `POST /calibration/versions/{v}/approve`, `POST /calibration/versions/{v}/retire`, `GET /calibration/resolve` |
| Capsule | `POST /capsules/seal`, `GET /capsules/{id}` |
| Training extract | `POST /training-extract/build`, `GET /training-extract/datasets/{version}` |
| Model core | `POST /model/chariot/{train,infer,scenarios,promote}`, `GET /model/chariot/versions`, `POST /model/residual/{train,propose}`, `GET /model/residual/versions`, `POST /model/scenarios/generate`, `GET /model/scenarios/batches/{id}` |
| Tribunal | `POST /tribunal/validate-batch`, `GET /tribunal/batches/{id}/retained`, `GET /tribunal/batches/{id}/rejections`, `GET /tribunal/rejections`, `GET /tribunal/rejections/{id}`, `GET/PUT /tribunal/policy` |
| Feedback / Governor | `POST /feedback/cortex/run`, `GET /feedback/jobs`, `GET /residual/governor/status`, `POST /residual/governor/{evaluate,quarantine,set-weight}` |
| Confidence | `POST /confidence/evaluate`, `GET /confidence/reports/{id}`, `GET/PUT /confidence/policy` |
| Advisory | `POST /advisory/forecast`, `GET /advisory/events`, `GET /advisory/confidence` |
| Meta | `GET /health` |

`GET /health` returns the live policy snapshot and `advisory_only: true`. There is
**no** control, dispatch or setpoint route — `test_no_control_endpoints_exist`
asserts that no path contains `control`, `dispatch`, `setpoint`, `actuate` or
`command`.
