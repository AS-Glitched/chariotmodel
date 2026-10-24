# CHARIOT — complete engineering report

**Repository:** `https://github.com/AS-Glitched/chariotmodel` (branch `arena/eb404247-chariotmodel`, PR #1)
**Scope:** CHARIOT v3 (the operational spine) **plus** the additive, offline PyMC reference path.
**Status at the time of writing:** 35/35 tests green, CI green on Python 3.10/3.11/3.12, PR #1 open.

Every number in this document comes from a file in the repository or from a command
that is quoted verbatim. Nothing is estimated.

---

## 0. Table of contents

1. [What was asked, and what was delivered](#1-what-was-asked-and-what-was-delivered)
2. [The v3 operational spine (phase 1)](#2-the-v3-operational-spine-phase-1)
3. [The offline PyMC reference path (phase 2)](#3-the-offline-pymc-reference-path-phase-2)
4. [Every formula used](#4-every-formula-used)
5. [Every test](#5-every-test)
6. [Every failure encountered and how it was resolved](#6-every-failure-encountered-and-how-it-was-resolved)
7. [Measured results](#7-measured-results)
8. [Everything used: tools, libraries, commands, environment](#8-everything-used-tools-libraries-commands-environment)
9. [What is deliberately NOT claimed](#9-what-is-deliberately-not-claimed)
10. [How to reproduce everything](#10-how-to-reproduce-everything)

---

## 1. What was asked, and what was delivered

### 1.1 Phase 1 — "read this whole github, and chariot 3 pdf is final software"

`CHARIOT-3.pdf` (26 pages, the *Final Design Freeze*) was parsed with `pypdf`
(`pdftotext` is not installed in this sandbox) and implemented layer by layer.

| Deliverable | Status | Where |
|---|---|---|
| v3 spine, 23 modules | done | `chariot-vscode/chariot/` (9 557 lines) |
| HTTP surface | done | 46 paths / 50 operations, `chariot/api.py` |
| Full-year run on the Alice Springs 2022 slice | done | `runs/chariot3_alice_springs_2022/` |
| Tests | done | 22 (now 35) |
| Docs | done | `README.md`, `docs/ARCHITECTURE.md` |
| GitHub Actions CI | done | `.github/workflows/ci.yml` |
| Git history / PR | done | commits `1d45346`, `ae4d67a`, `9e83a9a`, `782de77`; PR #1 |

### 1.2 Phase 2 — "add an offline PyMC reference path for solar clearness"

The brief, verbatim in intent:

1. A PyMC hierarchical/state-space model for **clearness only**: multiplicative
   log space, AR(1) latent weather state, elevation/clear-sky structure, optional
   seasonal selected on validation, robust observation noise.
2. **NUTS primary** (≥ 2 chains, modest CPU settings, documented counts); a
   VI/Pathfinder-style fallback clearly labelled **"VI reference"**, never as NUTS.
3. Posterior draws → the existing Scenario Genesis / Tribunal path, or a
   documented adapter, so retained scenarios still pass the physics gates.
4. A comparison of the PyMC reference vs the NumPy MoM+EKF ops model on the
   **same** Alice Springs 2022 time-ordered 60/20/20 split.
5. **Do not** make PyMC the ops default until the gates pass.
6. Entry points: `scripts/run_pymc_reference.py`; `POST /model/chariot/train_pymc`
   optional (CLI-only acceptable if safer).
7. Manifest fields: `inference_backend`, `model_version`, `chains`, `draws`,
   `divergences`, r-hat/ESS (or VI diagnostics), wall time, train-extract content hash.
8. Residual chamber OFF for the pure reference comparison, or the same ceiling
   policy as v3; never leak test targets.
9. Evaluation: physics survival rate (Tribunal), solar MAE/RMSE/bias, P10–P90
   coverage, a persistence baseline with an explicit definition, clear-sky naive
   metrics, a side-by-side ops-vs-PyMC table. Selection on **validation only**;
   **test is report-only**.
10. Promote toward ops only if survival is not materially worse, calibration is
    not collapsed, and MAE/coverage improve on validation without test peeking.

**Non-goals honoured:** no v3 module or NumPy ops model was deleted; no load/SOC
learning from fabricated meters; the Tribunal was not loosened; no weather-forecast
skill or DSM/CERC readiness is claimed; nothing was fitted on the test split; no
large CSV/Parquet/db/run blobs were committed.

**Hard rules honoured:** night and sub-threshold clear-sky → solar exactly 0; the
clearness ratio is capped at `CLEARNESS_MAX = 1.25` *after* the observation noise;
`clear_sky_w` is never used as `solar_w`; the ops default was never replaced.

### 1.3 What was *not* done, and why

| Item | Decision |
|---|---|
| `POST /model/chariot/train_pymc` | **Not implemented — CLI only.** An HTTP route that trains a *different* model is one `promote` call away from replacing the operational model. The brief explicitly allowed CLI-only "if safer". |
| Promoting PyMC to the ops default | **Not done.** The gates in requirement 10 are not met (the posterior diagnostics are poor — see §7.3). The ops model stays the default; the reference is a *check*, not a replacement. |
| Free Student-t `nu` | **Fixed at 4.** A free `nu` trades off against the latent state and leaves the observation scale unidentified (measured: r-hat 2.4, `obs_scale` → 0.002). Documented, not silent. |

---

## 2. The v3 operational spine (phase 1)

### 2.1 Layer map (PDF layer → module)

| PDF layer | Module | Lines | Contract |
|---|---|---|---|
| Site Master | `site_master.py` | 196 | `resolve_context(tenant, site) → SiteContext`; resolution order request override → active site row → missing (scale ceiling skipped, everything else still runs). |
| Admissibility Pre-Gate | `admissibility.py` | 294 | `evaluate_record`/`evaluate_frame` → {ADMISSIBLE, ADMISSIBLE_WITH_WARNINGS, REJECTED, INSUFFICIENT_DATA}. Hard: MANDATORY_FIELDS, SOLAR_NON_NEGATIVE, SOC_BOUNDS, POWER_FINITE, SOLAR_SCALE_CEILING (κ = 1.2). Soft: ENERGY_BALANCE, NIGHT_SOLAR, SOC_RATE_LIMIT. Every decision logged with `input_fingerprint` + `rule_set_version`. |
| River Flow Extraction Fabric | `river_flow.py` | 206 | `build_vertical_slice`/`build_horizontal_slice`/`issue_replay_certificate`/`verify_replay`. Content-addressed Parquet, idempotent, flags and gaps preserved. |
| Calibration Engine | `calibration.py` | 245 | Four effective parameters `c_f ∈ (0,1.2]`, `ℓ_c ∈ [0,0.5)`, `ℓ_d ∈ [0,0.5)`, `γ_pv ∈ (0,1.2]`. Scope priority equipment → site → fleet → identity. Lifecycle draft → approved → retired, every transition audited. |
| Immutable Capsule Sealing | `capsule.py` | 196 | `capsule_id = sha256(payload ‖ canonical_header)`; seal-time checks; payload never updated; `verify_capsule` rebuilds the header and re-hashes. |
| Training Extract | `training_extract.py` | 334 | Materialised Parquet + manifest + content hash; clear-sky **required** for trainable rows; deterministic time-ordered 60/20/20 split with recorded boundaries. |
| CHARIOT model | `model.py` | 1 288 | Hierarchical clear-sky Bayesian state-space world model, anchor `R_t = P_meas / max(P_clear, ε)`. |
| Residual Proposal Chamber | `residual.py` | 318 | `ŷ = μ_CHARIOT + w·r̂`, `0 ≤ w ≤ w_max = 0.20`, ridge family, versioned artefacts, leakage-safe features. |
| Scenario Genesis Engine | `genesis.py` | 261 | Multivariate trajectory ensembles in Tribunal format; no physics rejection inside Genesis; reads the effective `w` from the Governor. |
| Reality Tribunal + Rejection Echo | `tribunal.py` (308), `physics.py` (157) | | Per-timestep, fail-closed validation of every scenario; any single timestep failure rejects the whole scenario; the echo write is required for batch success. |
| Dual Feedback Cortex | `feedback.py` | 287 | Path A (model) and Path B (residual). |
| Residual Stability Governor | `governor.py` | 261 | **Sole runtime authority** for `w`; state machine `active ↔ attenuated ↔ quarantined`. |
| Physics-Survival Confidence Engine | `confidence.py` | 196 | `s = N_retained/N_proposed`, `u` = normalised P90−P10 width from **retained scenarios only**, `C = g(1−u)·s`. |
| Advisory API | `advisory.py` (270), `api.py` (1 240) | | Role modes, horizons, quantiles from retained scenarios only, abstention with explicit reasons, `advisory_only: true` on every response. |

Supporting: `config.py` (287), `hashing.py` (96), `store.py` (533), `site_master.py`,
`metrics.py` (161), `pipeline.py` (962), `pymc_clearness.py` (1 496).

### 2.2 The spine, in order

`chariot/pipeline.py::run` executes, for every rolling forecast window:

1. Load the 15-minute window.
2. **Admissibility** — every row against the hard/soft rule engine.
3. **River Flow** — one vertical slice + horizontal slices, then a **Replay Certificate**
   (`slice_ids`, `content_hashes`, `root_hash`, `schema_version`).
4. **Calibration** — resolve the effective factors, create a version, stamp onto the seal.
5. **Capsule** — seal the admitted payload; the ID is the content hash.
6. **Training Extract** — materialise, apply the clear-sky gate, assign the 60/20/20 split.
7. **CHARIOT model** — fit the clearness SSM on the **train** split.
8. **Residual** — fit the Ridge chamber on train, select `w` on **validate**, cap at `w_max`.
9. **Genesis** — draw the scenario ensemble with the Governor's effective `w`.
10. **Tribunal** — validate every scenario; write the Rejection Echo.
11. **Confidence** — `s`, `u`, `C` → status.
12. **Advisory** — quantiles, event probabilities, abstention reasons, lineage.

---

## 3. The offline PyMC reference path (phase 2)

File: `chariot-vscode/chariot/pymc_clearness.py` (1 496 lines). Harness:
`chariot-vscode/scripts/run_pymc_reference.py`.

### 3.1 The model

```text
ANCHOR           R_t = solar_w_t / max(clear_sky_w_t, ε)

OBSERVATION      log R_i  ~  StudentT(ν = 4,
                                     μ_i = log_profile[b(i)] + season(doy_i) + σ_proc·s_i,
                                     scale = obs_scale)

STATE            s_i = ρ·s_{i-1} + u_i          for consecutive observed rows
                 s_i = u_i                      after a night gap (the chain resets)
                 u_i ~ Normal(0, 1)             (free, non-centred)

PROFILE          log_profile_b = fleet_mu_b + fleet_sigma·z_b ,   z_b ~ Normal(0, 1)
                 fleet_mu_b ~ Normal(0, 0.5)      (one per elevation bin)
                 fleet_sigma  ~ HalfNormal(0.25)
                 + Potential: n_bins · log(fleet_sigma)      (Jacobian)

SEASON           season(doy) = season_cos·cos θ + season_sin·sin θ ,  θ = 2π·doy/365.25
                 season_cos, season_sin ~ Normal(0, 0.1)

PRIOR SUMMARY    ρ ~ Uniform(0.5, 0.999)          σ_proc ~ HalfNormal(0.3)
                 obs_scale ~ LogNormal(log 0.05, 0.75)          ν ≡ 4
```

### 3.2 Why each non-obvious choice was made (all measured on this host)

| Choice | Alternative rejected | Measurement that forced it |
|---|---|---|
| **Non-centred AR(1)** (sample `u`, build `s` with a `scan`) | centred: `pm.Flat` state + analytic `pm.Potential` | The centred posterior is a narrow ridge in 8 000 dimensions. NUTS stalled: r-hat **1.21**, `sigma_process` collapsed to **0.0015**, while the log-density at the true optimum was **~17 000 log units higher** (`logp`: default −22 935, MoM −25 536, degenerate −21 688, vs **−7 202** at ρ=0.928/σ=0.5 and **−4 104** at σ=0.6/obs=0.05). The `u → s` map is lower-triangular with unit diagonal, so the Jacobian is 1 and **the density is unchanged**. |
| **Chain reset at every night gap** | AR(1) over the whole 15-min grid | The full-grid chain carries information across the night (ρ → 0.999 boundary pile-up, the state degenerating into a within-day random walk). The clearness residual does not survive the night — the ops model breaks the chain at night too. |
| **Non-centred hierarchy + Jacobian term** | `log_profile ~ Normal(fleet_mu, fleet_sigma)` | The centred form shows the classic funnel: r-hat **1.13**, ESS **12** on the profile bins. Adding `n_bins·log(fleet_sigma)` keeps the joint density *identical* to the centred form while removing the funnel. |
| **Single-threaded BLAS** (env vars before `import numpy` + `threadpoolctl`) | default OpenBLAS thread pool | `velocity_energy` on a 21 038-element vector cost **15 990 µs** with the default pool and **35–72 µs** pinned to one thread (≈ 400×). NUTS iteration time fell from **24–98 s to 0.6–0.75 s** (≈ 100×). |
| **`cxx=`, `linker=numba`, `mode=Mode`, `optimizer_excluding=fusion`** | the C backend (unavailable: no `Python.h`) | Without Python headers PyTensor cannot emit *any* C code. `mode=NUMBA` is a **predefined** mode and silently ignores `optimizer_excluding`; `mode=Mode` (mixed case) applies it. Without the fusion exclusion, PyTensor's fused `Composite` op degrades to a scalar Python loop (≈ 10× slower). |
| **Log-normal prior on `obs_scale`, fixed `ν = 4`, `ρ ≥ 0.5`** | HalfNormal scale, free `ν`, `ρ ∈ (0, 0.999)` | With the flexible versions the latent state can absorb the observation noise: `obs_scale` collapsed to **0.002**, the posterior became weakly identified and the two chains ended at r-hat **2.43 / ESS 2.5**. (This is *not* fully solved — see §7.3.) |
| **No `POST /model/chariot/train_pymc`** | an HTTP training route | The brief allowed CLI-only "if safer"; an HTTP route is one `promote` call from replacing the operational model. |

### 3.3 The adapter (`to_channel_fit`)

The posterior is projected onto the v3 `ChannelFit` so the whole downstream spine can
be reused unchanged:

| PyMC | v3 `ChannelFit` | Note |
|---|---|---|
| `exp(log_profile)` | `profile` | the ops model stores a *shape*, the reference an absolute log ratio → `gamma = 1.0` |
| `rho` | `rho` | identical meaning |
| `sigma_process` | `sigma` | both are the **innovation** scale of the AR(1) |
| `obs_scale` | `obs_scale` | both in log units |
| `season_cos/sin` | `season_cos/sin` | the ops model applies `exp(·)` and clips to [0.6, 1.4]; the reference applies the raw additive term |

Reference scenarios are produced by `simulate_paths()` — a forward simulation from
the **posterior draws** (not a Gaussian summary) — and pushed through the **same**
Genesis seam `generate_batch(..., paths=...)`, so they get the same Tribunal, the
same content hash and the same lineage as operational scenarios.

### 3.4 The comparison protocol (`scripts/run_pymc_reference.py`)

1. The **same** spine runs down to the training extract → both models see byte-identical rows.
2. The ops model is fitted on **train**; the spread factor is calibrated on **validate**
   (target coverage 0.78, bounded to [0.70, 1.30]).
3. The PyMC reference is fitted on **train**; the seasonal harmonic is selected on
   **validate** and kept only if it improves the validation MAE by **≥ 1 %**.
4. The same spread calibration is applied to the reference ensemble.
5. Both models are rolled over the same rolling-origin windows on **validate** and **test**,
   horizon 96 steps, 200 scenarios/window, 20 windows/split, **residual chamber OFF for both**.
6. Every batch goes through the same Tribunal; the metrics are the same.
7. The **test split is report-only** — nothing is fitted or selected on it.

Outputs: `runs/pymc_reference/comparison.json` (174 KB, committed) and
`runs/pymc_reference/summary.md` (1.5 KB, committed).

---

## 4. Every formula used

### 4.1 The ops clearness model (`chariot/model.py`)

```text
ANCHOR            R_t = P_meas,t / max(P_clear,t, ε)

OBSERVATION       log(R_t) = μ_shape(bin(elev_t))·γ + seasonal_factor(doy_t) + u_t + ε_t
PROFILE           μ_shape(b) estimated by a weighted through-origin regression of
                  P_meas on P_clear inside each elevation bin (15 bins, edges 0…90° step 6°)
SEASONAL          seasonal_factor = clip(exp(a·cos θ + b·sin θ), 0.6, 1.4),  θ = 2π·doy/365.25
PROCESS           u_t = μ + ρ·(u_{t-1} − μ) + η_t ,   η_t ~ Student-t(ν = 5)/√(ν/2)
OBSERVATION NOISE ε_t ~ Huberised (robust) — the EKF uses a Huber weight
CLEARNESS CAP     u = clip(μ_shape·season·exp(v), 0, CLEARNESS_MAX = 1.25)
POWER             solar = γ · P_clear · u
SPREAD            ensemble spread × spread_factor (validation-fitted, bounded [0.70, 1.30])
```

Fitting is two-stage:

* **Stage 1 — method of moments.** The profile by weighted through-origin regression;
  the annual harmonic fitted and **selected on validation** (harmonics 0–3 compared on
  validation MAE); `ρ` and `σ` by a robust, Huber-weighted IRLS fit of the residual
  ratio against its *same-solar-elevation, consecutive-daylight* lag partner, with the
  stable level estimator

  ```text
      μ = Σ w_i·y_i / Σ w_i          (weighted sample mean, NOT the ill-conditioned
                                       (m_y − ρ·m_x)/(1 − ρ))
  ```
* **Stage 2 — one filter pass.** A Kalman filter on `log(ratio)` with the fitted
  process/measurement scales produces the filtered state and its variance.

The Kalman/EKF update (log space):

```text
  predict:   v_pred = μ + ρ(v_{t-1} − μ) ,      var_pred = ρ²·var_{t-1} + σ²
  update:    obs = log R_t − log(gain_t)
             K = var_pred / (var_pred + s²)      (s = observation scale)
             v_t = v_pred + K(obs − v_pred)      var_t = (1 − K)·var_pred
```

### 4.2 The PyMC reference model

See §3.1 for the model block. The three pieces of mathematics behind it:

**(a) The non-centred AR(1) and why the density is unchanged.**
The forward map `u ↦ s` is lower-triangular with unit diagonal:

```text
      s_1 = u_1
      s_2 = ρ·s_1 + u_2
      s_3 = ρ·s_2 + u_3          ⇒   s = L(ρ)·u ,  L lower triangular, diag(L) = 1
```

so `|∂s/∂u| = 1` and `p(s) = p(u)` under the change of variables. PyMC's own
`ar_logp` for `pm.AR` reduces to exactly this expression, which is why the centred
Potential and `pm.AR` gave the identical log-density (−22 935.193254) before the
switch. The inverse used for the initialisation is

```text
      u_1 = s_1 ,      u_i = s_i − ρ·s_{i-1}        (and u_i = s_i after a gap)
```

**(b) The non-centred hierarchy and its Jacobian.**
`log_profile_b = fleet_mu_b + fleet_sigma·z_b` with `z_b ~ N(0,1)` has Jacobian
`|∂log_profile/∂z| = fleet_sigma^{n_bins}`, so the compensating potential

```text
      log|J| = n_bins · log(fleet_sigma)
```

restores the centred joint density over `(fleet_mu, fleet_sigma, log_profile)` exactly,
while sampling `z` on a fixed standard-normal scale (no funnel).

**(c) The night reset.** The observed-row index gaps `Δ_i = idx_i − idx_{i−1}` are
compared with 1; wherever `Δ_i ≠ 1` the chain restarts:

```text
      keep_i = 1 if Δ_i = 1 else 0
      s_i    = keep_i · (ρ·s_{i-1}) + u_i
```

which is exactly `s_i = ρ·s_{i-1} + u_i` inside a day and `s_i = u_i` after a night.

### 4.3 Physics, Tribunal, confidence, advisory

```text
BUS BALANCE (per timestep)   P_pv + P_grid_import + P_batt_discharge
                             = P_load + P_grid_export + P_batt_charge + losses
ENERGY BALANCE RESIDUAL      |residual| ≤ 0.1 Wh          (hard, fail-closed)
SOC BOUNDS                   SOC_min ≤ SOC ≤ SOC_max     (5 % … 95 %)
POWER LIMITS                 charge ≤ 110 kW, discharge ≤ 110 kW,
                             grid import ≤ 60 kW, grid export ≤ 250 kW
RAMP LIMIT                   |P_t − P_{t−1}| ≤ 110 kW/step
SOLAR SCALE CEILING          P_solar ≤ 1.2 × rated_pv_w  (κ = 1.2)
NIGHT SOLAR                   P_solar = 0 whenever the clear sky is below threshold
CONFIDENCE                   s = N_retained/N_proposed
                             u = normalised P90−P10 width (retained scenarios only)
                             C = g(1 − u)·s
                             s < s_min ⇒ abstained ; C < C_low ⇒ reduced_confidence
ADVISORY QUANTILES           P10/P50/P90 from the retained scenarios only
```

### 4.4 The comparison metrics

```text
MAE      = mean(|solar_truth − solar_forecast|)
RMSE     = sqrt(mean((solar_truth − solar_forecast)²))
bias     = mean(solar_forecast − solar_truth)
coverage = fraction of timesteps inside [P10, P90] of the retained band
survival = N_retained / N_proposed           (Tribunal)
persistence baseline = the solar level at the forecast origin, held constant
clear-sky naive      = the site's clear-sky model used directly as the forecast
```

---

## 5. Every test

35 tests, 13.4 s locally (`python -m pytest tests -q` → `35 passed`).

### 5.1 `tests/test_spine.py` (16 tests, the v3 invariants)

| Test | What it asserts |
|---|---|
| `test_content_hash_is_stable_and_sensitive` | the canonical-JSON hash is deterministic and changes when any byte changes |
| `test_admissibility_rejects_negative_solar` | the hard rule `SOLAR_NON_NEGATIVE` fires |
| `test_admissibility_flags_clear_sky_gap` | a missing clear sky is rejected (`MISSING_CLEAR_SKY`) |
| `test_slices_are_content_addressed_and_replayable` | River Flow slices are content-addressed and the replay certificate verifies |
| `test_capsule_seals_and_verifies` | a capsule seals, verifies and cannot be edited in place |
| `test_pipeline_produces_a_full_lineage` | the spine emits every lineage layer |
| `test_lineage_chain_is_cryptographically_linked` | the lineage chain hashes are linked, not merely present |
| `test_advisory_is_advisory_only` | `advisory_only: true` and no control semantics anywhere |
| `test_every_published_path_passed_the_tribunal` | every retained scenario passed the Tribunal at every timestep |
| `test_rejections_are_recorded_not_dropped` | the Rejection Echo is complete |
| `test_residual_never_exceeds_its_ceiling` | `w ≤ w_max = 0.20` always |
| `test_confidence_reports_a_status` | the confidence engine returns a status, not a number alone |
| `test_bands_are_ordered` | P10 ≤ P50 ≤ P90 |
| `test_split_is_time_ordered_and_disjoint` | the 60/20/20 split is time-ordered and disjoint |
| `test_tribunal_rejects_impossible_trajectories` | the Tribunal rejects physically impossible paths |
| `test_tribunal_policy_is_fail_closed` | an unknown rule set fails closed |

### 5.2 `tests/test_api.py` (6 tests, the HTTP surface)

`test_health_and_policy_surface`, `test_no_control_endpoints_exist`,
`test_full_api_spine` (ingest → admit → slice → calibrate → seal → extract → train →
scenarios → tribunal → confidence → advisory → feedback),
`test_rejection_echo_is_addressable`, `test_policies_are_versioned_never_edited_in_place`,
`test_advisory_carries_full_lineage`.

### 5.3 `tests/test_pymc_reference.py` (13 tests, the offline reference)

| Test | Marker | What it asserts |
|---|---|---|
| `test_operational_spine_never_imports_pymc` | | a subprocess imports `chariot.pipeline`, `model`, `api`, `genesis` and asserts `pymc`/`pytensor` are **not** in `sys.modules` — the ops spine never touches PyMC |
| `test_reference_module_imports_and_reports_availability` | | `pymc_available()` is a bool; `require_pymc()` raises `PymcUnavailableError` with an install hint when PyMC is missing |
| `test_config_validation_rejects_nonsense` | | `backend="mcmc"`, `season="sometimes"`, `chains=0`, `draws=0` are all rejected |
| `test_prepared_data_mirrors_the_ops_lit_mask` | | the reference's observation mask is **exactly** the ops model's lit mask (`valid & clear > 1.0` then `clear > 0.2 × p90(clear)`); every observed row has a finite log clearness ratio |
| `test_reference_paths_are_night_zero_finite_and_capped` | | the reference ensemble is finite, non-negative, **exactly zero** on night rows, and the clearness ratio never exceeds `CLEARNESS_MAX = 1.25` |
| `test_reference_paths_respect_the_site_rating` | | `solar ≤ 1.2 × rated_pv_w` |
| `test_reference_paths_track_the_clear_sky_ratio` | | **regression test for the log-profile bug**: the median predicted clearness must lie in (0.4, 1.3) and the band must not be degenerate. With the log profile used as a ratio the reference produced ≈ 0 solar (a −97 kW bias) while still passing the finiteness/night/cap checks. |
| `test_reference_channel_fit_adapter_shape` | | the adapter returns `gamma = 1.0`, `mu = 0.0`, 15 profile bins, `0 ≤ ρ < 1`, a positive observation floor |
| `test_manifest_carries_every_required_field` | | the manifest carries `inference_backend`, `model_version`, `chains`, `draws`, `divergences`, `rhat_max`, `ess_min`, `wall_time_s`, `train_extract_content_hash` |
| `test_generate_batch_accepts_reference_paths_without_changing_the_default` | | the reference paths flow through `generate_batch(..., paths=...)`, the batch is used verbatim, the physics is still solved for every path, the SOC is finite, the lineage still points at the model version and a content hash exists — **and the operational default is untouched** |
| `test_tiny_nuts_fit_produces_a_usable_posterior` | `pymc` | a **real** 1-chain, 25+25 NUTS fit on the synthetic site: finite `ρ`, `σ_proc`, `obs_scale`, `0 < ρ < 1`, the manifest is populated, and the simulated paths are finite and night-zero |
| `test_model_builds_and_its_log_probability_is_finite` | `pymc` | the model builds and `compile_logp()(initial_point())` is finite |
| `test_ar1_state_resets_at_every_gap` | `pymc` | the scan reproduces the recursion exactly **and** the state after a gap depends only on the innovation after that gap (`s₀ = u₀`, `s₁ = ρu₀ + u₁`, `s₂ = u₂`, `s₃ = ρu₂ + u₃`, `s₄ = ρs₃ + u₄`) |

---

## 6. Every failure encountered and how it was resolved

### 6.1 v3 (phase 1)

| Symptom | Cause | Fix |
|---|---|---|
| The EM clearness fit collapsed (σ → 0) | the EM loop was fitting the wrong thing | abandoned the EM loop; log-space two-stage estimator |
| The profile inflated without bound | EM fed its own profile back into the next iteration | method of moments + a single filter pass |
| The left tail was under-covered | an additive clearness parameterisation | log-space state |
| The spread factor reached 2.14 | an unbounded spread calibration | the factor is bounded to [0.70, 1.30] |
| The seasonal extrapolated badly | a linear day-of-year trend | a bounded annual harmonic selected on validation |
| The P50 was biased +12 kW (the median −15 kW) | mean calibration | median calibration |
| 100 % `ENERGY_BALANCE` rejections | the generator's efficiency did not match the Tribunal's | generator/Tribunal efficiency alignment |
| `admissibility_decisions` has no `reason_codes_json` | a schema mismatch | the column was dropped |
| `river_slices` uses `slice_type` | a schema mismatch | the field name was corrected |
| Pydantic rejected non-numeric `calibration_factors` | a typing bug | typed as `dict[str, float]` |
| `ScenarioBatch` has no `.calibration` | a wrong attribute | the attribute was removed |
| `/advisory/events` lacked `horizon_steps` | the runtime state was not consulted | `runtime.retained[batch_id].retained_idx` |
| The residual feature leaked the target | the *realised* `R_t` was used as a feature | the model-implied `μ / P_clear` is used instead |
| `json.dumps` emitted bare `NaN` | non-finite floats | `_json_safe` + `allow_nan=False` |
| `next_policy_version` collided | a counter was not incremented | the counter is appended |

### 6.2 The PyMC reference (phase 2)

| Symptom | Cause | Fix | Evidence |
|---|---|---|---|
| NUTS 2 × (250+250) did not finish in 25 min | the gradient was fine (0.4 ms) but the *integrator* was slow | profiling showed `velocity_energy` costing 15 990 µs → **OpenBLAS thread contention**; pinning BLAS to one thread gave 35–72 µs and 0.6–0.75 s per iteration | §3.2 |
| `cProfile` reported "total time 0" | the profiler is useless under the Numba backend | manual timing instead | — |
| `pm.AR` gradient 278 ms on 21 001 rows | `pm.AR` is a random variable; its gradient goes through an `OpFromGraph` reverse scan | replaced by an explicit Potential, then by the non-centred `scan` | — |
| `pytensor.scan` raised `TypeError: TensorVariable cannot be converted to Python integer` | PyTensor slices the sequences with a symbolic step count, which fails on statically shaped vectors | the first sequence is reshaped to an unknown length, the reset flags are passed as an explicit `TensorVariable`, `n_steps` is **not** passed, and the length is restored with `specify_shape` | — |
| `scan` inside `pm.Model` raised `TypeError` | PyMC rewrites `pm.Model` inputs | a plain `pytensor.scan` with a concrete length is used inside the model | — |
| The first real fit gave ρ = 0.111, σ_proc = 0.0015, r-hat 1.21, ESS 17 | the **centred** parameterisation is a narrow ridge | the non-centred parameterisation | logp table in §3.2 |
| ρ piled up at the 0.999 boundary | the AR(1) carried information across the night | the chain resets at every gap | §3.2 |
| The profile's r-hat 1.13 / ESS 12 | the hierarchical funnel | the non-centred hierarchy + the Jacobian term | §3.2 |
| The reference predicted ≈ 0 solar (a −97 kW bias) while passing every finiteness/night/cap check | the **log** profile was used as a ratio (`base = profile · exp(season)`) | `log_u = log_profile(elev) + season + v; u = exp(log_u)` | regression test `test_reference_paths_track_the_clear_sky_ratio` |
| The clearness ratio reached 1.51 (> the 1.25 cap) | the observation floor was inside the product, and the multiplicative noise was applied after the cap | `solar = clear_sky · u` (the ops `simulate` convention) and the cap is re-applied **after** the noise | `test_reference_paths_are_night_zero_finite_and_capped` |
| The clear-sky naive baseline was missing from the table | the training extract names the column `clear_sky_power_w`, the scoreboard expects `clear_sky_w` | the frame is renamed for the baseline computation | the baseline table in §7.2 |
| The two chains ended at r-hat 2.43 / ESS 2.5 | the latent state absorbs the observation noise (`obs_scale` → 0.002) → the posterior is weakly identified | partially mitigated (log-normal `obs_scale` prior, fixed `ν = 4`, `ρ ≥ 0.5`); **not solved** — see §7.3 | §7.3 |

---

## 7. Measured results

### 7.1 The operational v3 run (full year, 40 windows, 200 scenarios/window)

`runs/chariot3_alice_springs_2022/{manifest.json,summary.json,windows.json}`

```text
dataset            dataset_b607cabd7da7, content hash 4aaad752153c895d…
split (trainable)  train 9 975 | validate 3 416 | test 3 796   (time-ordered 60/20/20)
admissibility      35 002 evaluated → 17 696 ADMISSIBLE, 16 568 WITH_WARNINGS,
                   738 INSUFFICIENT_DATA (hard: MANDATORY_FIELDS; soft: NIGHT_SOLAR)
model              hcbsswm_v3.0_empirical_bayes_c7f466ff60
  μ = 0.000875, ρ = 0.95449, σ = 0.19493, γ = 0.95607, obs_scale = 0.04680, floor = 50 W
  15-bin profile peak 1.15028, spread_factor 1.29531, seasonal NOT selected
residual           resid_solar_w_f2593127a1, ridge, w_max 0.20, selected w = 0.07
tribunal           survival 0.90575 (min 0.840, max 0.955); 8 000 proposed, 754 rejected
  constraint counts SOLAR_RAMP_LIMIT 465, SOLAR_OVERSCALE 293,
                    GRID_EXPORT_LIMIT 11, GRID_IMPORT_LIMIT 2
confidence         mean C 0.41718, mean u 0.53892; 28 eligible, 12 reduced_confidence
solar (advisory)   MAE 25 496.8 W, RMSE 34 324.1 W, bias +14 800.1 W, coverage 0.76544
baseline           persistence MAE 67 752.8 W
```

### 7.2 The ops vs PyMC reference comparison

`runs/pymc_reference/{comparison.json,summary.md}` — dataset `dataset_f2288a01f008`,
content hash `d6712722426cea9f…`, train 9 975 / validate 3 416 / test 3 796.

```text
Reference fit: backend pymc_nuts, chains 2, draws 250, tune 250, divergences 0,
               rhat_max 2.43, ess_min 2.50, wall 1 389.6 s
               model_version pymc_clearness_v3.0_nuts_51057c29ec
               posterior mean: ρ 0.9278, σ_proc 0.2037, obs_scale 0.00170, season not selected
Protocol: horizon 96 steps, 200 scenarios/window, 20 windows/split,
          residual chamber off (both models), selection on validation, test report-only

Ops model (same rows): μ 0.000785, ρ 0.95449, σ 0.19493, γ 0.95607, obs_scale 0.04680,
          profile peak 1.15028, seasonal not selected
Spread calibration (validation, target coverage 0.78):
          ops       achieved 0.7517 → factor 1.2953   (bounded to [0.70, 1.30])
          reference achieved 0.7778 → factor 1.0984
```

| split | model | survival | solar MAE (W) | solar RMSE (W) | solar bias (W) | P10–P90 coverage | confidence |
|---|---|---|---|---|---|---|---|
| validate | ops (NumPy) | 0.9592 | 21 179.7 | 29 127.9 | +4 631.0 | 0.7766 | 0.5000 |
| validate | **PyMC reference** | **0.9950** | **21 014.4** | 29 374.6 | +6 354.1 | **0.8016** | **0.5405** |
| test | ops (NumPy) | 0.8962 | 26 049.6 | 35 913.9 | +14 967.8 | 0.7224 | 0.4096 |
| test | **PyMC reference** | **0.9760** | 26 500.0 | 37 001.3 | +18 160.2 | 0.6958 | **0.4870** |

Baselines (the same windows, the same definition for both models):

| split | baseline | MAE (W) | RMSE (W) | bias (W) |
|---|---|---|---|---|
| validate | persistence_origin | 61 537.3 | 78 475.8 | +16 027.9 |
| validate | clear_sky_naive | 21 166.0 | 30 534.2 | +10 698.2 |
| test | persistence_origin | 72 370.5 | 86 570.4 | +15 347.4 |
| test | clear_sky_naive | 33 547.8 | 46 750.3 | +31 092.6 |

Read-out against the promotion gates of requirement 10:

| Gate | Verdict |
|---|---|
| Survival not materially worse | **pass** — the reference survives *better* (0.995 vs 0.959 on validation; 0.976 vs 0.896 on test) |
| Calibration not collapsed | **pass on the band** — coverage 0.802 (val) / 0.696 (test) vs the ops model's 0.777 / 0.722 |
| MAE / coverage improve on validation | **marginal pass** — MAE 21 014 vs 21 180 W (−0.8 %), coverage 0.802 vs 0.777 (+2.5 points) |
| Posterior is usable (r-hat/ESS) | **FAIL** — r-hat 2.43, ESS 2.50 (§7.3) |
| **Promotion decision** | **NO.** The ops model stays the operational default. The reference is reported as an independent check with its diagnostics attached. |

### 7.3 The one open problem: the posterior does not converge

`rhat_max = 2.43`, `ess_min = 2.50`, `divergences = 0`. The *predictions* are good
(§7.2) and the posterior means are **reproducible across independent runs**
(ρ 0.9273/0.9278, σ_proc 0.2037/0.2037), and an instrumented 2-chain run shows the
two chains sitting on **the same** elevation profile to four decimals — so the
sampler is finding the right region every time. What it is not doing is *moving*.

The posterior sds from the published run explain why:

| parameter | posterior mean | posterior sd | relative sd |
|---|---|---|---|
| `rho` | 0.927795 | 0.000446 | 4.8 × 10⁻⁴ |
| `sigma_process` | 0.203702 | 6.0 × 10⁻⁵ | 3.0 × 10⁻⁴ |
| `obs_scale` | 0.0016992 | 1.2 × 10⁻⁴ | 7.3 × 10⁻² |

The posterior is a **needle**. The cause is the opposite of what it looks like:
the model is *over*-identified, not weakly identified. The latent state carries
one innovation per observed row (`n_obs_rows = 8 108` on 9 975 grid rows) and the
Student-t observation noise is driven down to 0.0017, so the state plus the
15-bin profile can reproduce the observations almost exactly. The posterior is then
a near-zero-measure manifold, the adapted diagonal mass matrix is fine but the
trajectory has to crawl along it, and 500 iterations yield ~2.5 effective samples.

Mitigations already in the code: the non-centred AR(1), the non-centred hierarchy
with its Jacobian, the night reset, `ρ ≥ 0.5`, a log-normal `obs_scale` prior, a
fixed `ν = 4`, `adapt_diag` initialisation from the method-of-moments point, and a
single-threaded BLAS.

Not yet tried — the honest next steps, in the order I would take them:

1. **A coarser latent state.** Put the AR(1) on an hourly (or 30-minute) grid and
   interpolate to the 15-minute observations. That drops the state from ~8 100 to
   ~2 000 dimensions, which is the dimension that makes the posterior a needle.
2. **`init = "advi+adapt_diag"`** (mean-field ADVI warm start *and* a diagonal mass
   matrix) instead of the identity mass matrix the `adapt_diag` path starts from.
3. **A much longer tuning run** (≥ 1 000 steps) so the diagonal mass matrix is
   estimated from enough samples, and more draws.
4. **The VI fallback** (already implemented and labelled `pymc_vi`) as a labelled
   reference where NUTS cannot certify its own uncertainty.

Until at least one of these lands, the reference's *skill* numbers are reportable
and its *uncertainty* is not certified — which is exactly why it is not promoted.

---

## 8. Everything used: tools, libraries, commands, environment

### 8.1 Runtime dependencies (operational — unchanged, `requirements.txt`)

```text
numpy>=1.26      pandas>=2.2      pyarrow>=15
fastapi>=0.110   uvicorn>=0.27    pydantic>=2.5
pytest>=8.0      httpx>=0.27
```

### 8.2 Optional extra (the offline reference only — `requirements-pymc.txt`)

```text
pymc>=5.10   arviz>=0.17   pytensor>=2.20   numba>=0.59   threadpoolctl>=3.1
```

Installed versions in this workspace: Python **3.11**, PyMC **5.28.5**, PyTensor
(with the Numba backend), numba, threadpoolctl **3.7.0**, NumPy, pandas, pyarrow,
FastAPI, pydantic, pytest, httpx.

### 8.3 Commands

```bash
# the operational spine
python scripts/prepare_dataset.py --source <dka_alice_springs_solar_only_15min_2022.csv>
python scripts/run_chariot3.py --source <csv> --out runs/chariot3_alice_springs_2022
python scripts/diagnose_model.py --source <csv>

# the offline reference and the comparison
pip install -r requirements-pymc.txt
python scripts/run_pymc_reference.py \
    --source /tmp/arena-coding-uploads/dka_alice_springs_solar_only_15min_2022.csv \
    --out runs/pymc_reference --draws 250 --tune 250 --chains 2 --windows 20 --scenarios 200

# the tests
python -m pytest tests -q                 # 35 passed
python -m pytest tests -q -m pymc         # only the PyMC-marked tests
python -m pytest tests -q -m "not pymc"   # the CI matrix (PyMC absent)

# the HTTP surface
uvicorn chariot.api:app --reload --port 8000
```

### 8.4 Environment adaptations the reference module performs *before* importing PyTensor

```text
OPENBLAS_NUM_THREADS=1  OMP_NUM_THREADS=1  MKL_NUM_THREADS=1  NUMEXPR_NUM_THREADS=1
PYTENSOR_FLAGS = cxx=,linker=numba,mode=Mode,optimizer_excluding=fusion
threadpoolctl.threadpool_limits(limits=1)     (at runtime, from _configure_pytensor)
```

The operator's own `PYTENSOR_FLAGS` always win (`_pytensor_flags` only *appends*).

### 8.5 Tooling used during development (throwaway, deleted before the commit)

`scripts/_nuts_probe.py`, `_nuts_potential.py`, `_nuts_profile.py`, `_nuts_numba.py`,
`_mass_matrix.py`, `_logp_compare.py`, `_residual_check.py`, `_scan_test.py`,
`_bench_*.py`, `_compare_*.py`, `_diag_pymc.py`, `_fit_real.py`, `_inspect_pymc_linker.py`,
`_minimal_order.py`, `_profile_grad.py` — 24 diagnostic scripts, all removed; only
`prepare_dataset.py`, `run_chariot3.py`, `diagnose_model.py` and
`run_pymc_reference.py` remain.

---

## 9. What is deliberately NOT claimed

* **No weather-forecast skill.** Both models forecast the *clearness* channel one
  day ahead from the site's own history. Nothing here is a meteorological forecast
  product and neither model has an exogenous weather input.
* **No DSM / CERC readiness.** The site is a solar-only DKASC export; the battery
  and load are **declared engineering assumptions** (`load_profile_w`,
  `load_relative_sigma = 0.18`, initial SOC 50 %), not measured.
* **No load/SOC learning.** There is no load meter in the dataset; the load channel
  is a declared assumption for both models, identically.
* **The Tribunal was not loosened.** The reference scenarios go through exactly the
  same policy (`tribunal_v3.0`) and the same solver.
* **Nothing was fitted on the test split.** The seasonal harmonic, the spread
  factor, the chains and the draws were all chosen on train/validate; the test
  split is report-only.
* **The PyMC reference is not the operational model and was not promoted.**
* **The reference's posterior is not yet converged** (§7.3) — its *skill* numbers
  are reported; its *uncertainty* is not certified.

---

## 10. How to reproduce everything

```bash
git clone https://github.com/AS-Glitched/chariotmodel.git && cd chariotmodel/chariot-vscode
pip install -r requirements.txt

# 1. the operational spine on the Alice Springs 2022 slice
python scripts/prepare_dataset.py --source /path/to/dka_alice_springs_solar_only_15min_2022.csv
python scripts/run_chariot3.py --source /path/to/dka_alice_springs_solar_only_15min_2022.csv \
                               --out runs/chariot3_alice_springs_2022
# → runs/chariot3_alice_springs_2022/{manifest.json,summary.json,windows.json}

# 2. the offline reference and the comparison
pip install -r requirements-pymc.txt
python scripts/run_pymc_reference.py \
    --source /path/to/dka_alice_springs_solar_only_15min_2022.csv \
    --out runs/pymc_reference --draws 250 --tune 250 --chains 2 --windows 20 --scenarios 200
# → runs/pymc_reference/{comparison.json,summary.md}   (~25 min: the NUTS fit is ~23 min)

# 3. the tests
python -m pytest tests -q
```

Artefacts in the repository that back every number above:

```text
runs/chariot3_alice_springs_2022/manifest.json    v3 spine manifest (all layers)
runs/chariot3_alice_springs_2022/summary.json     v3 metrics + baselines
runs/chariot3_alice_springs_2022/windows.json     per-window rows
runs/pymc_reference/comparison.json               the full ops-vs-PyMC comparison
runs/pymc_reference/summary.md                    the side-by-side table
```
