# CHARIOT Laboratory Report

**Project:** CHARIOT — physics-governed hierarchical clear-sky Bayesian state-space advisory spine  
**Report type:** Lab evolution (version-by-version)  
**Environment:** Local Windows lab (`chariot-vscode`), Python 3.12  
**Primary dataset:** `data/site_1_30d_15min.csv` (synthetic 15-minute site series, ~30 days)  
**Report date:** 2026-10-19  
**Status of accuracy work on synthetic 30d:** Closed at v1.7; multi-horizon E1 started; schema v2 frozen as future target  

---

## 1. Executive summary

This report records the laboratory build of CHARIOT from first runnable prototype through model iterations, honest scoreboards, Monitor API, and evaluation redesign.

**What was proven**

1. End-to-end spine works: CSV → admissibility → seal → time-ordered split → HC-BSSWM → Reality Tribunal → three advisory modes.  
2. Time leakage controls pass (train end ≤ test start).  
3. Tribunal retention is stable (~93% survival on lab runs).  
4. Load diurnal profile reaches near 15-minute persistence.  
5. Coulomb SOC markedly improves on unstructured SOC.  
6. Solar structure beats naive clear-sky on energy and point metrics; it does **not** beat 15-minute persistence on open-loop point MAE.  
7. Uncertainty coverage became usable (solar ~0.87, SOC ~0.71 at v1.7).  
8. Multi-horizon evaluation (E1) shows model **beats clear-sky energy** on 24h and 72h horizons.  

**What was not claimed**

- Production forecast superiority  
- Closed-loop control  
- India-specific or market P&L guarantees  
- Beating persistence on 15-minute solar/SOC point forecasts  

---

## 2. Laboratory setup

| Item | Detail |
|------|--------|
| Package root | `chariot-vscode` |
| Core package | `chariot/` (`pipeline`, `hcbsswm`, `tribunal`, `admissibility`, `extract`, …) |
| Data | Synthetic 30-day, 15-minute; rated PV 5000 W; battery 10000 Wh |
| Sealed rows (typical) | 2876 after admissibility |
| Content hash (30d sealed) | `6a3066e0532ccff8df2bdb342ecd234317ed96f439af11055a9309df3a603031` |
| Split policy | `time_ordered_v0` (no random shuffle) |
| Advisory modes | Grid engineer, researcher, investor/planner |
| API | FastAPI Monitor on `127.0.0.1:8000` |

**Unit 0 laws enforced**

- Sealed extract + content hash  
- Time-ordered train/test only  
- Tribunal fail-closed on illegal scenarios  
- Advisory-only outputs  

---

## 3. Version-by-version evolution

### 3.0 Bootstrap — Prototype spine (pre-CSV wire)

| Item | Result |
|------|--------|
| Goal | Runnable end-to-end on synthetic 7-day frame |
| Data | In-code `make_site_frame(days=7)` |
| Issues | Windows Store `python` alias; OneDrive `Bad file descriptor` on writes |
| Fixes | Use `py` launcher; run outside OneDrive path issues |
| Outcome | `status=OK`, survival ≈ 0.9375, model `chariot_hcbsswm` |

**Evidence (7d synthetic run)**

- `n_rows` ≈ 671  
- `train_rows` 469 / `test_rows` 202  
- `solar_scale_mean` ≈ 0.685  
- Leakage check: train end 2026-07-24 21:15 ≤ test start 21:30 — **PASS**  

---

### 3.1 Wire 30-day CSV

| Item | Result |
|------|--------|
| Goal | Train/advise on sealed 30d file |
| Data source | `csv:site_1_30d_15min.csv` |
| Outcome | `n_rows=2876`, window 2026-07-20 → 2026-08-18 |
| Leakage | train end 2026-08-10 00:00 ≤ test start 00:15 — **PASS** |
| Survival | 0.9375 |

---

### 3.2 First honest solar score (single window)

**Script:** `scripts/score_vs_persistence.py`  
**Window:** first 96 test steps (one day from origin)

| Model | Solar MAE | Notes |
|-------|-----------|--------|
| Persistence 1-step | 106.1 | Strong short-horizon bar |
| Clear-sky naive | 99.3 | This day relatively clear |
| HC-BSSWM P50 (early) | **648.6** | Severe under-forecast |
| Coverage | 0.20 | Poor |
| Bias (P50−actual) | ≈ −647 W | Mean actual ~1507, P50 ~860 |

**Diagnosis:** Global train scale ~0.56 on a clearer day; night origin made naive “last solar” conditioning weak for daytime.

---

### 3.3 Model versions (HC-BSSWM)

#### v1.1 — Origin conditioning

| Change | Soft pull early path toward last observed solar |
| Result | MAE ~576; coverage collapsed toward 0 — insufficient |

#### v1.2 — Clear-sky index (`kt`) + wider scale

| Change | `last_kt`, scenario scales, night force zero |
| Result | Coverage improved (~0.5); MAE still large (~690 on stress window); `last_kt`~0.5 vs clear day |

#### v1.3 — Hour-of-day `kt` profile

| Change | Median/weighted `kt` by hour from train only |
| Single-window MAE | ~599 |
| **Rolling-origin (22 starts, 96 steps)** | **First fair board** |

**Rolling solar (v1.3 era)**

| Metric | Value |
|--------|-------|
| Mean MAE persistence | 73.1 |
| Mean MAE clear-sky | 599.3 |
| Mean MAE model P50 | **201.0** |
| Mean coverage | **0.777** |
| Mean survival | 0.933 |
| Verdict | PERSISTENCE_STRONGER_ON_AVERAGE |

**Interpretation:** Model beats clear-sky strongly; loses to 15-min persistence on open-loop paths — expected and documented.

#### v1.4 — Coulomb SOC

| Change | SOC integrated from scenario solar−load (η≈0.95, dt=0.25 h) |
| SOC MAE | 27.0 → **10.4** |
| SOC coverage | 0.019 → **0.51** |
| Bounds | 1.0 |
| Load | Still weak (~141 MAE vs persistence ~35) |

#### v1.5 — Diurnal load profile

| Change | Hour-of-day median load from train |
| Load MAE | 141 → **35.7** (persistence 34.6) |
| Load verdict | **MODEL_NEAR_PERSISTENCE** |
| SOC | MAE ~10.0, coverage ~0.54 |
| Prototype lineage | `hcbsswm_v1.5_diurnal_load`, scale ~0.59 |

**Empirical load profile (30d, UTC hour medians)**  
Night ~630–660 W → midday peak ~1050–1070 W → evening decline; global mean ~851 W.

#### v1.6 — Recency-weighted `kt` + residual blend + SOC noise

| Metric | v1.5 | v1.6 |
|--------|------|------|
| Solar MAE (rolling) | ~201 | **188.6** |
| Solar coverage | ~0.78 | **0.79** |
| SOC MAE | ~10.0 | ~9.96 |
| SOC coverage | ~0.54 | **0.58** |

Small gains; stretch targets (solar MAE≤150, SOC coverage≥0.70) not met.

#### v1.7 — Residual AR solar + wider SOC process noise (**accuracy freeze**)

| Metric | v1.6 | **v1.7** |
|--------|------|---------|
| Solar MAE | 188.6 | **191.4** |
| Solar coverage | 0.79 | **0.87** |
| SOC MAE | ~10.0 | **10.0** |
| SOC coverage | 0.58 | **0.71** |
| Load MAE | ~36.4 | **36.4** |
| Survival | 0.93 | 0.93 |

**Decision:** Stop synthetic MAE chasing. v1.7 optimizes **calibration** more than point MAE. Freeze folder: `runs/30d_v1.7_final` (when archived by operator).

#### v1.8 — Weather flag (in progress at report time)

| Change | `use_weather` on fit; cloud forecast attenuation in scenarios when `cloud_forecast_pct` present |
| Versions | `hcbsswm_v1.7_residual_ar` / `hcbsswm_v1.8_weather` |
| Status | Code path introduced; full E2 scoreboard pending successful multi-horizon weather run from project root |

---

## 4. Evaluation redesign

### 4.1 Why redesign

15-minute open-loop MAE vs 1-step persistence is a harsh and incomplete headline for a physics-scenario engine. Persistence reuses the last true meter every step; CHARIOT emits a constrained path distribution.

### 4.2 E1 — Multi-horizon (no weather)

**Script:** `scripts/score_multi_horizon.py`  
**Horizons:** H1=4h (16 steps), H2=24h (96), H3=72h (288)  
**Origins:** 20 daily starts (initial design)

**Results (initial E1)**

| Horizon | Model point MAE | Persistence | Model energy MAE (Wh) | Clear-sky energy MAE (Wh) | Coverage | Verdict |
|---------|-----------------|-------------|------------------------|---------------------------|----------|---------|
| H1 4h | 0.0* | 0.0* | 0.0* | 0.0* | 1.0* | Artifact |
| H2 24h | 192.6 | 72.8 | **4492** | 14453 | **0.86** | **Beats clear-sky energy** |
| H3 72h | 195.0 | 73.5 | **10877** | 42544 | **0.88** | **Beats clear-sky energy** |

\*H1 artifact: origins at midnight → night-only solar ≈ 0.  
\**Persistence energy MAE ≈ 0 is mathematical (lagged series nearly same integral), not a useful energy baseline.

### 4.3 E1b + E2 (specified; operator to complete)

- Origins at **06:00 UTC** for daylight-aware H1  
- Modes: `no_weather` vs `weather_forecast` (synthetic cloud forecast = truth + noise, labeled lab proxy)  
- Weather lift on H2/H3 energy MAE  

---

## 5. Monitor API and demo UI

| Component | Detail |
|-----------|--------|
| App | `chariot/monitor_app.py` |
| Run | `py -m uvicorn chariot.monitor_app:app --host 127.0.0.1 --port 8000` |
| Endpoints | `/health`, `/ready`, `GET/POST .../monitor/latest|run` |
| Limits object | Advisory-only; no controller claims |
| Demo UI | `frontend/monitor.html` |
| Lab status doc | `docs/LAB_STATUS.md` (checkpoint close) |

---

## 6. Target schema freeze (future data contract)

**ID:** `schema_v2_target` (frozen 2026-10-19)

- Core: tenant, site, UTC time, lat/lon, solar/load/SOC, ratings, gap, clear-sky  
- Strong: GHI, cloud, temperature, altitude  
- Optional: DNI/DHI, module temp, humidity, precip, wind  
- Forecast: issue time + cloud/GHI/temp forecasts (time-safe)  
- Rule: physics-universal; no India-only hard-coding  

Lab v1 columns map into v2; migration is explicit, not silent.

---

## 7. Consolidated scoreboard (final synthetic 30d story)

| Stream | Best lab result | vs bar | Coverage |
|--------|-----------------|--------|----------|
| Solar (rolling 24h path MAE) | ~191 (v1.7) | vs persistence ~73 — behind; vs clear-sky ~599 — ahead | ~0.87 |
| Solar energy 24h / 72h (E1) | Beats clear-sky energy by large margin | Primary multi-horizon claim | ~0.86–0.88 |
| Load | ~35.7–36.4 | Near persistence ~34.6 | — |
| SOC | ~10.0 | vs persistence ~1.65 — behind; coulomb vs random-walk — large gain | ~0.71 |
| Tribunal survival | ~0.93 | Stable | — |

---

## 8. Claims register

### Allowed

1. End-to-end physics-governed advisory spine operates on sealed lab data.  
2. Evaluation can be time-safe (ordered splits, rolling origins).  
3. Solar scenarios are structured vs naive clear-sky and carry usable uncertainty at v1.7.  
4. On multi-day energy error vs clear-sky, model wins on E1 H2/H3.  
5. Load diurnal structure is near short-horizon persistence on this site.  
6. SOC is coulomb-linked to power scenarios.  
7. Monitor API serves advisories with explicit limits.  

### Forbidden

1. “Best” or production-ready point forecast.  
2. Beating 15-minute persistence on solar/SOC as a general claim.  
3. Closed-loop control authority.  
4. Market profit or regulatory certification.  
5. Universality proven on real multi-climate fleets (schema supports it; not yet demonstrated).  

---

## 9. Operational runbook (lab)

```text
cd chariot-vscode
py scripts/run_prototype.py
py scripts/score_rolling_origin.py
py scripts/score_soc_rolling.py
py scripts/score_multi_horizon.py
py -m uvicorn chariot.monitor_app:app --host 127.0.0.1 --port 8000
```

Always run Python commands from `chariot-vscode` (package root), not the parent folder.

---

## 10. Conclusions

1. The lab successfully built a **fail-closed, time-safe, advisory** CHARIOT spine.  
2. Accuracy work on synthetic 30d reached **diminishing returns** on point MAE; **coverage and energy-vs-clear-sky** are the honest strengths.  
3. Further gains should come from **multi-horizon + time-safe weather** and **real Tier-A/B schema data**, not more noise knobs on the same board.  
4. Schema v2 is frozen as the **target contract** for portable, physics-first deployment.  

---

## 11. Appendix — Key artifact paths

| Path | Content |
|------|---------|
| `runs/latest/` | Latest prototype + scores |
| `runs/30d_v1.5_diurnal_load/` | Load profile freeze |
| `runs/30d_v1.6_solar_soc/` | v1.6 scores |
| `runs/30d_v1.7_final/` | Intended final synthetic accuracy freeze |
| `runs/E1_multi_horizon/` | Multi-horizon E1 JSON |
| `data/site_1_30d_15min.csv` | Lab training series |
| `docs/CHARIOT_TARGET_SCHEMA_V2.md` | Frozen target schema |
| `docs/LAB_STATUS.md` | Short checkpoint status |

---

*End of lab report.*
