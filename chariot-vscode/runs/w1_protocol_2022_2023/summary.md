# W1 — Data + protocol (2022 → 2023)

## Protocol

- **name**: year2022_2023
- **version**: chariot_protocol_2022_2023_v1.0
- **description**: TRAIN = the first 60 % of the 2022 window; VALIDATE = the remaining 40 % of the 2022 window; TEST = the whole 2023 window (report only, never fitted, never selected on).  The 2022 test tail of the legacy 60/20/20 protocol is discarded on purpose.
- **train**: first 60 % of the 2022 window (fit only)
- **validate**: remaining 40 % of the 2022 window (selection only)
- **test**: the whole 2023 window (report only — never fitted, never selected on)
- **model change**: none — the v3 ops spine is untouched (W1 is data and protocol only)

### Data and content hashes

| slice | content hash |
| --- | --- |
| train_2022 | `70713449bca00edb28263a6da58b6e567e4b01b5b51aa505d15d036bc764aa49` |
| validate_2022 | `052c1d161cb53e940df286a90c75ba296505c191e075dbfee340e330986d760a` |
| test_2023 | `6dc97430d82a10a2116ecbd22cc70eb4b9980a68608dc3f7a9e4b17ecf6c4acf` |
| combined | `17e207367329b7af8b58c2b9bc52c5daadec97f95cc57e52499ce818bffaabbc` |

| split | first row | last row | rows |
| --- | --- | --- | --- |
| train | 2022-01-01 00:00:00+00:00 | 2022-08-07 18:00:00+00:00 | 21,001 |
| validate | 2022-08-07 18:15:00+00:00 | 2022-12-31 14:00:00+00:00 | 14,000 |
| test | 2023-01-01 00:00:00+00:00 | 2023-12-31 14:15:00+00:00 | 35,002 |

### Model

- **model_name**: chariot_hcbsswm
- **model_version**: hcbsswm_v3.0_empirical_bayes_b979cd9917
- **clear_sky_version**: source export clear-sky (2022: 15° tilt; 2023: 17.5° tilt) — unchanged in W1, upgraded in W2
- **residual_version**: resid_solar_w_ade0a65c7f
- **selected_w**: 0.11
- **tribunal_policy**: tribunal_v3.0

## Metrics

### validate_2022 (selection)

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 21,143.314 | 28,568.035 | 4,402.742 | 0.806 | 350,591.972 | 0.944 | 0.855 | 40 |

### test_2023 (report only)

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 17,162.983 | 23,222.603 | 3,217.884 | 0.888 | 256,890.229 | 0.906 | 0.410 | 40 |

## Baselines

| baseline | MAE (W) | RMSE (W) | bias (W) | energy error (Wh) | windows |
| --- | --- | --- | --- | --- | --- |
| validate_2022 :: persistence_origin | 69,053.543 | 84,244.158 | 9,641.871 | n/a | n/a |
| validate_2022 :: clear_sky_naive | 22,757.437 | 32,241.444 | 15,362.543 | 451,934.586 | n/a |
| test_2023 :: persistence_origin | 68,233.874 | 82,049.849 | -4,082.676 | n/a | n/a |
| test_2023 :: clear_sky_naive | 20,303.562 | 28,940.617 | 17,518.528 | 424,765.353 | n/a |

## Gates

| gate | target | observed | verdict |
| --- | --- | --- | --- |
| 2022 train / validate / 2023 test boundaries are exact | train < validate < test, test entirely in 2023 | train 2022-01-01→2022-08-07, validate 2022-08-07→2022-12-31, test 2023-01-01→2023-12-31 | PASS |
| QC: night solar ≈ 0 | max night solar = 0 W in every slice | train 0.000 W, validate 0.000 W, test 0.000 W | PASS |
| QC: strict monotonic 15-minute cadence | 0 duplicates, 0 non-15-min steps | 0 duplicates and 0 non-15-min steps in all three slices | PASS |
| QC: clear-sky reference is a distinct series | clear_sky_w ≠ solar_w | max |Δ| 192.6 kW (2022) / 188.3 kW (2023), correlation 0.90–0.95 | PASS |
| QC: daylight clearness > 1.05 quantified | reported per slice | train 27.68 %, validate 22.45 %, test 18.85 % | PASS |
| QC: daylight clearness > 1.20 quantified | reported per slice | train 5.20 %, validate 4.97 %, test 1.29 % | PASS |
| No model change in W1 | the ops default stays the versioned NumPy path | ops model_version unchanged (hcbsswm_v3.0_empirical_bayes_b979cd9917); only the data/protocol changed | PASS |
| The clear-sky reference differs between the two years | diagnostic (not a gate): a single consistent clear-sky geometry | 2022 export used a 15° tilt, the 2023 export a 17.5° tilt (single-plane equivalents) → the year-over-year clearness shift is a clear-sky artefact, not a plant artefact | FAIL |

> This is the W2 entry point. W1 reports it; W1 does not repair it, and nothing is fitted to the 2023 error.


## Claims

**Allowed**

- The 2022 → 2023 protocol is reproducible from the two source exports (content hashes and exact boundaries are in the manifest).
- The QC numbers for both years are measured, not assumed.
- The ops v3 spine runs unchanged on the new protocol and its 2023 numbers are reported once, as a report-only baseline.
- The 2022 validation numbers are the selection metrics for every later stage.

**Forbidden**

- Any claim of forecast skill on 2023 — 2023 is a report-only window.
- Any claim that the 2022 and 2023 clear-sky references are the same geometry.
- Any weather-forecast or DSM/CERC readiness claim.
- Any control action: the platform remains advisory-only.

## Notes

- Both exports carry 35,040 rows on a strict 15-minute grid with no duplicate timestamps; 2022 has 749 `is_gap` rows and 2023 has 1,464. `solar_w` is NaN on the gap rows, so the QC masks non-finite values.
- The worst 2023 window is 2023-01-22 07:00 UTC with a survival rate of 0.41, dominated by 115 SOLAR_RAMP_LIMIT rejections: a clear-sky ramp artefact, which is what W2 targets. The Tribunal is not relaxed.
- The 2023 daylight clearness rate above 1.05 (18.85 %) is already lower than the 2022 rate (27.68 %) because the 2023 export uses a 17.5° tilt — the shift between the years is a reference change, not a change in the plant.
- Residual selection (w = 0.11) is unchanged: it is selected on the 2022 validation split only.
