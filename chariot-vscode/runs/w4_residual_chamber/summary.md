# W4 — Constrained residual chamber

## Protocol

- **name**: year2022_2023
- **train**: first 60 % of the 2022 window (fit only)
- **validate**: remaining 40 % of the 2022 window (selection only)
- **test**: the whole 2023 window (report only — never fitted, never selected on)
- **split_policy**: pre_assigned (upstream protocol)
- **model note**: the residual chamber stays a bounded shadow corrector: the governor is the only runtime authority and the Tribunal still certifies every path

### Data and content hashes

| slice | content hash |
| --- | --- |
| dataset | `7cc39f8673ccb2fb4ee2183f9eea4eea33456831906ec5e635487c55b9c20607` |

| split | first row | last row | rows |
| --- | --- | --- | --- |
| test | 2023-01-01 00:00:00+00:00 | 2023-12-31 14:00:00+00:00 | 33,827 |
| train | 2022-01-01 00:00:00+00:00 | 2022-08-07 18:00:00+00:00 | 20,263 |
| validate | 2022-08-07 18:15:00+00:00 | 2022-12-31 14:00:00+00:00 | 14,000 |

### Model

- **model_name**: chariot_hcbsswm
- **model_version**: hcbsswm_v3.0_empirical_bayes_d5f67dc1eb
- **tribunal_policy**: tribunal_v3.0
- **residual_version**: resid_solar_w_0033adc063
- **selected_w**: 0.0
- **residual_family**: ridge
- **w_max**: 0.2
- **night_forcing**: the correction is forced to zero when there is no clear sky
- **features**: model means, clear-sky level, the model's own expected ratio, lags, hour/doy harmonics, the previous correction (never the same-time measurement)

## Metrics

### validate_2022 (selection) — W2 baseline

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 21,312.013 | 29,150.534 | 4,557.940 | 0.843 | 360,246.617 | 0.956 | 0.930 | 40 |

### validate_2022 (selection) — W4 residual run

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 21,312.013 | 29,150.534 | 4,557.940 | 0.843 | 360,246.617 | 0.956 | 0.930 | 40 |

## Baselines

| baseline | MAE (W) | RMSE (W) | bias (W) | energy error (Wh) | windows |
| --- | --- | --- | --- | --- | --- |
| validate_2022 :: persistence_origin | 69,867.299 | 85,382.284 | 9,710.407 | n/a | n/a |
| validate_2022 :: clear_sky_naive | 41,560.106 | 49,825.332 | 41,559.998 | 997,439.949 | n/a |

## Gates

| gate | target | observed | verdict |
| --- | --- | --- | --- |
| GATE: >= 1 % validation MAE improvement from the residual | at least 1 % better than w = 0 | 0.000 % (8,624 W at w = 0 vs 8,624 W at the best w) | FAIL |

> The ridge explains 2.5 % of the residual variance (R^2 vs the mean): the clearness innovation is not predictable at a 15-minute horizon, so there is nothing for the chamber to correct.

| GATE: cleaner bias | |bias| below the W2 baseline | 4,558 W — unchanged (w = 0) | FAIL |
| The weight stays at or near zero | w <= 0.20 and the selection picks w ~ 0 | selected w = 0.0 (ceiling 0.2) | PASS |
| The governor is the only runtime authority | the residual weight applied at run time is the governor's effective weight | the pipeline sets the governor weight from the selection and the genesis uses governor.get_effective_w(); the Tribunal certifies every path afterwards | PASS |
| Night forcing | the correction is ~0 when there is no clear sky | the residual is zeroed wherever the clear sky is below 1 W, so it cannot invent night power | PASS |

## Claims

**Allowed**

- The residual chamber is bounded, governed, Tribunal-checked and leakage-safe.
- Its weight is selected on the 2022 validation window and is now 0.
- The clearness innovation is not predictable at this horizon (R^2 = 0.025).

**Forbidden**

- Any claim that the residual improves the forecast: the selected weight is 0.
- Any claim that the residual is a control path: it never bypasses the Tribunal.

## Notes

- The chamber gained two things in W4: the residual's own lag as a feature (the correction proposed for the previous step, never the realised measurement) and a hard night forcing.
- Because the selected weight is 0, the operational scenarios are bit-identical to the W2 baseline.
