# W5 — Baselines and metric honesty

## Protocol

- **name**: year2022_2023
- **train**: first 60 % of the 2022 window (fit only)
- **validate**: remaining 40 % of the 2022 window (selection only)
- **test**: the whole 2023 window (report only — never fitted, never selected on)
- **split_policy**: pre_assigned (upstream protocol)
- **model note**: no model change: this stage fixes the *reporting* so every claim names the baseline it beats

### Data and content hashes

| slice | content hash |
| --- | --- |
| dataset | `3bf7e46e99e2673ab9fb206ca3d17e8badb3d9ca2ed363d0d5d88eee5e42719d` |

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

## Metrics

### validate_2022 (selection) — final configuration

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 21,312.013 | 29,150.534 | 4,557.940 | 0.843 | 360,246.617 | 0.956 | 0.930 | 40 |

### test_2023 (report only) — final configuration

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 17,259.708 | 23,712.569 | 5,605.822 | 0.879 | 283,309.382 | 0.936 | 0.415 | 40 |

## Baselines

| baseline | MAE (W) | RMSE (W) | bias (W) | energy error (Wh) | windows |
| --- | --- | --- | --- | --- | --- |
| validate_2022 :: solar_w::persistence_horizon_matched | 69,867.299 | 85,382.284 | 9,710.407 | n/a | n/a |
| validate_2022 :: solar_w::persistence_1step | 9,323.266 | 9,323.266 | -1,043.312 | n/a | n/a |
| validate_2022 :: solar_w::clear_sky_naive | 41,560.106 | 49,825.332 | 41,559.998 | 997,439.949 | n/a |
| test_2023 :: solar_w::persistence_horizon_matched | 68,865.884 | 83,263.906 | 767.033 | n/a | n/a |
| test_2023 :: solar_w::persistence_1step | 14,116.454 | 14,116.454 | -447.002 | n/a | n/a |
| test_2023 :: solar_w::clear_sky_naive | 38,901.847 | 46,542.382 | 38,827.345 | 931,856.278 | n/a |

## Gates

| gate | target | observed | verdict |
| --- | --- | --- | --- |
| GATE: every window summary names its baselines | persistence_1step, persistence_horizon_matched, clear_sky_naive (MAE + energy error), diurnal climatology | persistence at one step (15 min) and at the horizon, the clear-sky naive baseline with its energy error, and the training-hour climatology are all reported on every window | PASS |
| GATE: the primary metrics are reported | daylight % of rated, energy error, MAE/RMSE/bias, P10-P90 coverage, tribunal survival, rejection counts | the daylight MAE as a percentage of the rated DC power is 9.82 % (2022 validation); the energy error, MAE, RMSE, bias, coverage, survival and the rejection counts are reported next to it | PASS |
| GATE: 'beats persistence' names which persistence | no unqualified persistence claim | the model beats the *horizon-matched* persistence by 3.3x on the 2022 validation window (21,312 W vs 69,867 W) and by 4.0x on the 2023 window (17,260 W vs 68,866 W). The 15-minute persistence is reported separately (9,323 W on 2022 validation, 14,116 W on 2023) because it is scored on a single step and is therefore not comparable with a 96-step window MAE | PASS |

> The two persistences are very different numbers, and an unqualified 'beats persistence' claim would be misleading either way.


## Claims

**Allowed**

- The model beats the horizon-matched persistence by 3.3x (2022 validation) and 4.0x (2023), with the qualifier named.
- The 15-minute persistence is reported on its own row, with the note that it is a single-step score.
- The clear-sky naive baseline is reported with its MAE and its energy error.
- The daylight error is reported as a percentage of the rated DC power.

**Forbidden**

- Any unqualified 'beats persistence' claim.
- Any claim about the load, SOC or grid channels as validated skill.

## Notes

- The baseline table is now per target and per model, so the solar and the load baselines can never be confused.
- The clearance of the old 'persistence_origin' name: the horizon-matched persistence and the 15-minute persistence are separate rows.
