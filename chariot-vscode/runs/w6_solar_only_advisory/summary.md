# W6 — Solar-only advisory mode

## Protocol

- **name**: year2022_2023
- **train**: first 60 % of the 2022 window (fit only)
- **validate**: remaining 40 % of the 2022 window (selection only)
- **test**: the whole 2023 window (report only — never fitted, never selected on)
- **split_policy**: pre_assigned (upstream protocol)
- **model note**: no model change: the advisory contract changes, the physics does not

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
| GATE: the hybrid event probabilities are absent or explicitly marked as assumed | no published SOC/grid probability without a measured load and state of charge | hybrid_assumed=True; the SOC and grid probabilities are nulled and the payload carries the note | PASS |
| GATE: the solar events are untouched | the PV event is still published | p_pv_above_certified_ac = None | FAIL |
| GATE: the advisory stays advisory-only | no control endpoint, no dispatch | unchanged — the platform publishes forecasts, confidence and event probabilities for human review | PASS |

## Claims

**Allowed**

- The solar channel, its uncertainty and the physics survival rate are validated on the site.
- The hybrid events are published as assumed, and the payload says so.

**Forbidden**

- Any claim that the load, SOC or grid forecast is validated skill on this site: there is no meter for any of them.

## Notes

- The site card declares the load profile, the battery and the starting state of charge; those are engineering assumptions, not measurements.
- The mode is the default; `--hybrid-advisory` restores the previous behaviour for a site that does have the meters.
