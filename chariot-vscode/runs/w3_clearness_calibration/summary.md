# W3 — Ops bias / hour-elevation calibration

## Protocol

- **name**: year2022_2023
- **train**: first 60 % of the 2022 window (fit only)
- **validate**: remaining 40 % of the 2022 window (selection only)
- **test**: the whole 2023 window (report only — never fitted, never selected on)
- **split_policy**: pre_assigned (upstream protocol)
- **model note**: the clearness calibration is additive and off by default; the inference code is unchanged

### Data and content hashes

| slice | content hash |
| --- | --- |
| dataset | `0a6eb0182571917b45a53221d00d95b46589eb101a316fa3445b71ca769c17cb` |

| split | first row | last row | rows |
| --- | --- | --- | --- |
| test | 2023-01-01 00:00:00+00:00 | 2023-12-31 14:00:00+00:00 | 33,827 |
| train | 2022-01-01 00:00:00+00:00 | 2022-08-07 18:00:00+00:00 | 20,263 |
| validate | 2022-08-07 18:15:00+00:00 | 2022-12-31 14:00:00+00:00 | 14,000 |

### Model

- **model_name**: chariot_hcbsswm
- **model_version**: hcbsswm_v3.0_empirical_bayes_ece118889a
- **tribunal_policy**: tribunal_v3.0
- **residual_version**: resid_solar_w_0033adc063
- **selected_w**: 0.0
- **calibration_version**: chariot_clearness_calibration_v1.0
- **selected_shrinkage**: 0.0
- **fitted_on**: train_2022 (rolling origins)
- **selected_on**: validate_2022

## Metrics

### validate_2022 (selection) — W2 baseline

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 21,312.013 | 29,150.534 | 4,557.940 | 0.843 | 360,246.617 | 0.956 | 0.930 | 40 |

### validate_2022 (selection) — W3 calibration run

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
| GATE: |bias| down on the 2022 validation window | smaller than the W2 baseline (4,558 W) | 4,558 W — unchanged: the selection kept the shrinkage at 0.0, so no correction was applied | FAIL |

> The calibration *can* remove the bias (at shrinkage 1.0 the validation bias moves from +4,776 W to -2,649 W), but it costs 12 % of the MAE, so the bias-first selection rule rejects it.

| GATE: MAE not materially worse on the 2022 validation window | not more than 2 % worse than the W2 baseline | 21,312 W — unchanged | PASS |
| The correction is a *level* error, not a tail error | diagnostic: the model's median clearness is unbiased | the median log-ratio residual is +0.03 (validation) and -0.004 (training) while the mean is -0.17: the bias lives in the tail, so a level shift trades MAE for bias | FAIL |

> A bin-wise level correction cannot fix a skewed bias without moving the rows that are already right. The machinery lands and stays off by default; the lever for a tail bias is the ensemble spread, not the level.

| Nothing fitted on 2023 | the offsets are fitted on train_2022 and the shrinkage is selected on validate_2022 | fitted_on=train_2022 (rolling origins), selected_on=validate_2022 | PASS |

## Claims

**Allowed**

- The calibration machinery works end to end and is off by default.
- The selection rejected every candidate, so the operational model is bit-identical to the W2 baseline.
- The model's clearness bias is a tail effect, not a level error.

**Forbidden**

- Any claim that W3 improved the model: it did not, and the default is unchanged.
- Any claim that the bias was removed.

## Notes

- The first implementation fitted the offsets with a plain mean; 90 rows where the plant delivered nothing (a real outage, not a gap) turned the mean log-ratio into -inf. The estimator is now a trimmed mean over the rows that are actually clearness observations, with a shrinkage prior of 50 rows and a minimum of 20 rows per bin.
- The fitting windows are spread across the whole training split; the operational evaluation windows still tile the split from its start, which is the v3 behaviour.
