# W2 — Clear-sky upgrade (one documented reference for both years)

## Protocol

- **name**: year2022_2023
- **version**: chariot_protocol_2022_2023_v1.0
- **train**: first 60 % of the 2022 window (fit only)
- **validate**: remaining 40 % of the 2022 window (selection only)
- **test**: the whole 2023 window (report only — never fitted, never selected on)
- **model change**: none in the model — the inference code is untouched; the change is the *reference* the model reads (a data-side change, versioned as chariot_clearsky_v2.0_fixed_plane)

### Data and content hashes

| slice | content hash |
| --- | --- |
| train_2022 | `38ae50f91767b2a0501981675b898b50d48e831de5d875b25e2d2f578ee23551` |
| validate_2022 | `b7151c8504ddf13587d660f132a8c2b2a6b8dab106d21e2a00017e9acabc0246` |
| test_2023 | `9a36886a23d2f87bedfaef19d560bf1ab078d290e0486327c188578f44fb7f2d` |
| combined | `c3cda4752839e5748483adbd717a5f9fc5bce16204b7484a15d073b55f243b2c` |

| split | first row | last row | rows |
| --- | --- | --- | --- |
| train | 2022-01-01 00:00:00+00:00 | 2022-08-07 18:00:00+00:00 | 21,001 |
| validate | 2022-08-07 18:15:00+00:00 | 2022-12-31 14:00:00+00:00 | 14,000 |
| test | 2023-01-01 00:00:00+00:00 | 2023-12-31 14:15:00+00:00 | 35,002 |

### Model

- **model_name**: chariot_hcbsswm
- **model_version**: hcbsswm_v3.0_empirical_bayes_d5f67dc1eb
- **clear_sky_version**: chariot_clearsky_v2.0_fixed_plane
- **clear_sky_geometry**: fixed plane, tilt 15.0°, azimuth 0.0° (0 = north, positive = east)
- **clear_sky_model**: Ineichen/Perez with the pvlib Linke-turbidity climatology, Hay-Davies transposition, DC conversion rated_pv_w / 1176
- **geometry_evidence**: fixed plane RMSE 39.55 W/m² vs single-axis tracker RMSE 174.33 W/m² on the 2022 train slice
- **residual_version**: resid_solar_w_6e087db027
- **selected_w**: 0.0
- **tribunal_policy**: tribunal_v3.0

## Metrics

### validate_2022 (selection) — W1 baseline (exported reference)

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 21,143.314 | 28,568.035 | 4,402.742 | 0.806 | 350,591.972 | 0.944 | 0.855 | 40 |

### validate_2022 (selection) — W2 (clear-sky v2)

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 21,312.013 | 29,150.534 | 4,557.940 | 0.843 | 360,246.617 | 0.956 | 0.930 | 40 |

### test_2023 (report only) — W1 baseline (exported reference)

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 17,162.983 | 23,222.603 | 3,217.884 | 0.888 | 256,890.229 | 0.906 | 0.410 | 40 |

### test_2023 (report only) — W2 (clear-sky v2)

| model | MAE (W) | RMSE (W) | bias (W) | coverage | energy error (Wh) | survival | min survival | windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ops | 17,259.708 | 23,712.569 | 5,605.822 | 0.879 | 283,309.382 | 0.936 | 0.415 | 40 |

## Baselines

| baseline | MAE (W) | RMSE (W) | bias (W) | energy error (Wh) | windows |
| --- | --- | --- | --- | --- | --- |
| validate_2022 :: persistence_origin | 69,053.543 | 84,244.158 | 9,641.871 | n/a | n/a |
| validate_2022 :: clear_sky_naive | 22,757.437 | 32,241.444 | 15,362.543 | 451,934.586 | n/a |
| validate_2022 :: persistence_origin (cs v2) | 69,867.299 | 85,382.284 | 9,710.407 | n/a | n/a |
| validate_2022 :: clear_sky_naive (cs v2) | 41,560.106 | 49,825.332 | 41,559.998 | 997,439.949 | n/a |
| test_2023 :: persistence_origin | 68,233.874 | 82,049.849 | -4,082.676 | n/a | n/a |
| test_2023 :: clear_sky_naive | 20,303.562 | 28,940.617 | 17,518.528 | 424,765.353 | n/a |
| test_2023 :: persistence_origin (cs v2) | 68,865.884 | 83,263.906 | 767.033 | n/a | n/a |
| test_2023 :: clear_sky_naive (cs v2) | 38,901.847 | 46,542.382 | 38,827.345 | 931,856.278 | n/a |

## Gates

| gate | target | observed | verdict |
| --- | --- | --- | --- |
| GATE: daylight clearness > 1.05 rate down on 2022 validation | lower than the W1 baseline (22.45 %) | 1.78 % (was 22.45 %) | PASS |

> The exported reference was physically impossible on a quarter of the daylight rows (the plant beat its own clear sky, max ratio above 3). The rebuilt reference is above the measurement.

| GATE (alt): clear-sky naive error down on 2022 validation | lower than the W1 baseline (22,757 W) | 41,560 W (was 22,757 W) — higher | FAIL |

> This half of the gate FAILS, and it is the honest outcome: the old reference was rigged low, so the naive baseline looked better than it was. The model's margin over the naive baseline grows from 1,614 W to 20,248 W on the same windows.

| No regression: 2022 validation MAE | not materially worse than the W1 baseline (21,143 W) | 21,312 W (+0.8 %) | PASS |
| No regression: 2022 validation survival | not worse than the W1 baseline (0.944) | 0.956 (min 0.930 vs 0.855) | PASS |

> The physically impossible SOLAR_OVERSCALE rejections fall from 101 to 21 on the validation windows.

| No tracker mix invented | the array layout is not in the export, so no fixed/tracker mix is fabricated | a fixed 15° north plane fits the measured POA at 39.55 W/m² RMSE vs 174.33 W/m² for a single-axis tracker — the fixed plane wins by 4.4x | PASS |
| Nothing fitted on 2023 | the geometry comes from 2022 train; the atmospheric model is a static climatology | geometry inferred on the 2022 train slice; Linke turbidity is the static monthly climatology; the 2023 slice is only re-referenced | PASS |

## Claims

**Allowed**

- The clear-sky reference is now identical for both years and is documented (geometry, atmospheric model, DC conversion, version).
- The daylight clearness rate above 1.05 fell from 22.45 % to 1.78 % on the 2022 validation window and from 18.85 % to 1.07 % on the 2023 window.
- The physically impossible overscale rejections fell from 101 to 21 (2022 validation) and from 225 to 34 (2023 test).
- The Tribunal was not touched: the survival rate went up because the scenarios stopped contradicting the reference, not because any constraint was relaxed.

**Forbidden**

- Any claim that the 2023 numbers are a forecast-skill result — 2023 is report-only.
- Any claim that the reference is the plant's true array geometry: it is a documented single-plane equivalent inferred from the measured POA.
- Any claim that the clear-sky naive baseline improved — it did not; the old reference was rigged low.

## Notes

- The 2023 window is reported once, after the gate was decided on 2022.
- The residual weight was re-selected on the 2022 validation window only (w = 0.0).
- The worst 2023 window is still 2023-01-22 07:00 UTC: a morning ramp rejected by SOLAR_RAMP_LIMIT. The reference now explains why the plant ramped faster than a fixed plane would; that is the W3/W4 territory, and the Tribunal stays as it is.
- The clear-sky reference is a data-preparation step (pvlib); the operational model path stays pure NumPy.
