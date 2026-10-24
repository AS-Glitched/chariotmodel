# Ops (NumPy) vs PyMC reference — Alice Springs 2022

Dataset `dataset_f2288a01f008` (content hash `d6712722426cea9f…`), time-ordered 60/20/20 split; train=9975 validate=3416 test=3796.

Protocol: horizon 96 steps, 200 scenarios/window, residual chamber off (both models, pure clearness comparison), selection on the validation split, test split report only.

Reference fit: backend **pymc_nuts**, chains 2, draws 250, divergences 0, rhat_max 2.429386032821015, ess_min 2.502667354288694, wall 1389.571 s.

| split | model | survival | solar MAE (W) | solar RMSE (W) | solar bias (W) | P10-P90 coverage | confidence |
|---|---|---|---|---|---|---|---|
| validate | ops (NumPy) | 0.9592 | 21,180 | 29,128 | 4,631 | 0.7766 | 0.5000 |
| validate | PyMC reference | 0.9950 | 21,014 | 29,375 | 6,354 | 0.8016 | 0.5405 |
| test | ops (NumPy) | 0.8962 | 26,050 | 35,914 | 14,968 | 0.7224 | 0.4096 |
| test | PyMC reference | 0.9760 | 26,500 | 37,001 | 18,160 | 0.6958 | 0.4870 |

### Baselines (same windows)

| split | baseline | MAE (W) | RMSE (W) | bias (W) |
|---|---|---|---|---|
| validate | clear_sky_naive | 21,166 | 30,534 | 10,698 |
| validate | persistence_origin | 61,537 | 78,476 | 16,028 |
| test | clear_sky_naive | 33,548 | 46,750 | 31,093 |
| test | persistence_origin | 72,370 | 86,570 | 15,347 |

Persistence baseline = the solar level measured at the forecast origin, held constant over the window.  Clear-sky naive = the clear-sky model of the site, used directly as the forecast.
