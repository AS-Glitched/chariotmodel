# W1 — data + the 2022→2023 protocol (QC)

Protocol: **2022 train → 2022 validate → 2023 test**.

Rows: train_2022 21001, validate_2022 14000, test_2023 35002.

Content hashes: `train_2022` 38ae50f91767b2a0…, `validate_2022` b7151c8504ddf135…, `test_2023` 9a36886a23d2f87b…, `combined` c3cda4752839e574…

| slice | rows | status | cadence | night solar max | clear-sky distinct | daylight kt>1.05 | daylight kt>1.20 | max solar |
|---|---|---|---|---|---|---|---|---|
| train_2022 | 21001 | ok | ok | 0.000 W | ok | 0.90 % | 0.86 % | 191903 W (0.872× rated) |
| validate_2022 | 14000 | ok | ok | 0.000 W | ok | 1.78 % | 1.63 % | 202564 W (0.921× rated) |
| test_2023 | 35002 | ok | ok | 0.000 W | ok | 1.07 % | 0.98 % | 204925 W (0.931× rated) |

## train_2022

- rows: **21001**, gaps flagged: 747, NaN solar: 738
- cadence: duplicates 0, non-15-min steps 0, monotonic True
- night: 10768 rows, max solar 0.0000 W, non-zero rows 0
- clear sky distinct from solar: identical=False, max |Δ| 215985.1 W, corr 0.9471
- clearness ratio on 9852 daylight rows: p05 0.0557, p50 0.7606, p95 0.9231, max 15.1345
- daylight solar above clear sky: **0.90 %** above 1.05×, **0.86 %** above 1.20×
- energy: max measured 191903 W (0.872× rated), max clear sky 228157 W

## validate_2022

- rows: **14000**, gaps flagged: 2, NaN solar: 0
- cadence: duplicates 0, non-15-min steps 0, monotonic True
- night: 6652 rows, max solar 0.0000 W, non-zero rows 0
- clear sky distinct from solar: identical=False, max |Δ| 214359.9 W, corr 0.9064
- clearness ratio on 7348 daylight rows: p05 0.0411, p50 0.7284, p95 0.9173, max 12.2541
- daylight solar above clear sky: **1.78 %** above 1.05×, **1.63 %** above 1.20×
- energy: max measured 202564 W (0.921× rated), max clear sky 228170 W

## test_2023

- rows: **35002**, gaps flagged: 1464, NaN solar: 1175
- cadence: duplicates 0, non-15-min steps 0, monotonic True
- night: 17424 rows, max solar 0.0000 W, non-zero rows 0
- clear sky distinct from solar: identical=False, max |Δ| 211977.4 W, corr 0.9341
- clearness ratio on 16936 daylight rows: p05 0.0426, p50 0.7522, p95 0.9124, max 14.7641
- daylight solar above clear sky: **1.07 %** above 1.05×, **0.98 %** above 1.20×
- energy: max measured 204925 W (0.931× rated), max clear sky 228171 W

## Notes

- The two years use **different clear-sky models**: the 2022 export was built with a 15 degree tilt, the 2023 export with a 17.5 degree tilt (both single-plane equivalents inferred from the measured POA).  That is a change of the *reference*, not of the site, and it is the first suspect for the year-to-year shift in the clearness ratio.
- The clearness ratio above 1.0 (the 'clear-sky under-binding') is the known defect this stage quantifies; W2 (the clear-sky upgrade) is gated on the 2022 validation slice and the 2023 slice is reported once.
- The load, battery and SOC channels are **declared** site-card assumptions; nothing in this stage measures them.
