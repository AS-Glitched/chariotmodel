# W1 — data + the 2022→2023 protocol (QC)

Protocol: **2022 train → 2022 validate → 2023 test**.

Rows: train_2022 21001, validate_2022 14000, test_2023 35002.

Content hashes: `train_2022` 70713449bca00edb…, `validate_2022` 052c1d161cb53e94…, `test_2023` 6dc97430d82a10a2…, `combined` 17e207367329b7af…

| slice | rows | status | cadence | night solar max | clear-sky distinct | daylight kt>1.05 | daylight kt>1.20 | max solar |
|---|---|---|---|---|---|---|---|---|
| train_2022 | 21001 | ok | ok | 0.000 W | ok | 27.68 % | 5.20 % | 191903 W (0.872× rated) |
| validate_2022 | 14000 | ok | ok | 0.000 W | ok | 22.45 % | 4.97 % | 202564 W (0.921× rated) |
| test_2023 | 35002 | ok | ok | 0.000 W | ok | 18.85 % | 1.29 % | 204925 W (0.931× rated) |

## train_2022

- rows: **21001**, gaps flagged: 747, NaN solar: 738
- cadence: duplicates 0, non-15-min steps 0, monotonic True
- night: 10771 rows, max solar 0.0000 W, non-zero rows 0
- clear sky distinct from solar: identical=False, max |Δ| 192569.9 W, corr 0.9472
- clearness ratio on 9849 daylight rows: p05 0.1453, p50 0.9356, p95 1.2024, max 10.2180
- daylight solar above clear sky: **27.68 %** above 1.05×, **5.20 %** above 1.20×
- energy: max measured 191903 W (0.872× rated), max clear sky 205166 W

## validate_2022

- rows: **14000**, gaps flagged: 2, NaN solar: 0
- cadence: duplicates 0, non-15-min steps 0, monotonic True
- night: 6655 rows, max solar 0.0000 W, non-zero rows 0
- clear sky distinct from solar: identical=False, max |Δ| 191958.5 W, corr 0.9036
- clearness ratio on 7345 daylight rows: p05 0.0666, p50 0.9038, p95 1.1995, max 3.6977
- daylight solar above clear sky: **22.45 %** above 1.05×, **4.97 %** above 1.20×
- energy: max measured 202564 W (0.921× rated), max clear sky 203304 W

## test_2023

- rows: **35002**, gaps flagged: 1464, NaN solar: 1175
- cadence: duplicates 0, non-15-min steps 0, monotonic True
- night: 17429 rows, max solar 0.0000 W, non-zero rows 0
- clear sky distinct from solar: identical=False, max |Δ| 188298.7 W, corr 0.9360
- clearness ratio on 16931 daylight rows: p05 0.0907, p50 0.9183, p95 1.1265, max 4.8524
- daylight solar above clear sky: **18.85 %** above 1.05×, **1.29 %** above 1.20×
- energy: max measured 204925 W (0.931× rated), max clear sky 204571 W

## Notes

- The two years use **different clear-sky models**: the 2022 export was built with a 15 degree tilt, the 2023 export with a 17.5 degree tilt (both single-plane equivalents inferred from the measured POA).  That is a change of the *reference*, not of the site, and it is the first suspect for the year-to-year shift in the clearness ratio.
- The clearness ratio above 1.0 (the 'clear-sky under-binding') is the known defect this stage quantifies; W2 (the clear-sky upgrade) is gated on the 2022 validation slice and the 2023 slice is reported once.
- The load, battery and SOC channels are **declared** site-card assumptions; nothing in this stage measures them.
