# Final QD-only SOP12 Results

The final experiment uses the SOP12 QD-only feature set, `N = 100`, and the
85% Q0 EOL target. Q0 is defined as the median discharge capacity over cycles
2-5: `median(QD[1:5])`.

| Experiment | Model | MAE | SMAPE | R2 |
| --- | --- | ---: | ---: | ---: |
| MATR -> MATR | Elastic Net | 265.64 | 37.34 | -0.370 |
| MATR -> MATR | XGBoost | 221.66 | 30.59 | 0.087 |
| HUST -> HUST | Elastic Net | 194.11 | 13.58 | 0.326 |
| HUST -> HUST | XGBoost | 162.05 | 11.32 | 0.384 |
| MATR -> HUST | Elastic Net | 1034.03 | 110.90 | -16.500 |
| MATR -> HUST | XGBoost | 909.32 | 87.87 | -10.476 |

HUST within-dataset prediction is the clearest success case, especially with
XGBoost. MATR within-dataset prediction is modest. MATR -> HUST transfer remains
poor, so cross-dataset domain mismatch is still the main unresolved issue.
