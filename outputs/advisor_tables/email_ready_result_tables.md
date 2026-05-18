Top-k Cross-Dataset Results

| k  | Direction    | Best model       | Best R2 | MAE   |
| -- | ------------ | ---------------- | ------- | ----- |
| 3  | HUST -> MATR | elastic_net      | -0.691  | 334.9 |
| 3  | MATR -> HUST | gaussian_process | -8.132  | 781.5 |
| 6  | HUST -> MATR | elastic_net      | -1.912  | 325.9 |
| 6  | MATR -> HUST | catboost         | -7.971  | 757.7 |
| 10 | HUST -> MATR | catboost         | -2.932  | 663.1 |
| 10 | MATR -> HUST | gaussian_process | -7.948  | 771.5 |
| 12 | HUST -> MATR | elastic_net      | -1.691  | 347.7 |
| 12 | MATR -> HUST | gaussian_process | -8.000  | 774.1 |

CORAL + Target-Side Calibration Results

| Direction    | Setting                    | k target | R2     | MAE   | sMAPE | Runs |
| ------------ | -------------------------- | -------- | ------ | ----- | ----- | ---- |
| HUST -> MATR | CORAL only                 | 0        | -1.514 | 491.7 | 54.35 | 5    |
| HUST -> MATR | CORAL + target calibration | 20       | -0.198 | 308.7 | 39.64 | 100  |
| MATR -> HUST | CORAL only                 | 0        | -9.412 | 811.5 | 74.77 | 5    |
| MATR -> HUST | CORAL + target calibration | 20       | -0.867 | 305.6 | 20.61 | 100  |
