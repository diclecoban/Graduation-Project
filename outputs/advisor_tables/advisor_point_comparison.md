# Advisor Point-Prediction Comparison

| Direction    | Setting                          | Model            | k  | R2     | MAE   | Note                             |
| ------------ | -------------------------------- | ---------------- | -- | ------ | ----- | -------------------------------- |
| HUST -> HUST | All features, within-dataset     | random_forest    |    | 0.340  | 178.0 | Full feature reference           |
| HUST -> HUST | Best top-k, within-dataset       | random_forest    | 3  | 0.394  | 174.0 | Compact feature subset           |
| HUST -> MATR | Raw transfer, all features       | gaussian_process |    | -2.052 | 569.2 | No target labels                 |
| HUST -> MATR | Best top-k raw transfer          | elastic_net      | 3  | -0.691 | 334.9 | Feature selection only           |
| HUST -> MATR | CORAL-only                       | coral_mlp        | 0  | -1.514 | 491.7 | Covariate alignment baseline     |
| HUST -> MATR | Target calibration, all features | stacking         | 20 | -0.049 | 283.6 | Linear target-side calibration   |
| HUST -> MATR | CORAL + target calibration       | coral_mlp        | 20 | -0.198 | 308.7 | CORAL plus residual target shift |
| MATR -> HUST | Raw transfer, all features       | gaussian_process |    | -8.125 | 781.2 | No target labels                 |
| MATR -> HUST | Best top-k raw transfer          | gaussian_process | 10 | -7.948 | 771.5 | Feature selection only           |
| MATR -> HUST | CORAL-only                       | coral_mlp        | 0  | -9.412 | 811.5 | Covariate alignment baseline     |
| MATR -> HUST | Target calibration, all features | pls              | 20 | -0.020 | 225.5 | Linear target-side calibration   |
| MATR -> HUST | CORAL + target calibration       | coral_mlp        | 20 | -0.867 | 305.6 | CORAL plus residual target shift |
| MATR -> MATR | All features, within-dataset     | catboost         |    | 0.575  | 171.7 | Full feature reference           |
| MATR -> MATR | Best top-k, within-dataset       | catboost         | 12 | 0.518  | 179.1 | Compact feature subset           |
