# Advisor Conformal-Prediction Comparison

| Direction    | Setting               | Model         | Confidence | k target | k adapter | Coverage | Median width | R2      | Runs |
| ------------ | --------------------- | ------------- | ---------- | -------- | --------- | -------- | ------------ | ------- | ---- |
| HUST -> MATR | Source CP             | catboost      | 0.900      |          |           | 0.270    | 918.8        | -3.065  | 5    |
| HUST -> MATR | Source CP             | random_forest | 0.900      |          |           | 0.305    | 941.1        | -2.403  | 5    |
| HUST -> MATR | CORAL-after-source CP | coral_mlp     | 0.900      | 0        |           | 0.526    | 1010.7       | -1.514  | 5    |
| HUST -> MATR | Target-domain CP      | catboost      | 0.900      | 20       |           | 0.903    | 2074.8       | -3.104  | 100  |
| HUST -> MATR | Target-domain CP      | random_forest | 0.900      | 20       |           | 0.908    | 1904.4       | -2.374  | 100  |
| HUST -> MATR | Target-adapted CP     | catboost      | 0.900      | 20       | 20        | 0.905    | 1302.3       | -0.020  | 100  |
| HUST -> MATR | Target-adapted CP     | random_forest | 0.900      | 20       | 20        | 0.909    | 1281.4       | -0.027  | 100  |
| MATR -> HUST | Source CP             | catboost      | 0.900      |          |           | 0.190    | 1118.6       | -9.995  | 5    |
| MATR -> HUST | Source CP             | random_forest | 0.900      |          |           | 0.148    | 954.6        | -10.929 | 5    |
| MATR -> HUST | CORAL-after-source CP | coral_mlp     | 0.900      | 0        |           | 0.296    | 1232.8       | -9.412  | 5    |
| MATR -> HUST | Target-domain CP      | catboost      | 0.900      | 20       |           | 0.902    | 2514.0       | -9.912  | 100  |
| MATR -> HUST | Target-domain CP      | random_forest | 0.900      | 20       |           | 0.885    | 2568.1       | -11.089 | 100  |
| MATR -> HUST | Target-adapted CP     | catboost      | 0.900      | 20       | 20        | 0.908    | 998.7        | -0.194  | 100  |
| MATR -> HUST | Target-adapted CP     | random_forest | 0.900      | 20       | 20        | 0.907    | 1057.2       | -0.407  | 100  |
