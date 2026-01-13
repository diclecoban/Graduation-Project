# Graduation-Project

## Eight-Feature Pipeline (Batch 1)

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat` -> `data/raw/batch1.pkl` -> `data/intermediate/features_top8_cycles.csv` (138 rows, 46 cells x 3 early-cycle windows).
- Feature list: `IR_delta`, `dQd_slope`, `Qd_mean`, `IR_slope`, `Tavg_mean`, `IR_mean`, `Qd_std`, `IR_std`. Dropping Qd_std simply removes that column from the same CSV.
- Each model trains with an 80/20 split inside every `n_cycles in {25, 50, 100}` slice. Metrics are stored in `outputs/results/results_top8_metrics.json` and summarized below.

### Dataset overview

| Dataset | Cells | Rows (25/50/100) | Signals per row |
|---------|-------|------------------|-----------------|
| Batch 1 (2017-05-12) | 46 | 138 | 8 |
| Batch 2 (2018-02-20) | 47 | 141 | 8 |
| Combined | 93 | 279 | 8 |

Sample curves from the raw MAT data:

- Voltage vs time for two discharge cycles of cell `b1c0`: `plots/sample_voltage_curves.png`
- Capacity fade trajectories for representative cells from Batch 1 and Batch 2: `plots/sample_capacity_fade.png`

### Model results: with vs without Qd_std

| Model | n_cycles      | MAE (Qd_std var) | R2 (Qd_std var) | MAE (Qd_std yok) | R2 (Qd_std yok) |
|-------|----------     |------------------|-----------------|------------------|-----------------|
| Random Forest | 25    | 123.21 | 0.04 | 113.21 | 0.18                                         |
| Random Forest | 50    | 95.42 | 0.43 | 93.40 | 0.48                                           |
| Random Forest | 100   | 78.13 | 0.62 | 70.56 | 0.69                                          |
| XGBoost | 25 | 125.15 | -0.08 | 136.16 | -0.17                                             |
| XGBoost | 50 | 70.52  | 0.63 | 71.47 | 0.63                                                 |
| XGBoost | 100 | 58.55 | 0.73 | 63.12 | 0.71                                                |
| CatBoost | 25 | 130.09 | -0.36 | 136.01 | -0.42                                            |
| CatBoost | 50 | 124.98 | 0.06 | 102.11 | 0.34                                              |
| CatBoost | 100 | 76.31 | 0.59 | 79.12 | 0.56                                               |

### MAE vs n_cycles

| Model | 25 (var) | 50 (var) | 100 (var) | 25 (yok) | 50 (yok) | 100 (yok) |
|-------|----------|----------|-----------|----------|----------|-----------|
| Random Forest | 123.21 | 95.42 | 78.13 | 113.21 | 93.40 | 70.56 |
| XGBoost | 125.15 | 70.52 | 58.55 | 136.16 | 71.47 | 63.12 |
| CatBoost | 130.09 | 124.98 | 76.31 | 136.01 | 102.11 | 79.12 |

### MAPE vs n_cycles

| Model | 25 (var) | 50 (var) | 100 (var) | 25 (yok) | 50 (yok) | 100 (yok) |
|-------|----------|----------|-----------|----------|----------|-----------|
| Random Forest | 14.46% | 10.80% | 9.05% | 13.24% | 10.58% | 8.12% |
| XGBoost | 14.83% | 8.12% | 6.60% | 16.15% | 8.22% | 7.22% |
| CatBoost | 15.12% | 13.82% | 8.47% | 16.16% | 11.54% | 9.16% |

### Single table: MAE | MAPE (var vs yok)

| Model | 25 | 50 | 100 |
|-------|----|----|-----|
| Random Forest | var: 123.21 / 14.46%<br>yok: 113.21 / 13.24% | var: 95.42 / 10.80%<br>yok: 93.40 / 10.58% | var: 78.13 / 9.05%<br>yok: 70.56 / 8.12% |
| XGBoost | var: 125.15 / 14.83%<br>yok: 136.16 / 16.15% | var: 70.52 / 8.12%<br>yok: 71.47 / 8.22% | var: 58.55 / 6.60%<br>yok: 63.12 / 7.22% |
| CatBoost | var: 130.09 / 15.12%<br>yok: 136.01 / 16.16% | var: 124.98 / 13.82%<br>yok: 102.11 / 11.54% | var: 76.31 / 8.47%<br>yok: 79.12 / 9.16% |

### Naive baseline performance (test split)

| Baseline | MAE | R2 | MAPE (%) | SMAPE (%) |
|----------|-----|----|----------|-----------|
| Mean predictor | 243.48 | -0.00 | 38.49 | 34.70 |
| Batch-only predictor | 166.75 | 0.45 | 26.70 | 24.04 |
| Cycle-count-only predictor | 243.48 | -0.00 | 38.49 | 34.70 |

### Figures and tables

- MAPE trends: `plots/mape_random_forest.png`, `plots/mape_xgboost.png`, `plots/mape_catboost.png`
- Table snapshots (MAE | MAPE): `plots/table_random_forest.png`, `plots/table_xgboost.png`, `plots/table_catboost.png`
- Aggregated metrics tables: `plots/table_results_mae_r2.png` (MAE/R²) and `plots/table_results_mape_smape.png` (MAPE/SMAPE) covering all models and Qd_std settings.
- CatBoost CV selection summary: `plots/table_catboost_cv_selected.png` (best hyperparameters per slice + test metrics) and hold-out vs CV comparison: `plots/table_holdout_vs_cv.png`.
- Conformal prediction plots: `plots/conformal_random_forest.png`, `plots/conformal_xgboost.png`, `plots/conformal_catboost.png`, `plots/conformal_elastic_net.png`.
- Feature vs cycle coverage: `plots/table_features_cycles.png`
- Feature importance per model/cycle plus combined mosaic: `plots/feature_importance_<model>_<cycles>.png`, `plots/feature_importance_combined.png`

### Notes

- Removing Qd_std improves MAE and MAPE for RF and XGB at 100 cycles, while CatBoost still benefits from keeping it for the early windows.
- Only 46 samples exist per cycle slice (138 total), so the models are variance-limited; adding more batches would make the comparison more stable.
- Even without Qd_std, `IR_delta`, `IR_slope`, and `Tavg_mean` stay dominant at 100 cycles, whereas Qd_std matters most in the 25-50 cycle windows.

### Train / validation / test splits

- Use `python 2_modeling_featuring/split_train_val_test.py` to generate deterministic CSVs for each split. By default the script shuffles `cell_id` values with seed 42 and writes `data/splits/features_top8_cycles_{train,val,test}.csv`.
- All `n_cycles` windows belonging to the same cell stay together, so the validation/test metrics reflect performance on unseen batteries rather than unseen windows from the same battery.
- Point notebooks or training scripts to these split files to keep evaluation consistent with the cross-validation and hold-out reports.

### Cross-validation

- The script `2_modeling_featuring/cross_validate_models.py` evaluates Random Forest, XGBoost, CatBoost, and ElasticNet with grouped 5-fold cross-validation (grouped by `cell_id`) so that every battery is held out entirely in at least one fold.
- Running `python 2_modeling_featuring/cross_validate_models.py` produces `outputs/results/results_top8_cv_metrics.json`, which stores the mean +/- std of MAE, R², MAPE, and SMAPE for each model / feature set / `n_cycles` slice.
- Use these CV aggregates—together with the deterministic train/val/test splits—to report performance with lower variance than a single 80/20 hold-out.

### CV-guided final models

- `python 2_modeling_featuring/train_catboost_cv_selected.py` combines the train and val CSVs, runs grouped CV over a small CatBoost hyperparameter grid (depth, learning_rate, iterations, l2_leaf_reg), picks the best configuration per `n_cycles`, and then retrains on train+val to score the held-out test cells.
- The script saves `outputs/results/results_catboost_cv_selected.json` and a companion table `plots/table_catboost_cv_selected.png` summarizing the chosen hyperparameters, CV MAE (mean +/- std), and final test MAE/R²/MAPE/SMAPE for both feature sets (with vs without Qd_std).
- Use this output to quote statements such as “CV ile seçilen CatBoost (depth=4, lr=0.05) 100 döngüde test SMAPE ≈ 13%” when comparing with future models.

## Appendices

### Additional Figures

- `plots/sample_voltage_curves.png`: Example discharge voltage vs time curves (cycles 6 and 96 for cell b1c0).
- `plots/sample_capacity_fade.png`: Capacity trajectories for representative Batch 1 and Batch 2 cells.
- Per-model MAE/MAPE tables and importance plots under `plots/`.
- `plots/naive_baselines_metrics.png`: Bar charts of mean/batch-only/cycle-count-only predictors (MAE, R², MAPE, SMAPE).

### Detailed Results Tables

- `outputs/results/results_top8_metrics.json`: Hold-out MAE/R²/MAPE for RandomForest, XGBoost, CatBoost (with/without Qd_std, per window).
- `outputs/results/results_top8_cv_metrics.json`: Grouped 5-fold CV mean +/- std for MAE/R²/MAPE/SMAPE (all four models).
- `outputs/results/results_catboost_cv_selected.json`: CV-selected CatBoost hyperparameters plus final train+val→test metrics.
- `outputs/results/results_conformal_<model>.json`: per-model conformal quantiles and coverage values (train on train, calibration on val, evaluated on test).

### Hyperparameter Configurations

- RandomForest: 400 estimators, min_samples_leaf=2, max_features=√, random_state=42.
- XGBoost: 800 trees, max_depth=6, learning_rate=0.03, subsample/colsample=0.8, reg_lambda=2, reg_alpha=1.
- CatBoost (baseline): iterations=400, learning_rate=0.05, depth=6, loss=MAE. CV-selected variants listed in `outputs/results/results_catboost_cv_selected.json`.
- ElasticNet: StandardScaler + ElasticNetCV with l1_ratio ∈ {0.1,0.5,0.9}, α grid logspace(1e-4,1), max_iter=10k.

### Complete Feature List

1. `IR_delta` – difference between end and start IR over the window.
2. `dQd_slope` – slope of discharge capacity sequence (proxy for dQ/dV shifts).
3. `Qd_mean`
4. `Qd_std`
5. `IR_slope`
6. `Tavg_mean`
7. `IR_mean`
8. `IR_std`
(Raw PKL files also contain per-cycle `I`, `V`, `Qd`, `T`, `dQ/dV` arrays and policy metadata.)

### Additional Experiments

- dQ/dV-derived feature sets (`dqdv_features.csv`) explored in `ExtractDQdVFeatures.ipynb`.
- ElasticNet baseline with StandardScaler assessed via `plot_mape_vs_cycles.py`.
- Hold-out vs CV comparison script (`make_holdout_vs_cv_table.py`) to quantify variance reduction.
- Conformal prediction intervals generated via `make_conformal_predictions.py` for each model.
