# Battery Cycle-Life Prediction

This project predicts battery end-of-life from early-cycle discharge-capacity
signals. The current final workflow uses a QD-only SOP12 feature set for MATR
and HUST, an EOL target defined at 85% of early-cycle Q0, and fixed
cell-level SOP JSON splits.

## Current Final Protocol

- Feature source: discharge capacity `QD` only.
- Q0 definition: `median(QD[1:5])`, corresponding to cycles 2-5 in 0-indexed
  arrays.
- Target: first cycle where `QD <= 0.85 * Q0`.
- Window: `N = 100` cycles.
- Models reported: Elastic Net and XGBoost.
- Experiments:
  - `matr_to_matr`
  - `hust_to_hust`
  - `matr_to_hust`

## SOP12 Features

The active SOP12 feature list is:

1. `Qdis_N`
2. `delta_Qdis`
3. `retention_ratio`
4. `slope_linear`
5. `variance_Qdis`
6. `range_Qdis`
7. `max_drop`
8. `std_diff`
9. `skewness_Qdis`
10. `slope_ratio`
11. `Qdis_cycle10`
12. `mean_diff`

## Active Project Layout

```text
0_data_prep/                 Raw MAT to pickle conversion
1_feature_engineering/       Active label and SOP12 feature builders
2_modeling_featuring/        Active split/modeling utilities
data/raw/                    Raw pickle inputs
data/intermediate/           Current final intermediate CSVs
outputs/results/             Current final result JSON/CSV files
splits/                      Current SOP JSON splits
archive/                     Legacy scripts, features, plots, and results
docs/                        Human-readable result summaries
```

## Reproduce Final Results

Build labels and QD-only SOP12 feature tables:

```bash
python3 1_feature_engineering/build_raw_label_table.py
python3 1_feature_engineering/build_sop12_features.py --windows 100
```

Generate fixed cell-level SOP splits:

```bash
python3 2_modeling_featuring/generate_json_splits.py \
  --dataset data/intermediate/features_matr_hust_sop12.csv \
  --dataset-prefixes matr hust \
  --output-dir splits/sop_matr_hust_qdonly
```

Run final within-dataset and cross-dataset experiments:

```bash
python3 2_modeling_featuring/run_sop_protocol_baselines.py \
  --dataset data/intermediate/features_matr_hust_sop12.csv \
  --feature-set sop12 \
  --label-column cycle_life \
  --split-dir splits/sop_matr_hust_qdonly \
  --within-prefixes matr hust \
  --cross-pairs matr:hust \
  --windows 100 \
  --models elastic_net xgboost \
  --output outputs/results/final_sop12_qdonly_eol85_n100.json
```

## Final Results

Summary table: `outputs/results/final_sop12_qdonly_summary.csv`

Detailed JSON:

- `outputs/results/final_sop12_qdonly_eol85_n100.json`
- `outputs/results/final_sop12_qdonly_within_matr_eol85_n100.json`

| Experiment | Model | MAE | SMAPE | R2 |
| --- | --- | ---: | ---: | ---: |
| MATR -> MATR | Elastic Net | 265.64 | 37.34 | -0.370 |
| MATR -> MATR | XGBoost | 221.66 | 30.59 | 0.087 |
| HUST -> HUST | Elastic Net | 194.11 | 13.58 | 0.326 |
| HUST -> HUST | XGBoost | 162.05 | 11.32 | 0.384 |
| MATR -> HUST | Elastic Net | 1034.03 | 110.90 | -16.500 |
| MATR -> HUST | XGBoost | 909.32 | 87.87 | -10.476 |

The strongest result is HUST within-dataset with XGBoost. The MATR -> HUST
cross-dataset setting remains weak, showing that the domain gap is still the
main limitation.

## Archive

Older top8, SOP-transition, plotting, and report-generation work has been moved
under `archive/`:

- `archive/legacy_experiment_scripts/`
- `archive/legacy_features/`
- `archive/legacy_results/`
- `archive/legacy_plots/`
