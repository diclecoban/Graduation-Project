# Student Contribution: Least-Fragile Feature Transfer Test

## Question

The feature-transfer analysis identified a small set of capacity-only features
whose distributions and rank relationships were relatively more stable across
MATR and HUST. This extension tests whether using only those features improves
naive cross-dataset transfer.

## Feature Set

The least-fragile feature subset is saved at:

```text
data/intermediate/feature_set_least_fragile6.txt
```

It contains:

- `cycle_to_98pct`
- `exp_decay_k`
- `slope_linear`
- `slope_ratio`
- `linearity_r2`
- `cycle_to_99pct`

These were selected from the top of
`data/intermediate/feature_transfer_stability_report.txt`.

## Command

```bash
python3 2_models/run_experiments.py \
  --cross-dataset \
  --log-target \
  --windows 100 \
  --features-from data/intermediate/feature_set_least_fragile6.txt \
  --output-dir outputs/results_v2_cross_least_fragile6
```

## Result

The least-fragile subset did not improve naive transfer. It is competitive in
the MATR -> HUST direction only in the sense that all feature sets fail badly,
but it is worse than the 12/24/34 feature sets for HUST -> MATR.

| Direction | Feature set | Best R2 model | Best R2 | MAE at best R2 |
|---|---|---:|---:|---:|
| HUST -> MATR | 12 features | Random Forest | -1.725 | 542.8 |
| HUST -> MATR | 24 features | Random Forest | -1.804 | 552.0 |
| HUST -> MATR | 34 features | Gaussian Process | -2.052 | 569.2 |
| HUST -> MATR | least-fragile 6 | Gaussian Process | -3.638 | 723.2 |
| MATR -> HUST | 12 features | Gaussian Process | -8.832 | 812.6 |
| MATR -> HUST | 24 features | Gaussian Process | -8.135 | 781.7 |
| MATR -> HUST | 34 features | Gaussian Process | -8.125 | 781.2 |
| MATR -> HUST | least-fragile 6 | Gaussian Process | -8.814 | 813.1 |

Full comparison table:

```text
outputs/results_v2_cross_least_fragile6/feature_set_transfer_comparison.csv
```

## Interpretation

This is a useful negative result. Feature stability is not the same as
predictive sufficiency: the least-fragile features are more stable across
datasets, but they discard too much within-domain lifetime signal. The result
supports the broader thesis claim that naive cross-dataset transfer is limited
by conditional/concept shift, not just by a few fragile feature distributions.

For the thesis, this can be framed as a student-led ablation showing that
feature filtering alone is insufficient; target-domain calibration remains the
more effective repair.
