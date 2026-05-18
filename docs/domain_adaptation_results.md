# Domain Adaptation Results

Feature set: auto-detected from CSV (34 features)

This experiment tests lightweight neural feature alignment against the current critical bottleneck: cross-dataset transfer under conditional shift.

Methods:

- `source_mlp`: source-only log-cycle MLP.
- `coral_mlp`: source MLP plus CORAL covariance alignment on target features.
- `mmd_mlp`: source MLP plus RBF-MMD alignment on target features.
- `residual_mean`: small labeled target-side residual calibration applied after each predictor.

## Best Rows By Direction

### hust_to_matr

- Best feature-alignment-only row: `coral_mlp` with R2=-1.514, MAE=491.7.
- Best target-calibrated row: `coral_mlp + residual_mean` at k=20 with R2=-0.194, MAE=307.2.

### matr_to_hust

- Best feature-alignment-only row: `mmd_mlp` with R2=-8.532, MAE=758.2.
- Best target-calibrated row: `coral_mlp + residual_mean` at k=20 with R2=-0.892, MAE=305.7.

## Interpretation

If CORAL/MMD improves only slightly while residual target calibration improves much more, the result strengthens the thesis claim: marginal feature alignment is insufficient under conditional shift, and labeled target-side calibration is the practical repair.

Full tables:

- `outputs/results_v2_domain_adaptation/results_detailed.csv`
- `outputs/results_v2_domain_adaptation/results_summary.csv`
