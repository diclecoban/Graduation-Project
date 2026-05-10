# Student Next Steps

This repository now uses the SOPv2 pipeline as the active thesis workflow.
The earlier exploratory pipeline has been removed from the working tree so new
work should build on:

- `0_data/`
- `1_features/`
- `2_models/`
- `3_analysis/`
- `run_pipeline.py`

## Recommended student contribution path

1. Reproduce the current status locally:

   ```bash
   python3 run_pipeline.py --status
   ```

2. Use the 34-feature, log-target within-dataset results as the main baseline:

   ```bash
   outputs/results_v2_34feat_log/results_summary.csv
   ```

3. Frame the thesis contribution around the transfer question:

   Capacity-only early-cycle features improve within-dataset prediction, but
   naive MATR-HUST transfer fails because the lifetime mapping changes across
   datasets. Target-side calibration and conformal analysis show how much of
   that failure can be corrected with a small labeled target set.

4. Add one focused extension rather than many unrelated experiments. Good
   candidates:

   - stable-feature-only cross-dataset transfer using the least fragile
     features from `feature_transfer_stability.csv` (completed in
     `docs/student_contribution_least_fragile_transfer.md`);
   - a table comparing 12, 24, and 34 features for transfer robustness;
   - a compact thesis figure explaining within-dataset vs cross-dataset
     performance;
   - a short literature comparison section for why cross-dataset battery-life
     transfer is hard.

5. In the thesis/report, describe the earlier pipeline as the initial version
   and SOPv2 as the corrected final protocol. Avoid presenting the old SOP12
   results and the SOPv2 results as if they came from the same experimental
   setup.

## Suggested thesis wording

The initial project implemented early-cycle battery lifetime prediction using
capacity-derived features and baseline machine-learning models. Following
advisor feedback, the pipeline was revised to a reproducible SOPv2 protocol
with corrected Q0/EOL definitions, cell-level stratified splits, expanded
capacity-only features, within-dataset model comparison, and cross-dataset
transfer diagnostics. The final analysis focuses on why within-dataset
performance improves while naive cross-dataset generalization remains poor.
