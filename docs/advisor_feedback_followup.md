# Advisor Feedback Follow-up: Top-k and Shift/Calibration Comparison

This note answers the advisor's latest request:

1. Run a short top-k feature sweep with k = 3, 6, 10, 12.
2. Compare those results with conditional shift, target-side calibration, and
   conformal prediction findings already present in the repository.

## Reproducible Command

```bash
python3 3_analysis/top_k_sweep.py
```

The sweep ranks features by mean SHAP relative importance from
`data/intermediate/shap_feature_importance.csv`, writes feature-list files to
`data/intermediate/feature_set_top{k}_shap.txt`, and runs the standard SOPv2
experiment protocol at N = 100.

Default models for this short sweep:

- CatBoost
- Random Forest
- Gaussian Process
- Elastic Net

Outputs are saved under:

```text
outputs/results_v2_topk_shap/
```

## Top-k Feature Sets

| k | Features |
|---:|---|
| 3 | `Qdis_N`, `linearity_r2`, `accel_mean` |
| 6 | + `Qdis_cycle10`, `poly2_c`, `range_Qdis` |
| 10 | + `slope_last_quarter`, `poly2_a`, `variance_Qdis`, `slope_ratio` |
| 12 | + `sample_entropy`, `mad_Qdis` |

## Within-dataset Top-k Results

Best model by mean R2:

| k | HUST -> HUST R2 / MAE | MATR -> MATR R2 / MAE |
|---:|---:|---:|
| 3 | 0.394 / 174.0 | 0.316 / 217.7 |
| 6 | 0.323 / 180.9 | 0.462 / 187.2 |
| 10 | 0.316 / 183.6 | 0.483 / 187.4 |
| 12 | 0.327 / 181.1 | 0.518 / 179.1 |

Reference full 34-feature headline within-dataset results:

| Dataset | Best full-feature model | R2 / MAE |
|---|---|---:|
| HUST | Random Forest | 0.340 / 178.0 |
| MATR | CatBoost | 0.575 / 171.7 |

Interpretation:

- HUST does not benefit from adding many top-ranked features; top-3 is already
  competitive with the full 34-feature result.
- MATR improves as k increases; top-12 gets close to the full 34-feature
  result, but does not fully match it.
- The top-k sweep supports a compact-feature story for within-dataset
  prediction, especially for HUST.

## Cross-dataset Top-k Results

Best model by mean R2:

| k | HUST -> MATR R2 / MAE | MATR -> HUST R2 / MAE |
|---:|---:|---:|
| 3 | -0.691 / 334.9 | -8.132 / 781.5 |
| 6 | -1.912 / 325.9 | -7.971 / 757.7 |
| 10 | -2.932 / 663.1 | -7.948 / 771.5 |
| 12 | -1.691 / 347.7 | -8.000 / 774.1 |

Reference naive full-feature cross-dataset results:

| Direction | Best full-feature R2 / MAE |
|---|---:|
| HUST -> MATR | -2.052 / 569.2 |
| MATR -> HUST | -8.125 / 781.2 |

Interpretation:

- Top-k selection can improve HUST -> MATR substantially, especially top-3
  with Elastic Net.
- MATR -> HUST remains catastrophically poor for all k values.
- This asymmetry matches the existing shift diagnostics: HUST -> MATR retains
  weak rank signal, while MATR -> HUST is nearly uncorrelated with target
  lifetime.

## Conditional Shift Comparison

Existing file:

```text
data/intermediate/conditional_shift_summary.json
```

Main findings:

- Universal HUST/MATR life ratio: 2.09x.
- Slope-stable features: 18 / 34.
- Slope-shifted features: 16 / 34.
- HUST -> MATR keeps weak positive rank signal:
  - CatBoost Pearson r = 0.22
  - Random Forest Pearson r = 0.27
- MATR -> HUST has near-zero/negative rank signal:
  - CatBoost Pearson r = -0.12
  - Random Forest Pearson r = -0.14

Comparison with top-k:

- The best top-k cross result appears in HUST -> MATR, the same direction where
  conditional-shift analysis says some transfer signal remains.
- MATR -> HUST does not improve meaningfully with top-k, consistent with the
  near-zero transfer signal.

## Target-side Calibration Comparison

Existing file:

```text
outputs/results_v2_target_rescale/results_summary.csv
```

At N = 100, target-side calibration with k = 20 improves naive cross transfer
much more than top-k feature selection:

| Direction | Example model | Naive R2 | k=20 calibrated R2 |
|---|---|---:|---:|
| MATR -> HUST | CatBoost | -9.995 | -0.126 |
| MATR -> HUST | Elastic Net | -15.546 | -0.079 |
| HUST -> MATR | CatBoost | -3.065 | -0.071 |
| HUST -> MATR | Random Forest | -2.403 | -0.057 |

Interpretation:

- Top-k changes the representation.
- Target-side calibration directly corrects target-domain offset and therefore
  gives a much larger cross-domain improvement.

## Domain Adaptation Follow-up

New files:

```text
3_analysis/domain_adaptation.py
outputs/results_v2_domain_adaptation/results_summary.csv
docs/domain_adaptation_results.md
```

This follow-up tests whether feature-level alignment can repair the transfer
problem more directly than top-k selection. Three lightweight PyTorch models
are evaluated:

| Method | Meaning |
|---|---|
| `source_mlp` | Source-only log-cycle MLP |
| `coral_mlp` | Source MLP + CORAL covariance alignment on target features |
| `mmd_mlp` | Source MLP + RBF-MMD alignment on target features |

Best results:

| Direction | Best alignment-only R2 / MAE | Best target-calibrated DA R2 / MAE |
|---|---:|---:|
| HUST -> MATR | -1.514 / 491.7 | -0.194 / 307.2 |
| MATR -> HUST | -8.532 / 758.2 | -0.892 / 305.7 |

Interpretation:

- CORAL improves HUST -> MATR over the source-only neural baseline, but
  alignment-only transfer remains negative-R2.
- MATR -> HUST remains hard even with MMD/CORAL alignment.
- Adding residual target calibration after CORAL gives the strongest domain
  adaptation result, supporting the main claim that marginal feature alignment
  is insufficient under conditional shift.

## Conformal Prediction Comparison

Existing files:

```text
outputs/results_v2_conformal/paper_cp_summary.csv
outputs/results_v2_conformal/paper_cp_summary.md
```

Main 90% coverage pattern:

| Scenario | Coverage pattern |
|---|---|
| Within-dataset CP | close to nominal coverage |
| Source-calibrated cross CP | severe undercoverage |
| Target-domain CP | recovers nominal coverage |
| Residual-mean target-adapted CP | recovers coverage with narrower/more useful intervals |

Representative 90% examples:

| Scenario | Direction/model | Coverage |
|---|---|---:|
| Within CP | HUST / RF | 0.967 |
| Within CP | MATR / CatBoost | 0.920 |
| Source-calibrated cross CP | MATR -> HUST / RF | 0.148 |
| Source-calibrated cross CP | HUST -> MATR / RF | 0.305 |
| Target-domain CP, k=20 | MATR -> HUST / CatBoost | 0.902 |
| Target-domain CP, k=20 | HUST -> MATR / RF | 0.908 |
| Residual-mean target-adapted CP, k=20+20 | MATR -> HUST / CatBoost | 0.908 |
| Residual-mean target-adapted CP, k=20+20 | HUST -> MATR / RF | 0.909 |

Interpretation:

- Top-k is useful as a compact feature ablation, but it does not solve
  cross-dataset validity.
- Conformal prediction shows that valid uncertainty requires target-domain
  calibration under this shift.

## Suggested Thesis/Meeting Takeaway

The top-k sweep confirms that a small set of SHAP-important capacity features
can preserve much of the within-dataset performance, especially for HUST.
However, top-k selection alone does not solve cross-dataset transfer. The
remaining failure aligns with the conditional-shift analysis: the target
mapping changes across datasets, with a dominant life-offset component and
directional asymmetry. Target-side calibration and target-domain conformal
prediction are therefore more effective than feature selection alone for
handling MATR-HUST transfer.
