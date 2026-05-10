# Least-Fragile Feature Transfer: Result and Next Actions

## What Was Done

A new student-led ablation was added to test whether cross-dataset transfer improves when the model uses only the features that looked most stable across MATR and HUST.

The selected feature subset is saved in:

```text
data/intermediate/feature_set_least_fragile6.txt
```

It contains six relatively least-fragile features from the feature-transfer report:

- `cycle_to_98pct`
- `exp_decay_k`
- `slope_linear`
- `slope_ratio`
- `linearity_r2`
- `cycle_to_99pct`

The cross-dataset experiment was run with the same SOPv2 model runner, log-target training, N=100, and 5 source-split seeds:

```bash
python3 2_models/run_experiments.py \
  --cross-dataset \
  --log-target \
  --windows 100 \
  --features-from data/intermediate/feature_set_least_fragile6.txt \
  --output-dir outputs/results_v2_cross_least_fragile6
```

## Main Visual

Use this figure in the report or presentation:

```text
outputs/results_v2_cross_least_fragile6/least_fragile_transfer_r2_comparison.png
```

A PDF version is also available:

```text
outputs/results_v2_cross_least_fragile6/least_fragile_transfer_r2_comparison.pdf
```

## What Happened

The stability-based 6-feature subset did **not** improve naive cross-dataset transfer. All feature sets still have negative R2, meaning they perform worse than simply predicting the target dataset mean.

| Direction | Feature Set | Best R2 Model | Best R2 | MAE at Best R2 | Best MAE Model | Best MAE |
|---|---|---:|---:|---:|---:|---:|
| HUST -> MATR | 12 features | random_forest | -1.725 | 542.8 | random_forest | 542.8 |
| HUST -> MATR | 24 features | random_forest | -1.804 | 552.0 | random_forest | 552.0 |
| HUST -> MATR | 34 features | gaussian_process | -2.052 | 569.2 | gaussian_process | 569.2 |
| HUST -> MATR | least-fragile 6 | gaussian_process | -3.638 | 723.2 | elastic_net | 709.6 |
| MATR -> HUST | 12 features | gaussian_process | -8.832 | 812.6 | gaussian_process | 812.6 |
| MATR -> HUST | 24 features | gaussian_process | -8.135 | 781.7 | gaussian_process | 781.7 |
| MATR -> HUST | 34 features | gaussian_process | -8.125 | 781.2 | gaussian_process | 781.2 |
| MATR -> HUST | least-fragile 6 | gaussian_process | -8.814 | 813.1 | gaussian_process | 813.1 |

Key observations:

- For `MATR -> HUST`, the best 34-feature result is R2 = -8.125, while least-fragile 6 gives R2 = -8.814.
- For `HUST -> MATR`, the best 12-feature result is R2 = -1.725, while least-fragile 6 gives R2 = -3.638.
- The stable subset removes fragile features, but also removes useful within-domain predictive signal.
- This supports the thesis argument that feature filtering alone is not enough; the main issue is conditional/concept shift in the mapping from early-cycle features to lifetime.

## What This Contributes

This is a useful negative result. It shows that:

1. Feature stability is not the same as predictive sufficiency.
2. Cross-dataset failure cannot be fixed by simply selecting the most stable-looking features.
3. Target-domain calibration remains more promising than feature filtering alone.

This can be described as a student contribution because it directly tests a follow-up hypothesis created from the feature-transfer diagnostics.

## What Is Still Missing

The current ablation is intentionally small. It answers one question, but it does not exhaust all possible transfer fixes.

Remaining gaps:

- The subset was selected manually from the feature-transfer report, not by a formally optimized selection rule.
- Only N=100 was tested.
- It tests naive transfer only; it does not combine least-fragile features with target-side calibration.
- It does not test whether capacity normalization plus least-fragile filtering helps.
- It does not test intermediate subset sizes such as top 8, top 10, or top 12 stability-ranked features.
- It does not include uncertainty intervals specifically for the feature-set comparison table.

## How To Improve or Fix Next

Recommended next steps, from easiest to strongest:

1. **Top-k stability sweep**

   Run cross-dataset transfer for top-k stability-ranked features, for example k = 4, 6, 8, 10, 12, 16. This checks whether 6 features was too restrictive.

2. **Capacity-normalized least-fragile test**

   Rebuild features with `--capacity-normalize`, then rerun the least-fragile experiment. This tests whether stable filtering and scale correction help together.

3. **Least-fragile plus target calibration**

   Run target rescaling or conformal target-adapted analysis using the least-fragile feature subset. This checks whether stable features reduce how much calibration is needed.

4. **Report as negative evidence if no improvement appears**

   If these still fail, keep the result as evidence that the transfer problem is not mainly a feature-selection problem. That is scientifically useful and supports the current thesis story.

## Suggested Thesis Wording

A stability-guided feature subset was evaluated as an additional ablation. The six selected features had comparatively lower distribution or relationship instability across MATR and HUST. However, using only these features did not improve naive cross-dataset transfer; R2 remained strongly negative in both directions and became worse than the 12-feature baseline for HUST -> MATR. This indicates that stable marginal feature behavior is insufficient for transfer when the conditional relationship between early capacity dynamics and lifetime changes across datasets. Therefore, target-domain calibration is a more effective remedy than feature filtering alone.
