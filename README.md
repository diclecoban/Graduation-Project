# Battery Lifetime Prediction

Early-cycle lithium-ion battery lifetime prediction on the **MATR/Severson**
and **HUST** public LFP datasets, with a reproducible SOPv2 pipeline for
within-dataset modeling, cross-dataset transfer analysis, concept-shift
diagnostics, target-side calibration, and conformal prediction.

The main scientific finding is that cross-dataset failure is not explained by
feature count, model choice, or covariate shift alone. MATR and HUST exhibit
**concept shift**: the same early-cycle capacity features do not preserve the
same relationship with lifetime across datasets.

---

## Highlights

- SOPv2-compliant labels: `Q0 = median(QD cycles 2..5)`, EOL at `0.85 * Q0`.
- 34 capacity-only early-cycle features from discharge-capacity trajectories.
- 5-seed, cell-level, lifetime-stratified `70/15/15` train/calibration/test splits.
- Seven model families: Elastic Net, PLS, Random Forest, XGBoost, CatBoost,
  Gaussian Process, and Stacking.
- Cross-dataset transfer diagnostics for MATR -> HUST and HUST -> MATR.
- SHAP, feature-transfer stability, survival/censoring, DMD/Koopman pilot,
  CORAL/MMD domain adaptation, target calibration, and MAPIE split conformal
  prediction.
- Advisor-facing result tables under `outputs/advisor_tables/`.

---

## Headline Results

### Within-Dataset Prediction

Primary protocol: 34 capacity-only features, log-target training, 5 seeds,
`N=100` early cycles. Best model is selected by mean R2 within the fixed
protocol.

| Dataset | Best model | MAE | sMAPE | R2 |
|---|---|---:|---:|---:|
| MATR | CatBoost | 171.7 | 23.7 | 0.575 |
| HUST | Random Forest | 178.0 | 12.2 | 0.340 |

Full summary:

```text
outputs/results_v2_34feat_log/results_summary.csv
```

### Cross-Dataset Transfer

Naive source-to-target transfer fails in both directions. Negative R2 means the
model is worse than predicting the target-domain mean.

| Direction | Raw transfer setting | Best raw R2 | Best raw MAE | Target-calibrated R2, k=20 | Interpretation |
|---|---|---:|---:|---:|---|
| HUST -> MATR | All features / best model | -2.052 | 569.2 | about -0.05 to -0.11 | Transfer signal exists but is miscalibrated |
| MATR -> HUST | All features / best model | -8.125 | 781.2 | about -0.02 to -0.13 | Stronger transfer collapse |

Advisor comparison tables:

```text
outputs/advisor_tables/advisor_point_comparison.md
outputs/advisor_tables/advisor_conformal_comparison.md
```

---

## Repository Layout

```text
.
├── 0_data/                    # Raw data download and dataset audits
│   ├── download_data.py
│   ├── build_matr_audit.py
│   └── build_hust_audit.py
├── 1_features/                # SOPv2 feature construction
│   └── build_features.py
├── 2_models/                  # Splits, VIF, within/cross experiments
│   ├── generate_splits.py
│   ├── vif_screening.py
│   ├── run_experiments.py
│   └── metrics_utils.py
├── 3_analysis/                # Shift, XAI, calibration, CP, DA analyses
│   ├── shift_metrics.py
│   ├── feature_transfer_stability.py
│   ├── shap_feature_importance.py
│   ├── survival_censoring.py
│   ├── concept_shift_diagnostics.py
│   ├── conditional_shift_decomposition.py
│   ├── target_rescaling.py
│   ├── domain_adaptation.py
│   ├── coral_target_calibration.py
│   ├── coral_source_conformal.py
│   ├── conformal_prediction.py
│   └── make_advisor_comparison_tables.py
├── notebooks/
│   └── run_pipeline_colab.ipynb
├── data/
│   ├── raw/                   # Gitignored raw .pkl files
│   └── intermediate/          # Feature tables, audits, diagnostics
├── splits/sop_v2/             # 5-seed MATR/HUST split JSON files
├── outputs/                   # Reproducible result CSV/JSON/figures
├── docs/                      # Project summaries and thesis text drafts
├── legacy/                    # Archived older code and reference material
├── run_pipeline.py            # Main pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Two-Phase Workflow

| Phase | Environment | Purpose | Main outputs |
|---|---|---|---|
| Phase A: Extract | Colab / Drive | Read large raw `.pkl` files, audit cells, build feature CSVs | `data/intermediate/features_sop12_combined.csv` |
| Phase B: Model + Analyze | Local CPU | Run splits, models, shift diagnostics, calibration, CP | `outputs/**`, `docs/**` |

Phase A is only needed when raw data or feature definitions change. Phase B can
be repeated locally using the committed intermediate CSV files.

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Check which stages already have outputs:

```bash
python3 run_pipeline.py --status
```

Run the modeling phase:

```bash
python3 run_pipeline.py --phase model
```

Run the analysis phase:

```bash
python3 run_pipeline.py --phase analysis
```

Run selected stages:

```bash
python3 run_pipeline.py --stages splits experiments
python3 run_pipeline.py --stages shift shap survival
python3 run_pipeline.py --stages target_rescale conformal
```

Every non-status pipeline run writes local metadata under `outputs/runs/`,
including the selected stages, Git state, environment details, stage commands,
return codes, elapsed time, and output-file existence checks.

---

## Interactive Demo Dashboard

The dependency-free research dashboard presents the thesis evidence chain
using the repository's existing CSV/JSON outputs:

```bash
python3 dashboard/run_dashboard.py
```

Open `http://localhost:8765`. The dashboard includes within/cross-dataset
results, transfer controls, conformal-prediction comparisons, an interactive
feature--lifetime explorer, pipeline status, and a guided video-demo mode.

The recommended narration is available in
`docs/demo_video_script_en.md`.

---

## Reproducible Experiments

Within-dataset headline protocol:

```bash
python3 2_models/run_experiments.py \
  --log-target \
  --output-dir outputs/results_v2_34feat_log
```

Cross-dataset transfer:

```bash
python3 2_models/run_experiments.py \
  --cross-dataset \
  --log-target \
  --output-dir outputs/results_v2_cross_34feat_log
```

Top-k feature sweep:

```bash
python3 3_analysis/top_k_sweep.py
```

CORAL + target calibration:

```bash
python3 3_analysis/coral_target_calibration.py
```

CORAL-after-source conformal prediction:

```bash
python3 3_analysis/coral_source_conformal.py
```

Standard split conformal prediction:

```bash
python3 3_analysis/conformal_prediction.py
python3 3_analysis/summarize_conformal_results.py
```

Advisor-facing comparison tables:

```bash
python3 3_analysis/make_advisor_comparison_tables.py
```

---

## Key Interpretation

Top-k feature selection helps build compact within-dataset models, especially
for HUST, but it does not solve cross-dataset negative-R2 transfer.

CORAL is treated as a **covariate-alignment baseline**, not as the main
solution. CORAL-only and CORAL-after-source CP remain inadequate, showing that
matching feature distributions alone is not enough.

Target-side calibration and target-domain conformal prediction are more
effective under concept shift. A small labeled target set corrects systematic
source-to-target bias and restores useful uncertainty calibration.

---

## Important Documentation

| File | Purpose |
|---|---|
| `docs/proje_baslangictan_bugune_ozet.md` | Full Turkish project history and technical summary |
| `docs/akademik_deney_organizasyonu.md` | Evidence map: how existing experiments support the thesis argument |
| `docs/bitirme_projesi_sunum_taslagi.md` | Slide-by-slide graduation project presentation outline |
| `docs/graduation_presentation_outline_en.md` | English slide content and visual mapping |
| `docs/presentation_assets/README.md` | Draw.io/Mermaid diagrams and generated presentation figures |
| `docs/ieee_report/battery_lifetime_ieee.tex` | IEEE-format final technical report |
| `docs/gtu_report/main.tex` | Graduation report in the official GTU LaTeX design |
| `docs/tez_giris_metodoloji_tartisma_taslagi.md` | Thesis-ready Introduction, Methodology, Discussion draft |
| `docs/advisor_feedback_followup.md` | Advisor-request follow-up and result interpretation |
| `docs/mlops_refactoring_recommendations.md` | Suggested next-step architecture and MLOps roadmap |
| `outputs/advisor_tables/advisor_point_comparison.md` | Point-prediction comparison table |
| `outputs/advisor_tables/advisor_conformal_comparison.md` | Conformal-prediction comparison table |

---

## Project Status

The engineering and analysis pipeline is complete for the current thesis scope.
Recommended future work is to modularize the pipeline with YAML/Hydra-style
configuration, add structured experiment tracking, and package the analysis
stages as reusable, logged, versioned workflows.
