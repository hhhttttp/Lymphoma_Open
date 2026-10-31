# Lymphoma multimodal prediction

Python code for lymphoma prediction using CellProfiler (CP), QuPath (QP), and clinical features, including multimodal fusion, external validation, model interpretation, and survival analysis.

The code is organized into data processing, model construction, external evaluation, and analysis. Input data are supplied separately by the user; trained models, predictions, figures, and result tables are generated locally.

## Repository layout

```text
Lymphoma_OpenSource/
├── code/
│   ├── 1_data_processing/       # Feature aggregation and sample alignment
│   ├── 2_model_construction/    # Fusion, classifiers, and molecular tasks
│   └── 3_external_validation/  # TCGA validation scenarios
│       ├── external_validation.py           # Shared evaluation of saved models
│       ├── validate_image_only.py
│       ├── validate_imputed_clinical.py
│       └── validate_real_clinical.py
├── analysis/                   # PR, DCA, SHAP, Cox, survival, and demographics
├── model_artifacts.py          # Save/load baseline folds and apply stored preprocessing
├── settings.py                 # Input/output path configuration
├── workflow_utils.py           # Shared preprocessing and input checks
├── requirements.txt
├── requirements-analysis.txt
└── README.md
```

## Installation

Use Python 3.10 and the pinned dependencies below.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the additional dependencies for interpretation, survival, and classifier comparisons:

```bash
python -m pip install -r requirements-analysis.txt
```

Dependency versions are pinned for Python 3.10. Chinese plot labels may require a CJK font.

`settings.py` configures paths through `LYMPHOMA_DATA_DIR` and `LYMPHOMA_OUTPUT_DIR`, defaulting to `data/` and `results/` under the repository. `workflow_utils.py` supplies shared preprocessing and input checks.

## Input preparation

Provide your own input files under the following directory structure. These directories and files are not included in the release. Set `LYMPHOMA_DATA_DIR` to use a different input root.

```text
data/
├── internal/
│   ├── linba_cellprofiler_full.csv
│   ├── linba_qupath_full.csv
│   ├── linba_linchuang_6.xlsx
│   ├── clinical_survival_200.xlsx            # Survival/IPI/demographics tasks
│   ├── linba_linchuang_with_3new_target.xlsx # ABC subtype task
│   └── DLBCL.xlsx                           # c-MYC prediction task
├── external/
│   ├── linba_TCGA_cellprofiler.csv
│   ├── linba_TCGA_qupath.csv
│   ├── TCGA临床1.xlsx                       # Real-clinical validation/survival
│   └── TCGA标签.xlsx                        # External survival
└── raw/
    ├── qupath/                              # Optional per-object TSV exports
    └── cellprofiler/                        # Optional MyExpt_Image.csv exports
```

CP/QP tables use the first column as the sample identifier and `label` as the binary outcome column; the baseline loader falls back to the second column if `label` is absent. Use `negative`/`positive` or numeric `0`/`1`. The remaining feature columns must be numeric. Feature columns must agree between development and external cohorts. Labels are required separately from raw feature aggregation.

The internal clinical table uses `Image` for sample alignment and the clinical columns `Gender`, `AgeGroup（≥60）`, `Extranodalinvolvement`, and `Stage Group`. Identifiers are stripped of `.svs` and surrounding whitespace before inner joins. Check matching with `check_sample_alignment` before training.

Internal survival uses `Image`, `OS` (months), and `OS_status` (1 = death, 0 = censored). The `cox.py` loader additionally requires `Bsympstoms`, `CD10`, `COO`, `HS`, `MYD88`, `c-myc(40%)`, and `IPI`, although c-MYC and MYD88 are excluded from the fitted Cox model. See its loader for cohort filtering details. The demographics script uses `AgeGroup（≥60）_y` and `Gender_y` in the survival table. External survival uses `Image`, `day`, and `生存状态` (1 = death, 0 = censored). The ABC task requires `ABC型`. The c-MYC task uses `DLBCL.xlsx` with `Image`, the clinical columns listed above, and `c-myc(40%)` as the target. Preserve these column names or adapt the loaders to your own schema.

`aggregate_qupath` writes `external/linba_TCGA_qupath.csv` from raw TSV exports. `aggregate_cellprofiler` writes `external/linba_TCGA_cellprofiler.csv` from raw image measurements. These are optional external feature-preparation utilities, not WSI segmentation or complete label-preparation pipelines.

## Baseline model and PR curves

Run scripts directly from the repository root. Train the baseline once before running its analyses:

```bash
MPLBACKEND=Agg python code/1_data_processing/check_sample_alignment.py
MPLBACKEND=Agg python code/2_model_construction/train_fusion.py
MPLBACKEND=Agg python analysis/precision_recall.py
```

`train_fusion` defaults to seed 27. CP/QP/clinical/MEFF use random forests with 100 trees. PSF uses logistic regression trained on in-sample base probabilities plus clinical features, and PLAF averages CP/QP probabilities.

The training script writes two artifacts:

- `results/fusion_baseline/trained_folds_seed_27.joblib`: fitted models from all five folds, scalers, training-only imputation values, selected features, train/test indices, normalized sample IDs, input-feature snapshot, OOF probabilities, thresholds, and run ID.
- `results/fusion_baseline/OOF_probs_ROC_seed_27.npz`: portable probability arrays with sample IDs, fold IDs, and the matching run ID.

All prediction-based scripts in `analysis/` and all scripts in `code/3_external_validation/` read the joblib artifact. Internal evaluation uses its saved OOF predictions, never predictions from an ensemble evaluated on its own training cohort. External validation and survival apply the exact saved five-fold ensemble to external inputs. Missing or incompatible artifacts produce an error; these workflows never automatically retrain a classifier. Existing probability-only NPZ files are insufficient: run the updated training script once to save the actual models.

`precision_recall` writes `results/fusion_baseline/PR_all_models_seed_27.png`; its legend reports average precision (AP), which differs from ROC AUC. Set `LYMPHOMA_MODEL_PATH` to select another trusted baseline joblib artifact. The artifact seed must match the analysis seed and its scikit-learn version must match the current environment. Loading checks that the fold models reproduce their saved OOF predictions.

The joblib artifact contains sample IDs and input features; keep it with private research data. Only load artifacts from a trusted training run.

`MPLBACKEND=Agg` selects a non-interactive Matplotlib backend for these commands. The default analysis seed is 27. If changing the baseline seed, also update the analysis seed or its `load_baseline` argument.

Examples of follow-up analyses (no prediction-model fitting):

```bash
python analysis/decision_curve.py
python analysis/shap_explanation.py
python analysis/survival.py
python analysis/cox.py
python analysis/external_survival.py
python analysis/ipi_comparison.py
```

Cox and Kaplan–Meier fitting, descriptive PCA/UMAP, and curve-equation estimation remain part of statistical analysis. They do not refit the baseline classifiers. Internal Cox calibration remains an apparent, same-cohort analysis, and OOF Youden thresholds are descriptive when evaluated on the same internal cohort.

`cox.py` and `survival_curve_estimation.py` now use the baseline **multimodal** PSF OOF score, rather than fitting an image-only PSF. `ipi_comparison.py` compares the observed IPI score with baseline OOF predictions on the same matched samples; it no longer fits an IPI-only classifier or IPI-based fusion model. SHAP explains the saved models on each fold's held-out samples: RF attributions are on the probability scale, while PSF attributions describe meta-model log odds over CP/QP probabilities and clinical inputs. It does not substitute a separately trained RF for the stack.

External survival averages fold predictions per slide, then averages slide probabilities per patient before joining one survival record per patient. Missing external clinical values use the stored baseline zero-fill rule, matching the real-clinical validation scenario below.

`survival_curve_estimation.py` evaluates 6, 12, ..., 60 **months** directly against input `OS` in months, with no day/month conversion. Exported tables and curve equations use months. Internal survival KM axes use months; one-, three- and five-year AUC horizons are 12, 36 and 60 months, and three-year calibration is evaluated at 36 months. The c-MYC overall ROC legend reports the pooled OOF AUC; the fold mean and standard deviation are reported separately.

## External validation with saved models

After running `train_fusion.py` once, choose a scenario:

```bash
python code/3_external_validation/validate_image_only.py
python code/3_external_validation/validate_imputed_clinical.py
python code/3_external_validation/validate_real_clinical.py
```

All three entry points use `external_validation.py` in the same `code/3_external_validation/` directory. They do not read internal source tables, select new features, fit models/scalers, or retune thresholds. Image preprocessing comes from each saved fold, and classification thresholds come directly from its saved development OOF results. External labels are used only for evaluation, and CP/QP labels must agree on shared sample IDs. Use a `label` column, or a separate second-column label distinct from the feature columns.

- **Image-only:** evaluates saved CP, QP, and PLAF only; no clinical table or clinical predictor columns are needed.
- **Imputed clinical:** evaluates all six baseline models. Every clinical predictor is replaced by a constant calculated from the saved development-feature snapshot: mode for variables with at most 10 unique values, otherwise median. No external clinical table is read, and external labels do not affect these values.
- **Real clinical:** evaluates all six baseline models using `external/TCGA临床1.xlsx`, matched by the first 12 characters of the TCGA patient ID. Missing variables/values are passed to each saved fold's clinical fill rule (zero in the baseline). Available `Extranodalinvolvement` values are mapped and retained.

Validation metrics are calculated per slide, so a patient with multiple slides contributes multiple observations; survival analysis instead aggregates to one prediction per patient. Each scenario saves metrics, sample-level predictions, probabilities, model inputs, stored fold-feature lists, threshold information, ROC/PR/confusion-matrix figures, and `validation_source.json` containing the source run ID and scenario. Bootstrap intervals condition on external predictions and do not include refitting variability or patient clustering. Image model names are consistently `CP`, `QP`, and `PLAF` across artifacts.

## Additional workflows

The names below correspond to Python script filenames without the `.py` extension. Baseline analyses and external validation require `train_fusion` first. Other scripts in `code/2_model_construction/` remain independent training experiments; their outputs are not substituted for the baseline analysis artifact.

| Script | Behavior / prerequisite | Output subdirectory |
|---|---|---|
| `risk_stratification` | Group saved baseline OOF probabilities | `fusion_baseline` |
| `compare_classifiers` | Compare single-modality classifiers; requires XGBoost | `classifier_comparison` |
| `predict_abc_subtype` | ABC subtype prediction; fixed seed 72 | `abc_subtype` |
| `predict_cmyc` | c-MYC prediction; fixed seed 0 | `cmyc` |
| `validate_image_only` | Saved CP/QP/PLAF fold ensemble, no clinical input | `external_image_only` |
| `validate_imputed_clinical` | Saved baseline ensemble, clinical constants from saved development snapshot | `external_imputed` |
| `validate_real_clinical` | Saved baseline ensemble with observed external clinical inputs | `external_real_clinical` |
| `decision_curve` | Saved baseline OOF DCA and descriptive embeddings | `decision_curve` |
| `shap_explanation` | Held-out SHAP of saved baseline fold models | `explainability` |
| `feature_correlation` | Correlations of the saved selected-feature union | `feature_correlation` |
| `survival` | Saved baseline OOF scores for KM/Cox, C-index, time-dependent AUC and apparent calibration | `survival` |
| `cox` | Adjusted Cox using saved multimodal PSF; excludes c-MYC/MYD88 covariates | `cox` |
| `external_survival` | Saved baseline fold ensemble; external clinical and survival tables required | `external_survival` |
| `survival_curve_estimation` | Curve estimation using saved multimodal PSF OOF scores | `survival_curve_estimation` |
| `ipi_comparison` | Observed IPI vs saved baseline OOF predictions on matched samples | `ipi_comparison` |
| `demographics` | Cohort demographic distributions | `demographics` |

