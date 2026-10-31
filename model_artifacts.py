"""Persist baseline fold models and reuse their predictions without fitting models.

Only load joblib artifacts produced by a trusted local training run. They include
sample IDs and training features and must be treated as private research data.
"""

import os
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve

from settings import output_path
from workflow_utils import binary_labels, numeric_frame, validate_ids

MODELS = ("Clin", "CP", "QP", "MEFF", "PSF", "PLAF")
ARTIFACT_VERSION = 1
CLINICAL_MAPPING = {
    "Gender": "Clin_Gender",
    "AgeGroup（≥60）": "Clin_AgeGroup",
    "Extranodalinvolvement": "Clin_Extranodal",
    "Stage Group": "Clin_Stage",
}


def artifact_path(seed=27):
    default = output_path(f"fusion_baseline/trained_folds_seed_{seed}.joblib")
    return Path(os.environ.get("LYMPHOMA_MODEL_PATH", default)).expanduser().resolve()


def youden_threshold(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def save_baseline(seed, frame, columns, labels, probabilities, folds):
    """Save the exact fitted folds, cohort order, raw inputs and OOF probabilities."""
    inputs = frame[["Merge_ID"] + sum((columns[m] for m in ("CP", "QP", "Clin")), [])].copy()
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "run_id": uuid.uuid4().hex,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "protocol": "baseline_in_sample_stacking",
        "inputs": inputs.reset_index(drop=True),
        "columns": columns,
        "y_true": binary_labels(labels),
        "probabilities": probabilities,
        "folds": folds,
        "thresholds": {m: youden_threshold(labels, probabilities[m]) for m in MODELS},
    }
    path = artifact_path(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, path)
    return payload, path


def load_baseline(seed=27):
    path = artifact_path(seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing saved model artifact: {path}. Run "
            "python code/2_model_construction/train_fusion.py once with the updated "
            "training script. Old probability-only NPZ files cannot restore fitted models."
        )
    saved = joblib.load(path)
    if saved.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported model artifact version; regenerate with train_fusion.py.")
    if saved.get("protocol") != "baseline_in_sample_stacking":
        raise ValueError("Expected a train_fusion.py baseline artifact.")
    if saved.get("seed") != seed:
        raise ValueError(f"Expected seed {seed}; artifact contains {saved.get('seed')}.")
    if saved.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Use the same scikit-learn version as the saved training run.")
    inputs = saved["inputs"]
    ids = validate_ids(inputs, "Merge_ID", str(path))
    if not ids.equals(inputs["Merge_ID"]):
        raise ValueError("Artifact IDs must already be normalized.")
    y = binary_labels(saved["y_true"], "saved labels")
    if len(inputs) != len(y) or set(y) != {0, 1}:
        raise ValueError("Invalid saved training cohort.")
    for model in MODELS:
        scores = np.asarray(saved["probabilities"][model], dtype=float)
        if scores.shape != y.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"Invalid saved {model} OOF probabilities.")
    coverage = np.zeros(len(y), dtype=int)
    for fold in saved["folds"]:
        tr, te = np.asarray(fold["train_idx"]), np.asarray(fold["test_idx"])
        if (len(np.unique(te)) != len(te) or len(np.unique(tr)) != len(tr)
                or np.intersect1d(tr, te).size
                or not np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(len(y)))):
            raise ValueError("Invalid saved fold indices.")
        coverage[te] += 1
    if not np.all(coverage == 1):
        raise ValueError("Saved held-out folds must cover each sample exactly once.")
    # Catch accidental edits or mismatched probabilities before any analysis.
    for fold in saved["folds"]:
        te = fold["test_idx"]
        matrices, _ = fold_inputs(saved, fold, inputs.iloc[te])
        reconstructed = {
            m: fold["models"][m].predict_proba(matrices[m])[:, 1]
            for m in MODELS if m != "PLAF"
        }
        reconstructed["PLAF"] = (reconstructed["CP"] + reconstructed["QP"]) / 2
        for model in MODELS:
            if not np.allclose(reconstructed[model], saved["probabilities"][model][te],
                               atol=1e-10, rtol=1e-8):
                raise ValueError(f"Saved {model} OOF scores do not match the stored fold model.")
    return saved


def prediction_frame(saved):
    result = saved["inputs"][["Merge_ID"]].copy()
    result["label"] = saved["y_true"]
    for model in MODELS:
        result[model] = saved["probabilities"][model]
    return result


def fold_inputs(saved, fold, frame, models=MODELS):
    """Apply only stored preprocessing; no fit or feature selection occurs here."""
    requested = set(models)
    if not requested or requested - set(MODELS):
        raise ValueError(f"Unknown or empty model selection: {models}")
    required = requested & {"CP", "QP", "Clin"}
    if requested & {"MEFF", "PSF"}:
        required.update(("CP", "QP", "Clin"))
    if "PLAF" in requested:
        required.update(("CP", "QP"))
    matrices, names = {}, {}
    for modality in ("CP", "QP", "Clin"):
        if modality not in required:
            continue
        features = fold["features"][modality]
        raw = numeric_frame(frame[features])
        raw = raw.fillna(fold["means"][modality])
        matrices[modality] = fold["scalers"][modality].transform(raw)
        names[modality] = features
    if "MEFF" in requested:
        matrices["MEFF"] = np.hstack([matrices[m] for m in ("CP", "QP", "Clin")])
        names["MEFF"] = sum((names[m] for m in ("CP", "QP", "Clin")), [])
    if "PSF" in requested:
        p_cp = fold["models"]["CP"].predict_proba(matrices["CP"])[:, 1]
        p_qp = fold["models"]["QP"].predict_proba(matrices["QP"])[:, 1]
        matrices["PSF"] = np.column_stack((p_cp, p_qp, matrices["Clin"]))
        names["PSF"] = ["P_CP", "P_QP"] + names["Clin"]
    return matrices, names


def predict_saved(saved, frame, models=MODELS):
    """Average external predictions from the exact saved outer-fold models."""
    models = tuple(models)
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError(f"Unknown, repeated or empty model selection: {models}")
    if not saved["folds"] or frame.empty:
        raise ValueError("Predictions require saved folds and nonempty external inputs.")
    sums = {m: np.zeros(len(frame)) for m in models}
    for fold in saved["folds"]:
        matrices, _ = fold_inputs(saved, fold, frame, models=models)
        base_models = set(models) - {"PLAF"}
        if "PLAF" in models:
            base_models.update(("CP", "QP"))
        scores = {m: fold["models"][m].predict_proba(matrices[m])[:, 1]
                  for m in base_models}
        if "PLAF" in models:
            scores["PLAF"] = (scores["CP"] + scores["QP"]) / 2
        for model in models:
            sums[model] += scores[model]
    return {m: scores / len(saved["folds"]) for m, scores in sums.items()}


def selected_features(saved, modality):
    return sorted({name for fold in saved["folds"] for name in fold["features"][modality]})


def load_external_inputs(saved, cp_path, qp_path, clinical_path=None,
                         clinical_mode="real", require_labels=False):
    """Align external inputs using saved schemas and an explicit clinical scenario.

    real: use observed clinical values; leave missing values for saved fold fills.
    imputed: replace clinical inputs by development-snapshot mode/median values.
    none: image inputs only; no clinical table or clinical columns are required.
    Classification labels are validated independently of predictor construction.
    """
    if clinical_mode not in {"real", "imputed", "none"}:
        raise ValueError(f"Unknown clinical mode: {clinical_mode}")
    tables = []
    for modality, path in (("CP", cp_path), ("QP", qp_path)):
        raw = pd.read_csv(path)
        ids = validate_ids(raw, raw.columns[0], str(path))
        features = saved["columns"][modality]
        source_names = [name[len(modality) + 1:] for name in features]
        missing = set(source_names) - set(raw.columns)
        if missing:
            raise ValueError(f"{path}: missing training features: {sorted(missing)}")
        table = numeric_frame(raw[source_names]).rename(columns=dict(zip(source_names, features)))
        table.insert(0, "Merge_ID", ids)
        if require_labels:
            label_col = "label" if "label" in raw else raw.columns[1]
            if label_col in source_names or label_col == raw.columns[0]:
                raise ValueError(f"{path}: a separate outcome label column is required.")
            table[f"__label_{modality}"] = binary_labels(raw[label_col], str(path))
        tables.append(table)
    result = tables[0].merge(tables[1], on="Merge_ID", validate="one_to_one")
    if result.empty:
        raise ValueError("External CP and QP tables have no matching sample IDs.")
    if require_labels:
        if not result["__label_CP"].eq(result["__label_QP"]).all():
            raise ValueError("External CP/QP labels disagree on shared sample IDs.")
        result["label"] = result.pop("__label_CP")
        result = result.drop(columns="__label_QP")
    result["Patient_ID"] = result.Merge_ID.str[:12]
    if clinical_mode == "none":
        return result
    cols = saved["columns"]["Clin"]
    if clinical_mode == "imputed":
        for name in cols:
            development = numeric_frame(saved["inputs"][[name]])[name].dropna()
            if development.empty:
                raise ValueError(f"No saved development values for clinical variable {name}.")
            result[name] = (float(development.mode().iloc[0])
                            if development.nunique() <= 10 else float(development.median()))
        return result
    if clinical_path is None:
        raise ValueError("A clinical table is required for real-clinical predictions.")
    clinical = pd.read_excel(clinical_path, engine="openpyxl")
    if "Image" not in clinical:
        raise ValueError("External clinical table requires Image.")
    if clinical.Image.isna().any():
        raise ValueError("External clinical Image IDs must be nonmissing.")
    clinical["Patient_ID"] = clinical.Image.astype(str).str.strip().str[:12]
    validate_ids(clinical, "Patient_ID", str(clinical_path))
    clinical = clinical.rename(columns=CLINICAL_MAPPING)
    # Keep missing values until each saved fold's training-only fill is applied.
    for name in cols:
        clinical[name] = pd.to_numeric(clinical[name], errors="raise") if name in clinical else np.nan
    result = result.merge(clinical[["Patient_ID"] + cols], on="Patient_ID", validate="many_to_one")
    if result.empty:
        raise ValueError("No external image samples match the clinical table.")
    return result


def auc_bootstrap_ci(y, scores, n_bootstrap=2000, ci=0.95, seed=27):
    from sklearn.metrics import roc_auc_score

    y, scores = np.asarray(y), np.asarray(scores)
    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            estimates.append(roc_auc_score(y[idx], scores[idx]))
    limits = np.percentile(estimates, [(1 - ci) * 50, (1 + ci) * 50]) if estimates else [np.nan, np.nan]
    return roc_auc_score(y, scores), *limits
