"""External classification evaluation of saved baseline folds, without fitting."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_curve,
)

from model_artifacts import (
    MODELS, artifact_path, auc_bootstrap_ci, load_baseline,
    load_external_inputs, predict_saved,
)
from settings import data_path, output_path


SCENARIOS = {
    "none": {
        "folder": "external_image_only",
        "models": ("CP", "QP", "PLAF"),
        "metrics": "TCGA_external_validation_metrics.csv",
        "prefix": "TCGA_external_validation",
    },
    "imputed": {
        "folder": "external_imputed",
        "models": MODELS,
        "metrics": "TCGA_imputed_external_validation_metrics.csv",
        "prefix": "TCGA_imputed_external_validation",
    },
    "real": {
        "folder": "external_real_clinical",
        "models": MODELS,
        "metrics": "TCGA_real_clinical_validation_metrics.csv",
        "prefix": "TCGA_real_clinical",
    },
}


def metric_row(name, labels, probabilities, threshold, seed):
    auc, lo, hi = auc_bootstrap_ci(labels, probabilities, seed=seed)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "Model": name, "N": len(labels), "Positive_n": int(labels.sum()),
        "AUC": auc, "AUC_CI_low": lo, "AUC_CI_high": hi,
        "Youden_threshold": threshold,
        "Youden_threshold_from_development_OOF": threshold,
        "Accuracy": accuracy_score(labels, predictions),
        "Sensitivity": recall_score(labels, predictions, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "Precision": precision_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "Average_precision": average_precision_score(labels, probabilities),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def plot_curves(labels, probabilities, metrics, path, kind="ROC"):
    fig, axis = plt.subplots(figsize=(8, 7))
    rows = metrics.set_index("Model")
    for model, scores in probabilities.items():
        row = rows.loc[model]
        if kind == "ROC":
            x, y, _ = roc_curve(labels, scores)
            label = f"{model}: {row.AUC:.3f} ({row.AUC_CI_low:.3f}-{row.AUC_CI_high:.3f})"
        else:
            y, x, _ = precision_recall_curve(labels, scores)
            label = f"{model}: AP={row.Average_precision:.3f}"
        axis.plot(x, y, linewidth=2, label=label)
    if kind == "ROC":
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    else:
        axis.axhline(labels.mean(), color="gray", linestyle="--", label="Positive fraction")
        axis.set(xlabel="Recall", ylabel="Precision")
    axis.set_title(f"TCGA {kind}: saved baseline fold ensemble")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_confusion_matrices(labels, probabilities, thresholds, path):
    columns = min(3, len(probabilities))
    rows = (len(probabilities) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    for axis, (model, scores) in zip(axes.flat, probabilities.items()):
        matrix = confusion_matrix(labels, scores >= thresholds[model], labels=[0, 1])
        axis.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
        axis.set_xticks([0, 1], ["Negative", "Positive"])
        axis.set_yticks([0, 1], ["Negative", "Positive"])
        axis.set(xlabel="Predicted", ylabel="True",
                 title=f"{model}: development OOF threshold={thresholds[model]:.3f}")
    for axis in list(axes.flat)[len(probabilities):]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_external_validation(clinical_mode, seed=27):
    """Freeze models, feature selection, scaling and development OOF thresholds."""
    if clinical_mode not in SCENARIOS:
        raise ValueError(f"Unknown clinical scenario: {clinical_mode}")
    config = SCENARIOS[clinical_mode]
    saved = load_baseline(seed)
    clinical_path = data_path("external/TCGA临床1.xlsx") if clinical_mode == "real" else None
    inputs = load_external_inputs(
        saved, data_path("external/linba_TCGA_cellprofiler.csv"),
        data_path("external/linba_TCGA_qupath.csv"), clinical_path,
        clinical_mode=clinical_mode, require_labels=True,
    )
    labels = inputs["label"].to_numpy(dtype=int)
    if np.unique(labels).size != 2:
        raise ValueError("The aligned external cohort needs both classes for ROC evaluation.")
    probabilities = predict_saved(saved, inputs, models=config["models"])
    thresholds = {model: float(saved["thresholds"][model]) for model in probabilities}
    out = Path(output_path(config["folder"]))
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([
        metric_row(model, labels, scores, thresholds[model], seed)
        for model, scores in probabilities.items()
    ])
    metrics["run_id"] = saved["run_id"]
    metrics["clinical_mode"] = clinical_mode
    metrics["threshold_source"] = "saved development OOF Youden"
    metrics.to_csv(out / config["metrics"], index=False)

    predictions = inputs[["Merge_ID", "Patient_ID"]].copy()
    predictions["True_label"] = labels
    predictions["run_id"] = saved["run_id"]
    predictions["clinical_mode"] = clinical_mode
    for model, scores in probabilities.items():
        predictions[f"{model}_probability"] = scores
        predictions[f"{model}_prediction"] = (scores >= thresholds[model]).astype(int)
    predictions.to_csv(out / "TCGA_sample_level_predictions.csv", index=False)
    inputs.to_csv(out / "TCGA_model_inputs.csv", index=False)
    np.savez(
        out / "TCGA_external_probs.npz", y_true=labels,
        sample_ids=inputs.Merge_ID.to_numpy(dtype=str),
        patient_ids=inputs.Patient_ID.to_numpy(dtype=str),
        run_id=saved["run_id"], clinical_mode=clinical_mode,
        protocol_version="baseline_saved_fold_ensemble_v1", **probabilities,
        **{f"threshold_{m}": t for m, t in thresholds.items()},
    )
    pd.DataFrame([
        {"Model": model, "Threshold": threshold,
         "threshold_source": "saved development OOF Youden",
         "clinical_data_used": clinical_mode != "none",
         "clinical_mode": clinical_mode, "run_id": saved["run_id"]}
        for model, threshold in thresholds.items()
    ]).to_csv(out / "thresholds_and_scope.csv", index=False)
    pd.DataFrame([
        {"fold": fold["fold"], "modality": modality, "feature": feature}
        for fold in saved["folds"]
        for modality in ("CP", "QP", "Clin")
        if modality != "Clin" or clinical_mode != "none"
        for feature in fold["features"][modality]
    ]).to_csv(out / "saved_fold_features.csv", index=False)
    provenance = {
        "model_path": str(artifact_path(seed)), "run_id": saved["run_id"],
        "training_protocol": saved["protocol"], "seed": seed,
        "fold_count": len(saved["folds"]), "clinical_mode": clinical_mode,
        "evaluation_unit": "slide", "n_slides": len(inputs),
        "n_patients": int(inputs.Patient_ID.nunique()),
        "threshold_source": "saved development OOF Youden",
        "interval_scope": "bootstrap of saved external predictions; no model refitting",
        "clinical_fill": ("development snapshot mode for <=10 unique values, otherwise median"
                          if clinical_mode == "imputed" else
                          "saved per-fold baseline fills for missing clinical values"
                          if clinical_mode == "real" else "no clinical predictors"),
    }
    if clinical_mode == "imputed":
        provenance["clinical_fill_values"] = {
            name: float(inputs[name].iloc[0]) for name in saved["columns"]["Clin"]
        }
    (out / "validation_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    prefix = config["prefix"]
    plot_curves(labels, probabilities, metrics, out / f"{prefix}_ROC.png")
    plot_curves(labels, probabilities, metrics, out / f"{prefix}_PR.png", kind="PR")
    if clinical_mode != "none":
        fusion = {m: probabilities[m] for m in ("MEFF", "PSF", "PLAF")}
        plot_curves(labels, fusion, metrics, out / f"{prefix}_ROC_fusion_only.png")
    plot_confusion_matrices(labels, probabilities, thresholds, out / f"{prefix}_confusion_matrices.png")
    print(f"Saved baseline run {saved['run_id']}; scenario={clinical_mode}; outputs={out}")
    print(metrics.to_string(index=False))
    return metrics
