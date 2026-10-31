"""Compare observed IPI scores with saved baseline OOF predictions on matched IDs.

IPI is an ordinal clinical score, not an estimated probability. This script does
not train an IPI classifier or substitute IPI into a newly fitted fusion model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import data_path, output_path
from model_artifacts import load_baseline, prediction_frame, auc_bootstrap_ci, MODELS
from workflow_utils import validate_ids

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def main():
    saved = load_baseline()
    path = data_path("internal/clinical_survival_200.xlsx")
    ipi = pd.read_excel(path, engine="openpyxl")
    ipi["Merge_ID"] = validate_ids(ipi, "Image", path)
    ipi["IPI"] = pd.to_numeric(ipi["IPI"], errors="raise")
    observed = ipi["IPI"].dropna().to_numpy()
    if not np.isfinite(observed).all() or ((observed < 0) | (observed > 5)).any():
        raise ValueError("IPI must be a clinical score between 0 and 5.")
    cohort = prediction_frame(saved).merge(ipi[["Merge_ID", "IPI"]],
                                            on="Merge_ID", validate="one_to_one").dropna(subset=["IPI"])
    if cohort.empty or cohort.label.nunique() != 2:
        raise ValueError("IPI comparison needs matched samples from both outcome classes.")
    out = Path(output_path("ipi_comparison"))
    out.mkdir(parents=True, exist_ok=True)
    cohort["run_id"] = saved["run_id"]
    cohort.to_csv(out / "matched_IPI_baseline_predictions.csv", index=False)
    rows = []
    plt.figure(figsize=(9, 7))
    for model in ("IPI",) + MODELS:
        auc, lo, hi = auc_bootstrap_ci(cohort.label, cohort[model], seed=saved["seed"])
        name = "IPI (observed clinical score)" if model == "IPI" else f"{model} (baseline OOF)"
        rows.append({"Model": name, "N": len(cohort), "AUC": auc,
                     "CI_low": lo, "CI_high": hi, "run_id": saved["run_id"]})
        fpr, tpr, _ = roc_curve(cohort.label, cohort[model])
        plt.plot(fpr, tpr, label=f"{name}: {auc:.3f} ({lo:.3f}-{hi:.3f})")
    pd.DataFrame(rows).to_csv(out / "AUC_saved_baseline_vs_observed_IPI.csv", index=False)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Observed IPI vs saved baseline OOF predictions (matched cohort)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "ROC_saved_baseline_vs_observed_IPI.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
