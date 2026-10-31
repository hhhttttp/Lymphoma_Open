"""Plot precision-recall curves using the saved baseline OOF probabilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import output_path
from model_artifacts import load_baseline

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

SEED = 27
MODEL_ORDER = ["PSF", "PLAF", "MEFF", "QP", "CP", "Clin"]
OUTPUT_DIR = Path(output_path("fusion_baseline"))


def main():
    saved = load_baseline(SEED)
    y = saved["y_true"]
    probabilities = saved["probabilities"]
    plt.figure(figsize=(8.2, 7.0))
    for model, values in probabilities.items():
        if (
            values.shape != y.shape
            or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
        ):
            raise ValueError(f"Invalid probabilities for {model}.")
        precision, recall, _ = precision_recall_curve(y, values)
        ap = average_precision_score(y, values)
        plt.step(
            recall, precision, where="post", linewidth=2, label=f"{model} (AP={ap:.3f})"
        )
    plt.axhline(
        y.mean(),
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"Baseline (positive fraction={y.mean():.3f})",
    )
    plt.title("Precision-recall curves (baseline OOF predictions)")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    path = OUTPUT_DIR / f"PR_all_models_seed_{SEED}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curves: {path}")


if __name__ == "__main__":
    main()
