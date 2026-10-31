"""Evaluate saved baseline OOF net benefit and descriptive feature embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import output_path
from model_artifacts import load_baseline

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 27
saved = load_baseline(SEED)
df_final = saved["inputs"].copy()
cols_cp, cols_qp, cols_clin = (saved["columns"][m] for m in ("CP", "QP", "Clin"))
X_cp_full = df_final[cols_cp].copy()
X_qp_full = df_final[cols_qp].copy()
X_clin_full = df_final[cols_clin].copy()
y = saved["y_true"]
true_labels = y
probs = saved["probabilities"]
n = len(y)

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from workflow_utils import impute_for_fold
import umap.umap_ as umap
import plotly.graph_objects as go
OUTPUT_DIR = output_path("decision_curve")
OUT_ROC_PNG = os.path.join(OUTPUT_DIR, f"ROC_seed_{SEED}.png")
OUT_DCA_PNG = os.path.join(OUTPUT_DIR, f"DCA_binary_OOF_seed_{SEED}.png")
OUT_DCA_CSV = os.path.join(OUTPUT_DIR, f"DCA_binary_OOF_seed_{SEED}.csv")
VIS_OUT_DIR = os.path.join(OUTPUT_DIR, "PCA_UMAP")
SANKEY_OUT_DIR = os.path.join(OUTPUT_DIR, "SANKEY")
for folder in (OUTPUT_DIR, VIS_OUT_DIR, SANKEY_OUT_DIR):
    os.makedirs(folder, exist_ok=True)

plt.figure(figsize=(10, 8))
for name in ["Clin", "CP", "QP", "MEFF", "PSF", "PLAF"]:
    fpr, tpr, _ = roc_curve(true_labels, probs[name])
    auc = roc_auc_score(true_labels, probs[name])
    plt.plot(fpr, tpr, lw=2.0, label=f"{name} AUC={auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title("ROC (OOF, outer 5-fold)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_ROC_PNG, dpi=300)
plt.show()
print("Saved:", OUT_ROC_PNG)


def decision_curve_net_benefit(y_true, p_pred, thresholds):
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    n = len(y_true)

    nb = []
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            nb.append(np.nan)
            continue

        y_hat = (p_pred >= pt).astype(int)
        tp = np.sum((y_hat == 1) & (y_true == 1))
        fp = np.sum((y_hat == 1) & (y_true == 0))

        net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt))
        nb.append(net_benefit)
    return np.array(nb)


def decision_curve_treat_all(y_true, thresholds):
    y_true = np.asarray(y_true).astype(int)
    prevalence = np.mean(y_true == 1)
    nb = []
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            nb.append(np.nan)
            continue
        net_benefit = prevalence - (1 - prevalence) * (pt / (1 - pt))
        nb.append(net_benefit)
    return np.array(nb)


def plot_dca(y_true, probs_dict, out_png, out_csv, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.linspace(0.01, 0.99, 99)

    rows = []
    plt.figure(figsize=(10, 7))

    nb_none = np.zeros_like(threshold_grid)
    plt.plot(threshold_grid, nb_none, "k--", lw=1.5, label="Treat none")

    nb_all = decision_curve_treat_all(y_true, threshold_grid)
    plt.plot(threshold_grid, nb_all, color="gray", lw=2.0, label="Treat all")

    for name, p in probs_dict.items():
        nb = decision_curve_net_benefit(y_true, p, threshold_grid)
        plt.plot(threshold_grid, nb, lw=2.3, label=name)
        for pt, nbi in zip(threshold_grid, nb):
            rows.append(
                {"Model": name, "Threshold": float(pt), "NetBenefit": float(nbi)}
            )

    for pt, nbi in zip(threshold_grid, nb_all):
        rows.append(
            {"Model": "Treat all", "Threshold": float(pt), "NetBenefit": float(nbi)}
        )
    for pt, nbi in zip(threshold_grid, nb_none):
        rows.append(
            {"Model": "Treat none", "Threshold": float(pt), "NetBenefit": float(nbi)}
        )

    plt.xlabel("Threshold probability (pt)")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (binary outcome, OOF probabilities)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    print("Saved:", out_png)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)


plot_dca(true_labels, probs, out_png=OUT_DCA_PNG, out_csv=OUT_DCA_CSV)


def scatter_2d(z, y, title, out_png):
    z = np.asarray(z)
    y = np.asarray(y).astype(int)

    plt.figure(figsize=(7.5, 6.5))
    for lab, color in [(0, "#1f77b4"), (1, "#d62728")]:
        idx = y == lab
        plt.scatter(
            z[idx, 0], z[idx, 1], s=22, alpha=0.75, c=color, label=f"Label={lab}"
        )
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    print("Saved:", out_png)


def run_pca_umap(X_df, y, prefix, seed=27):
    X = impute_for_fold(X_df, np.arange(len(X_df))).values.astype(float)
    Xs = StandardScaler().fit_transform(X)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed)
    z_pca = pca.fit_transform(Xs)
    out_pca = os.path.join(VIS_OUT_DIR, f"{prefix}_PCA2_seed_{seed}.png")
    scatter_2d(
        z_pca,
        y,
        title=f"{prefix} PCA (2D) | explained var={pca.explained_variance_ratio_.sum():.3f}",
        out_png=out_pca,
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    z_umap = reducer.fit_transform(Xs)
    out_umap = os.path.join(VIS_OUT_DIR, f"{prefix}_UMAP2_seed_{seed}.png")
    scatter_2d(z_umap, y, title=f"{prefix} UMAP (2D)", out_png=out_umap)


run_pca_umap(X_cp_full, y, prefix="CP", seed=SEED)
run_pca_umap(X_qp_full, y, prefix="QP", seed=SEED)
run_pca_umap(X_clin_full, y, prefix="Clin", seed=SEED)

X_concat = pd.concat([X_cp_full, X_qp_full, X_clin_full], axis=1)
run_pca_umap(X_concat, y, prefix="Concat_CP_QP_Clin", seed=SEED)


def make_sankey_for_model(y_true, p_pred, model_name="PSF", threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)

    thr_risk = float(np.median(p_pred))
    risk_group = np.where(p_pred >= thr_risk, "High risk", "Low risk")
    y_hat = (p_pred >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "True": np.where(y_true == 1, "True=1", "True=0"),
            "Risk": risk_group,
            "Pred": np.where(y_hat == 1, "Pred=1", "Pred=0"),
        }
    )

    nodes = ["True=0", "True=1", "Low risk", "High risk", "Pred=0", "Pred=1"]
    node_idx = {k: i for i, k in enumerate(nodes)}

    edges = []
    c1 = df.groupby(["True", "Risk"]).size().reset_index(name="count")
    for _, r in c1.iterrows():
        edges.append((node_idx[r["True"]], node_idx[r["Risk"]], int(r["count"])))

    c2 = df.groupby(["Risk", "Pred"]).size().reset_index(name="count")
    for _, r in c2.iterrows():
        edges.append((node_idx[r["Risk"]], node_idx[r["Pred"]], int(r["count"])))

    source = [e[0] for e in edges]
    target = [e[1] for e in edges]
    value = [e[2] for e in edges]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=18,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )

    fig.update_layout(
        title_text=f"Sankey: True → Risk(median) → Pred(thr={threshold}) | Model={model_name}",
        font_size=12,
        width=980,
        height=520,
    )
    return fig


for m in ["Clin", "CP", "QP", "MEFF", "PSF", "PLAF"]:
    out_html = os.path.join(SANKEY_OUT_DIR, f"Sankey_{m}_thr0.5_seed_{SEED}.html")
    fig = make_sankey_for_model(true_labels, probs[m], model_name=m, threshold=0.5)
    fig.write_html(out_html)
    print("Saved:", out_html)

print("\nAll done.")
print("Outputs:")
print(" - ROC:", OUT_ROC_PNG)
print(" - DCA:", OUT_DCA_PNG, OUT_DCA_CSV)
print(" - PCA/UMAP dir:", VIS_OUT_DIR)
print(" - Sankey dir:", SANKEY_OUT_DIR)
