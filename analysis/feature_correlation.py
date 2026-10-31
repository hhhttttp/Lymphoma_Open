"""Analyze the saved baseline feature selections without rerunning selection."""

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

from model_artifacts import selected_features
from workflow_utils import impute_for_fold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import networkx as nx
OUTPUT_DIR = output_path("feature_correlation")
RHO_THR, FDR_Q, MAX_HEATMAP_FEATURES = 0.50, 0.05, 80
OUT_DIR = os.path.join(OUTPUT_DIR, f"crossmodal_spearman_unionSelected_seed_{SEED}")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_UNION_CSV = os.path.join(OUT_DIR, "union_selected_features.csv")
OUT_NET_PNG = os.path.join(OUT_DIR, f"spearman_network_unionSelected_rho{RHO_THR}_fdr{FDR_Q}.png")
OUT_HEATMAP_PNG = os.path.join(OUT_DIR, f"spearman_heatmap_unionSelected_max{MAX_HEATMAP_FEATURES}.png")
OUT_EDGES_CSV = os.path.join(OUT_DIR, f"spearman_edges_unionSelected_rho{RHO_THR}_fdr{FDR_Q}.csv")
cp_union = selected_features(saved, "CP")
qp_union = selected_features(saved, "QP")

print("\nUnion sizes:")
print("CP union =", len(cp_union))
print("QP union =", len(qp_union))
print("Clin =", len(cols_clin))


union_df = pd.DataFrame(
    {
        "feature": cp_union + qp_union + cols_clin,
        "modality": (["CP"] * len(cp_union))
        + (["QP"] * len(qp_union))
        + (["Clin"] * len(cols_clin)),
    }
)
union_df.to_csv(OUT_UNION_CSV, index=False)
print("[Saved]", OUT_UNION_CSV)


X_union = pd.concat(
    [
        X_cp_full[cp_union] if len(cp_union) else pd.DataFrame(index=X_cp_full.index),
        X_qp_full[qp_union] if len(qp_union) else pd.DataFrame(index=X_qp_full.index),
        X_clin_full[cols_clin],
    ],
    axis=1,
)


scaler = StandardScaler()
X_union_std = pd.DataFrame(scaler.fit_transform(impute_for_fold(X_union, np.arange(len(X_union)))), columns=X_union.columns)

all_features = X_union_std.columns.tolist()
p = len(all_features)
print("Total features for correlation =", p)


pairs = []
rhos = []
pvals = []

print("Computing pairwise Spearman correlations ...")
for i in range(p):
    a = X_union_std.iloc[:, i].values
    for j in range(i + 1, p):
        b = X_union_std.iloc[:, j].values
        rho, pv = spearmanr(a, b)
        pairs.append((all_features[i], all_features[j]))
        rhos.append(rho)
        pvals.append(pv)

pvals = np.asarray(pvals)
reject, qvals, _, _ = multipletests(pvals, alpha=FDR_Q, method="fdr_bh")


edges = []
for (u, v), rho, q, ok in zip(pairs, rhos, qvals, reject):
    if ok and (abs(rho) >= RHO_THR):
        edges.append((u, v, rho, q))

edges_df = pd.DataFrame(edges, columns=["u", "v", "rho", "q_fdr"]).sort_values(
    "rho", key=lambda s: np.abs(s), ascending=False
)
edges_df.to_csv(OUT_EDGES_CSV, index=False)
print(f"Edges kept = {len(edges_df)} / {len(pairs)} (|rho|>={RHO_THR} & FDR<{FDR_Q})")
print("[Saved]", OUT_EDGES_CSV)


def modality_of_feature(fname):
    if fname.startswith("CP_"):
        return "CP"
    if fname.startswith("QP_"):
        return "QP"
    if fname.startswith("Clin_"):
        return "Clin"
    return "Other"


node_color_map = {
    "CP": "#1f77b4",
    "QP": "#ff7f0e",
    "Clin": "#2ca02c",
    "Other": "#7f7f7f",
}

G = nx.Graph()
for f in all_features:
    mod = modality_of_feature(f)
    G.add_node(f, modality=mod)

for _, r in edges_df.iterrows():
    G.add_edge(
        r["u"],
        r["v"],
        rho=float(r["rho"]),
        weight=abs(float(r["rho"])),
        q=float(r["q_fdr"]),
    )

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=SEED, k=0.45)

node_colors = [node_color_map[G.nodes[n]["modality"]] for n in G.nodes()]
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=220,
    alpha=0.95,
    edgecolors="black",
    linewidths=0.25,
)

if G.number_of_edges() > 0:
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    w_min, w_max = min(weights), max(weights)
    widths = [0.5 + 4.0 * (w - w_min) / (w_max - w_min + 1e-9) for w in weights]
    edge_colors = ["#d62728" if G[u][v]["rho"] < 0 else "#1f77b4" for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_colors, alpha=0.55)
else:
    print(
        "WARNING: no edges to draw (consider lowering RHO_THR to 0.4 or relaxing FDR)."
    )


degrees = dict(G.degree())
top_label_n = min(30, len(degrees))
label_nodes = sorted(degrees, key=lambda k: degrees[k], reverse=True)[:top_label_n]
nx.draw_networkx_labels(G, pos, labels={n: n for n in label_nodes}, font_size=7)


import matplotlib.patches as mpatches

legend_handles = [
    mpatches.Patch(color=node_color_map["CP"], label="CP features"),
    mpatches.Patch(color=node_color_map["QP"], label="QP features"),
    mpatches.Patch(color=node_color_map["Clin"], label="Clinical features"),
    mpatches.Patch(color="#1f77b4", label="Positive rho (edge)"),
    mpatches.Patch(color="#d62728", label="Negative rho (edge)"),
]
plt.legend(handles=legend_handles[:3], loc="upper right", frameon=True)
plt.title(
    f"Cross-modality Spearman network (union of selected CP/QP + Clin)\nEdges: |rho|≥{RHO_THR}, FDR(BH)<{FDR_Q} | Nodes labeled: top {top_label_n} by degree"
)
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_NET_PNG, dpi=300, bbox_inches="tight")
plt.close()
print("[Saved]", OUT_NET_PNG)


if MAX_HEATMAP_FEATURES is not None and p > MAX_HEATMAP_FEATURES:

    feats_for_heatmap = sorted(degrees, key=lambda k: degrees[k], reverse=True)[
        :MAX_HEATMAP_FEATURES
    ]
else:
    feats_for_heatmap = all_features

corr_mat = X_union_std[feats_for_heatmap].corr(method="spearman")

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_mat, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, linewidths=0.15
)
plt.title(f"Spearman correlation matrix (subset size={len(feats_for_heatmap)})")
plt.tight_layout()
plt.savefig(OUT_HEATMAP_PNG, dpi=300, bbox_inches="tight")
plt.close()
print("[Saved]", OUT_HEATMAP_PNG)

print("\nAll outputs in:", OUT_DIR)
print("Union feature list:", OUT_UNION_CSV)
print("Edges list:", OUT_EDGES_CSV)
