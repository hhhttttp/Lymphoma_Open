"""Stratify risk from saved baseline OOF predictions without retraining."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import output_path
from model_artifacts import load_baseline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2_contingency


OUTPUT_DIR = output_path("fusion_baseline")
SEED = 27

OUT_DIR = os.path.join(OUTPUT_DIR, f"risk_stratification_seed_{SEED}")
os.makedirs(OUT_DIR, exist_ok=True)


MODEL_ORDER = ["PSF", "PLAF", "MEFF", "QP", "CP", "Clin"]


STRAT_MODE = "tertiles"


def stratify_by_quantiles(scores, mode="tertiles"):
    scores = np.asarray(scores).astype(float)
    if mode == "tertiles":
        qs = [0.33, 0.67]
        cuts = np.quantile(scores, qs)
        bins = [-np.inf, cuts[0], cuts[1], np.inf]
        labels = ["Low", "Intermediate", "High"]
    elif mode == "quartiles":
        qs = [0.25, 0.50, 0.75]
        cuts = np.quantile(scores, qs)
        bins = [-np.inf, cuts[0], cuts[1], cuts[2], np.inf]
        labels = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    else:
        raise ValueError("mode must be 'tertiles' or 'quartiles'")
    bins = np.unique(bins)
    if len(bins) < 3:
        raise ValueError("Too few distinct score quantiles for the requested risk groups.")
    if len(labels) != len(bins) - 1:
        labels = [f"Group {i + 1}" for i in range(len(bins) - 1)]
    group = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
    return group, labels, bins


def event_rate_and_ci(y_true, mask):

    y = np.asarray(y_true).astype(int)
    n = int(np.sum(mask))
    if n == 0:
        return np.nan, np.nan, np.nan, 0, 0
    k = int(np.sum(y[mask] == 1))
    rate = k / n
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return rate, lo, hi, k, n


def plot_event_rate_bar(model_name, df_groups, out_png):

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    x = np.arange(len(df_groups))
    rates = df_groups["rate"].values
    yerr = np.vstack([rates - df_groups["lo"].values, df_groups["hi"].values - rates])

    ax.bar(
        x,
        rates,
        yerr=yerr,
        capsize=6,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df_groups["group"].tolist())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Event rate (Positive proportion)")
    ax.set_title(f"Risk stratification by predicted score ({model_name})")

    for i, r in df_groups.iterrows():
        ax.text(
            i,
            min(0.98, r["rate"] + 0.05),
            f"n={r['n']}\npos={r['k']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_score_box(model_name, scores, groups, labels, out_png):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    data = [np.asarray(scores)[groups == lab] for lab in labels]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Predicted risk score (OOF probability)")
    ax.set_title(f"Score distribution by risk group ({model_name})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


saved = load_baseline(SEED)
probabilities = saved["probabilities"]
y_true = saved["y_true"]
available_models = list(probabilities)

models_to_plot = [m for m in MODEL_ORDER if m in available_models]
if len(models_to_plot) == 0:
    models_to_plot = available_models

print("将进行风险分层的模型：", models_to_plot)
print("STRAT_MODE =", STRAT_MODE)

summary_rows = []


for m in models_to_plot:
    scores = np.asarray(probabilities[m]).astype(float)

    if np.unique(scores).size < 2:
        print(f"Skipping {m}: all saved OOF scores are identical.")
        continue
    groups, labels, bins = stratify_by_quantiles(scores, mode=STRAT_MODE)
    groups = groups.astype(str)

    rows = []
    for lab in labels:
        mask = groups == lab
        rate, lo, hi, k, n = event_rate_and_ci(y_true, mask)
        rows.append({"group": lab, "n": n, "k": k, "rate": rate, "lo": lo, "hi": hi})

    df_g = pd.DataFrame(rows)

    table = np.array([[r["k"], r["n"] - r["k"]] for _, r in df_g.iterrows()])
    nonempty = table.sum(axis=1) > 0
    p = (chi2_contingency(table[nonempty])[1]
         if nonempty.sum() >= 2 and (table.sum(axis=0) > 0).all() else np.nan)

    bar_png = os.path.join(OUT_DIR, f"risk_strat_bar_{m}_{STRAT_MODE}.png")
    box_png = os.path.join(OUT_DIR, f"risk_strat_box_{m}_{STRAT_MODE}.png")

    plot_event_rate_bar(m, df_g, bar_png)
    plot_score_box(m, scores, groups, labels, box_png)

    detail_csv = os.path.join(OUT_DIR, f"risk_strat_table_{m}_{STRAT_MODE}.csv")
    df_g.to_csv(detail_csv, index=False)

    summary_rows.append(
        {
            "Model": m,
            "StratMode": STRAT_MODE,
            "Chi2_p": p,
            "GroupCounts": "; ".join(
                [
                    f"{r['group']} n={int(r['n'])} pos={int(r['k'])} rate={r['rate']:.3f}"
                    for _, r in df_g.iterrows()
                ]
            ),
        }
    )

    print(f"[{m}] chi2 p={p:.4g} | saved: {bar_png}, {box_png}, {detail_csv}")


summary_df = pd.DataFrame(summary_rows, columns=["Model", "StratMode", "Chi2_p", "GroupCounts"]).sort_values("Chi2_p")
summary_csv = os.path.join(OUT_DIR, f"risk_strat_summary_{STRAT_MODE}.csv")
summary_df.to_csv(summary_csv, index=False)
print("\n汇总已保存：", summary_csv)
