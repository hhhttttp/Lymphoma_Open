"""Fit adjusted Cox associations of saved baseline multimodal PSF scores; OS is in months."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import data_path, output_path
from model_artifacts import load_baseline, youden_threshold
from workflow_utils import binary_labels, validate_ids

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
pick_threshold_by_youden = youden_threshold

from lifelines import CoxPHFitter
OUTPUT_DIR = output_path("cox")
SURV_PATH = data_path("internal/clinical_survival_200.xlsx")
EXCLUDE_VARS = ["c-myc(40%)", "MYD88"]
DROP_EXCLUDED_FROM_COHORT = os.environ.get("DROP_EXCLUDED_FROM_COHORT", "0") == "1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_survival_os_with_covariates(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine="openpyxl")

    required = [
        "Image",
        "OS",
        "OS_status",
        "Bsympstoms",
        "CD10",
        "COO",
        "HS",
        "MYD88",
        "c-myc(40%)",
        "IPI",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"生存表缺少必要列: {missing}；实际拥有的列: {df.columns.tolist()}")

    df["Merge_ID"] = (
        df["Image"].astype(str).str.replace(".svs", "", regex=False).str.strip()
    )
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS_status"] = pd.to_numeric(df["OS_status"], errors="coerce")

    covariates = ["Bsympstoms", "CD10", "COO", "HS", "MYD88", "c-myc(40%)", "IPI"]
    if DROP_EXCLUDED_FROM_COHORT:
        covariates = [c for c in covariates if c not in EXCLUDE_VARS]
    for cov in covariates:
        df[cov] = pd.to_numeric(df[cov], errors="coerce")

    df = df.dropna(subset=["Merge_ID", "OS", "OS_status"] + covariates)
    df["event"] = binary_labels(df["OS_status"], "OS_status").astype(bool)

    validate_ids(
        df,
        "Merge_ID"
        if "Merge_ID" in df
        else "Patient_ID"
        if "Patient_ID" in df
        else "Image",
        path,
    )
    return df[["Merge_ID", "OS", "event"] + covariates]


df_surv = load_survival_os_with_covariates(SURV_PATH)

df_pred = pd.DataFrame(
    {"Merge_ID": df_final["Merge_ID"].values, "PSF_prob": probs["PSF"]}
)
df_all = df_surv.merge(df_pred, on="Merge_ID", how="inner", validate="one_to_one").dropna()
print(f"生存队列最终对齐样本数: {len(df_all)}")


psf_threshold = pick_threshold_by_youden(true_labels, probs["PSF"])
df_all["PSF"] = (df_all["PSF_prob"] >= psf_threshold).astype(int)
print(f"PSF Youden 分类阈值: {psf_threshold:.4f}, 高风险样本数: {df_all['PSF'].sum()}")


print("\n正在运行多因素 Cox 回归分析 (Multivariate Cox Analysis)...")

multivariate_cols = [
    "OS",
    "event",
    "PSF",
    "Bsympstoms",
    "CD10",
    "COO",
    "HS",
    "MYD88",
    "c-myc(40%)",
    "IPI",
]
multivariate_cols = [c for c in multivariate_cols if c not in EXCLUDE_VARS]
df_multi = df_all[multivariate_cols].copy()
df_multi["event"] = df_multi["event"].astype(int)

cph_multi = CoxPHFitter()
cph_multi.fit(df_multi, duration_col="OS", event_col="event")

multi_summary = cph_multi.summary[
    ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
]
multi_summary = multi_summary.rename(
    columns={
        "exp(coef)": "HR",
        "exp(coef) lower 95%": "HR_lower",
        "exp(coef) upper 95%": "HR_upper",
        "p": "p_value",
    }
)


def get_p_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


multi_summary["stars"] = multi_summary["p_value"].apply(get_p_stars)

_sfx = (
    "nocmyc_nomyd88_drop" if DROP_EXCLUDED_FROM_COHORT else "nocmyc_nomyd88_keepCohort"
)
OUT_MULTI_CSV = os.path.join(OUTPUT_DIR, f"Multivariate_Cox_Summary_{_sfx}.csv")
multi_summary.to_csv(OUT_MULTI_CSV)


multi_summary = multi_summary.iloc[::-1]

display_label_map = {
    "PSF": "PSF (Baseline multimodal score)",
    "Bsympstoms": "Bsympstoms",
    "CD10": "CD10",
    "COO": "COO",
    "HS": "Hensen Score",
    "IPI": "IPI",
}


fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=300)
y_pos = np.arange(len(multi_summary))


ax.set_yticks(y_pos)
formatted_labels = [display_label_map.get(name, name) for name in multi_summary.index]
ax.set_yticklabels(formatted_labels, fontsize=12, fontweight="bold", color="#222222")

ax.set_xscale("log")
x_min, x_max = 0.1, 15.0
ax.set_xlim(x_min, x_max)


ax.set_ylim(-0.6, len(multi_summary) - 0.4)


ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(axis="both", which="major", labelsize=11, width=1.5, length=5)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")


ax.axvline(x=1.0, color="#D95319", linestyle="--", linewidth=1.5, alpha=0.9)


for i, (var_name, row) in enumerate(multi_summary.iterrows()):
    is_psf = var_name == "PSF"
    color = "black"

    ax.errorbar(
        x=row["HR"],
        y=i,
        xerr=[[row["HR"] - row["HR_lower"]], [row["HR_upper"] - row["HR"]]],
        fmt="o",
        color=color,
        markersize=5,
        elinewidth=1.8,
        capsize=4,
        capthick=1.8,
        zorder=4,
    )

    hr_ci_text = f"{row['HR']:.2f} ({row['HR_lower']:.2f}–{row['HR_upper']:.2f})"

    ax.text(
        x=row["HR"],
        y=i - 0.24,
        s=hr_ci_text,
        va="top",
        ha="center",
        fontsize=11,
        color="#111111",
        fontweight="bold" if is_psf else "normal",
    )

    if row["stars"] != "":
        ax.text(
            x=row["HR_upper"] * 1.05,
            y=i,
            s=row["stars"],
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="black",
        )


ax.set_xlabel("Hazard Ratio (log scale)", fontsize=13, fontweight="bold", labelpad=12)

plt.tight_layout()
OUT_FOREST_PNG = os.path.join(
    OUTPUT_DIR, f"Multivariate_Survival_Forest_Plot_{_sfx}.png"
)
plt.savefig(OUT_FOREST_PNG, dpi=300, bbox_inches="tight")
plt.close()

print(f"无 c-myc + 无 MYD88 森林图已保存至: {OUT_FOREST_PNG}")
print(
    f"队列口径: {'去掉剔除变量筛选' if DROP_EXCLUDED_FROM_COHORT else '沿用原口径'} | N = {len(df_all)} | 事件 = {int(df_all['event'].sum())}"
)
