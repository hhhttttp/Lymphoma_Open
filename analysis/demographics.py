"""Plot cohort demographic distributions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import data_path, output_path

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SURV_DATA_PATH = data_path("internal/clinical_survival_200.xlsx")
OUTPUT_DIR = output_path("demographics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("正在读取临床生存表并提取总体人口学指标构成...")

if not os.path.exists(SURV_DATA_PATH):
    raise FileNotFoundError(f"未找到指定的生存表路径，请核对：{SURV_DATA_PATH}")


df_raw = pd.read_excel(SURV_DATA_PATH, engine="openpyxl")


age_col = "AgeGroup（≥60）_y"
gender_col = "Gender_y"

if age_col not in df_raw.columns or gender_col not in df_raw.columns:
    raise KeyError(f"表格中缺少指定的分类列。当前表里的实际列名有：{df_raw.columns.tolist()}")


df_clean = df_raw[[age_col, gender_col]].dropna().copy()
total_n = len(df_clean)
print(f"有效分析样本总数：{total_n}")


age_counts = df_clean[age_col].value_counts()
pct_age_under60 = (age_counts.get(0, 0) / total_n) * 100
pct_age_over60 = (age_counts.get(1, 0) / total_n) * 100


gender_counts = df_clean[gender_col].value_counts()
pct_female = (gender_counts.get(0, 0) / total_n) * 100
pct_male = (gender_counts.get(1, 0) / total_n) * 100


fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

bar_width = 0.50
y_positions = [1, 0]


ax.barh(1, pct_age_under60, height=bar_width, color="#8baec4", edgecolor="none")
ax.barh(
    1,
    pct_age_over60,
    left=pct_age_under60,
    height=bar_width,
    color="#c5853c",
    edgecolor="none",
)


ax.barh(0, pct_female, height=bar_width, color="#2cb289", edgecolor="none")
ax.barh(
    0, pct_male, left=pct_female, height=bar_width, color="#c45a5a", edgecolor="none"
)


ax.text(
    pct_age_under60 / 2,
    1,
    f"Age < 60\n{pct_age_under60:.1f}%",
    va="center",
    ha="center",
    color="white",
    fontsize=11,
    fontweight="bold",
)
ax.text(
    pct_age_under60 + pct_age_over60 / 2,
    1,
    f"Age $\geq$ 60\n{pct_age_over60:.1f}%",
    va="center",
    ha="center",
    color="white",
    fontsize=11,
    fontweight="bold",
)


ax.text(
    pct_female / 2,
    0,
    f"Female\n{pct_female:.1f}%",
    va="center",
    ha="center",
    color="white",
    fontsize=11,
    fontweight="bold",
)
ax.text(
    pct_female + pct_male / 2,
    0,
    f"Male\n{pct_male:.1f}%",
    va="center",
    ha="center",
    color="white",
    fontsize=11,
    fontweight="bold",
)


ax.set_title(
    f"Demographic Baseline Distribution (n = {total_n})",
    fontsize=13,
    fontweight="bold",
    pad=15,
)


ax.set_yticks(y_positions)
ax.set_yticklabels(
    ["Age Group", "Gender"], fontsize=12, fontweight="bold", color="#222222"
)


ax.set_xlabel("Percentage (%)", fontsize=12, fontweight="bold", labelpad=8)
ax.set_xlim(0, 100)
ax.set_xticks([0, 20, 40, 60, 80, 100])


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")
ax.tick_params(axis="both", width=1.5, length=5, labelsize=11)

plt.tight_layout()


output_fig_path = os.path.join(
    OUTPUT_DIR, "Clinical_Demographics_Clean_Distribution.png"
)
plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n[成功] 纯净版总体临床特征构成图已完美保存至: {output_fig_path}")
