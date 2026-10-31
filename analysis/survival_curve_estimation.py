"""Estimate survival curves using the saved baseline multimodal PSF OOF scores."""

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

from lifelines import KaplanMeierFitter
from scipy import stats
OUTPUT_DIR = output_path("survival_curve_estimation")
SURV_PATH = data_path("internal/clinical_survival_200.xlsx")
TIME_POINTS_MONTHS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
# Input OS and evaluation times both use months; no conversion is needed.
Y_DEFINITION = "group_diff"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_survival(path):
    df = pd.read_excel(path, engine="openpyxl")
    df["Merge_ID"] = (
        df["Image"].astype(str).str.replace(".svs", "", regex=False).str.strip()
    )
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["event"] = binary_labels(
        pd.to_numeric(df["OS_status"], errors="raise"), "OS_status"
    )
    validate_ids(
        df,
        "Merge_ID"
        if "Merge_ID" in df
        else "Patient_ID"
        if "Patient_ID" in df
        else "Image",
        path,
    )
    return df.dropna(subset=["Merge_ID", "OS"])[["Merge_ID", "OS", "event"]]


df_surv = load_survival(SURV_PATH)
df_pred = pd.DataFrame(
    {"Merge_ID": df_final["Merge_ID"].values, "PSF_prob": probs["PSF"]}
)
df_all = df_surv.merge(df_pred, on="Merge_ID", how="inner", validate="one_to_one").dropna()
print(f"\n生存队列对齐样本数: {len(df_all)} | 事件数: {int(df_all['event'].sum())}")

thr = pick_threshold_by_youden(true_labels, probs["PSF"])
df_all["Group"] = np.where(df_all["PSF_prob"] >= thr, "High risk", "Low risk")
print(f"PSF Youden 阈值: {thr:.4f}")
print(df_all["Group"].value_counts().to_dict())


kmf_high = KaplanMeierFitter().fit(
    df_all.loc[df_all.Group == "High risk", "OS"],
    event_observed=df_all.loc[df_all.Group == "High risk", "event"],
)
kmf_low = KaplanMeierFitter().fit(
    df_all.loc[df_all.Group == "Low risk", "OS"],
    event_observed=df_all.loc[df_all.Group == "Low risk", "event"],
)
kmf_all = KaplanMeierFitter().fit(df_all["OS"], event_observed=df_all["event"])

rows = []
for month in TIME_POINTS_MONTHS:
    s_high = float(kmf_high.predict(month))
    s_low = float(kmf_low.predict(month))
    s_all = float(kmf_all.predict(month))
    n_risk_high = int((df_all.loc[df_all.Group == "High risk", "OS"] >= month).sum())
    n_risk_low = int((df_all.loc[df_all.Group == "Low risk", "OS"] >= month).sum())
    rows.append(
        {
            "Time_months": month,
            "N_at_risk_high": n_risk_high,
            "N_at_risk_low": n_risk_low,
            "KM_high": s_high,
            "KM_low": s_low,
            "KM_overall": s_all,
            "death_high": 1 - s_high,
            "death_low": 1 - s_low,
            "Y_group_diff": (1 - s_high) - (1 - s_low),
        }
    )

df_curve = pd.DataFrame(rows)
print("\n=== 各月份两组累积死亡比例（OS 和评价时间均以月为单位） ===")
print(
    df_curve[
        [
            "Time_months",
            "N_at_risk_high",
            "N_at_risk_low",
            "death_high",
            "death_low",
            "Y_group_diff",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# Curve-equation coefficients describe time in months.
X = df_curve["Time_months"].to_numpy(float)
Y = (
    df_curve["Y_group_diff"]
    if Y_DEFINITION == "group_diff"
    else 1 - df_curve["KM_overall"]
).to_numpy(float)

if np.any(Y <= 0):
    raise ValueError(
        f"对数族模型要求 Y>0，当前最小 Y = {Y.min():.4f}；请调整 TIME_POINTS_MONTHS 或 Y_DEFINITION"
    )


def fit_poly(X, Y, degree):
    """多项式 OLS，返回 (参数, R2, F, p)。参数顺序 [常数, X, X^2, ...]"""
    A = np.column_stack([X**k for k in range(degree + 1)])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    k = degree
    df_res = len(Y) - k - 1
    F = (r2 / k) / ((1 - r2) / df_res) if (1 - r2) > 0 and df_res > 0 else np.nan
    p = 1 - stats.f.cdf(F, k, df_res) if np.isfinite(F) else np.nan
    return coef, r2, F, p


def fit_loglinear(X, Y, kind):
    """对数族：ln(Y) = a + bX 上做 OLS，再换算成原参数。"""
    lny = np.log(Y)
    b, a = np.polyfit(X, lny, 1)
    pred_log = a + b * X
    ss_res = float(np.sum((lny - pred_log) ** 2))
    ss_tot = float(np.sum((lny - lny.mean()) ** 2))
    r2_log = 1 - ss_res / ss_tot

    pred_orig = np.exp(pred_log)
    ss_res_o = float(np.sum((Y - pred_orig) ** 2))
    ss_tot_o = float(np.sum((Y - Y.mean()) ** 2))
    r2_orig = 1 - ss_res_o / ss_tot_o

    df_res = len(Y) - 2
    F = (r2_log / 1) / ((1 - r2_log) / df_res) if r2_log < 1 else np.nan
    p = 1 - stats.f.cdf(F, 1, df_res) if np.isfinite(F) else np.nan

    if kind == "Compound":
        return [np.exp(a), np.exp(b)], r2_log, r2_orig, F, p
    if kind == "Growth":
        return [a, b], r2_log, r2_orig, F, p
    return [np.exp(a), b], r2_log, r2_orig, F, p


print("\n" + "=" * 110)
print("Model Summary".center(46) + "|" + "Parameter Estimates".center(60))
print(
    f"{'Equation':<12}{'R Square':>10}{'F':>12}{'p Value':>11}   |{'Constant':>12}{'B1':>12}{'B2':>12}{'B3':>12}"
)
print("-" * 110)

out = []
for name, kind in [
    ("Linear", "poly1"),
    ("Quadratic", "poly2"),
    ("Cubic", "poly3"),
    ("Compound", "Compound"),
    ("Growth", "Growth"),
    ("Exponential", "Exponential"),
]:
    if kind.startswith("poly"):
        deg = int(kind[-1])
        coef, r2, F, p = fit_poly(X, Y, deg)
        const = coef[0]
        bs = list(coef[1:]) + [np.nan] * (3 - deg)
        r2_log = np.nan
    else:
        bs_all, r2_log, r2, F, p = fit_loglinear(X, Y, kind)
        const = bs_all[0]
        bs = [bs_all[1], np.nan, np.nan]

    out.append(
        {
            "Equation": name,
            "Time_unit": "months",
            "R Square": r2,
            "F": F,
            "p Value": p,
            "Constant": const,
            "B1": bs[0],
            "B2": bs[1],
            "B3": bs[2],
            "R2_log_scale": r2_log,
        }
    )

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if v != 0 and abs(v) < 0.001:
            return f"{v:.3e}"
        return f"{v:.3f}"

    print(
        f"{name:<12}{r2:>10.3f}{F:>12.3f}{'<0.001' if p < 0.001 else f'{p:.3f}':>11}   |"
        f"{fmt(const):>12}{fmt(bs[0]):>12}{fmt(bs[1]):>12}{fmt(bs[2]):>12}"
    )

print("=" * 110)


dups = df_curve["Y_group_diff"].round(6).duplicated(keep=False)
if dups.any():
    tt = df_curve.loc[dups, "Time_months"].tolist()
    print(f"\n[提示] 以下月份的 Y 完全相同（KM 曲线已走平）: {tt}")
    print("       这些重复点会人为抬高 R²，且不提供新信息；")
    print("       如需更稳健的拟合，建议把 TIME_POINTS_MONTHS 截断到曲线仍在变化的区间。")

df_out = pd.DataFrame(out)
csv_data = os.path.join(OUTPUT_DIR, "CurveEstimation_Time_Difference.csv")
csv_table = os.path.join(OUTPUT_DIR, "CurveEstimation_Model_Summary.csv")
df_curve.to_csv(csv_data, index=False)
df_out.to_csv(csv_table, index=False)

print(f"\n(时点, 组间差异) 数据已保存: {csv_data}")
print(f"曲线估计结果表已保存: {csv_table}")
print(f"\nY 定义: {Y_DEFINITION} | 分界阈值(Youden): {thr:.4f}")
