"""Analyze internal OS in months using saved baseline OOF scores; no classifier fitting."""

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

OUTPUT_DIR = output_path("survival")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SURV_PATH = data_path("internal/clinical_survival_200.xlsx")


SURV_OUT_DIR = os.path.join(OUTPUT_DIR, "Survival_OS_Eval")
os.makedirs(SURV_OUT_DIR, exist_ok=True)


from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
)


def load_survival_os(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine="openpyxl")

    required = ["Image", "OS", "OS_status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"生存表缺少列: {missing}；实际列: {df.columns.tolist()}")

    df["Merge_ID"] = (
        df["Image"].astype(str).str.replace(".svs", "", regex=False).str.strip()
    )
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS_status"] = pd.to_numeric(df["OS_status"], errors="coerce")
    df = df.dropna(subset=["Merge_ID", "OS", "OS_status"])

    df["event"] = binary_labels(df["OS_status"], "OS_status").astype(bool)
    df["OS"] = df["OS"].astype(float)

    validate_ids(
        df,
        "Merge_ID"
        if "Merge_ID" in df
        else "Patient_ID"
        if "Patient_ID" in df
        else "Image",
        path,
    )
    return df[["Merge_ID", "OS", "event"]]


df_surv = load_survival_os(SURV_PATH)


df_pred = pd.DataFrame({"Merge_ID": df_final["Merge_ID"].values})
for k in probs.keys():
    df_pred[k] = probs[k]


df_all = df_surv.merge(df_pred, on="Merge_ID", how="inner", validate="one_to_one").dropna()
print("Survival aligned n =", len(df_all))


if len(df_all) < 20:
    print(df_all.head())
    raise ValueError("对齐到生存表的样本数过少，请检查 Merge_ID 与 Image 是否一致（是否含 .svs、空格等）。")


# OS is recorded in months: one-, three- and five-year horizons.
t1, t3, t5 = 12.0, 36.0, 60.0
eval_times = np.array([t1, t3, t5])


def plot_km_median_split(d, score_col, out_png, title_prefix=""):
    dd = d[["OS", "event", score_col]].dropna().copy()
    thr = dd[score_col].median()
    dd["Group"] = np.where(dd[score_col] >= thr, "High risk", "Low risk")

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(9, 7))
    ax = None

    for g in ["Low risk", "High risk"]:
        dg = dd[dd["Group"] == g]
        kmf.fit(
            durations=dg["OS"], event_observed=dg["event"], label=f"{g} (n={len(dg)})"
        )
        ax = kmf.plot(ci_show=True, linewidth=2.6, ax=ax)

    d1 = dd[dd["Group"] == "Low risk"]
    d2 = dd[dd["Group"] == "High risk"]
    lr = logrank_test(
        d1["OS"], d2["OS"], event_observed_A=d1["event"], event_observed_B=d2["event"]
    )
    p_text = f"log-rank p = {lr.p_value:.4g}"

    ax.set_title(
        f"{title_prefix} OS KM by risk ({score_col}, median split)\n{p_text}",
        fontsize=14,
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()

    return float(lr.p_value), float(thr)


def cox_univariate(d, score_col):
    dd = d[["OS", "event", score_col]].dropna().copy()

    dd["event"] = dd["event"].astype(int)

    cph = CoxPHFitter()
    cph.fit(dd, duration_col="OS", event_col="event")
    s = cph.summary.loc[score_col]

    hr = float(np.exp(s["coef"]))
    ci_low = float(np.exp(s["coef lower 95%"]))
    ci_high = float(np.exp(s["coef upper 95%"]))
    p = float(s["p"])
    return hr, ci_low, ci_high, p


def calc_cindex(d, score_col):
    dd = d[["OS", "event", score_col]].dropna().copy()

    ci = concordance_index_censored(
        dd["event"].values.astype(bool),
        dd["OS"].values.astype(float),
        dd[score_col].values.astype(float),
    )
    return float(ci[0])


def time_dependent_auc_1_3_5(d, score_col):
    dd = d[["OS", "event", score_col]].dropna().copy()
    if np.any(eval_times <= dd["OS"].min()) or np.any(eval_times >= dd["OS"].max()):
        raise ValueError(
            "AUC horizons must lie strictly inside observed follow-up; adjust eval_times."
        )

    y_s = Surv.from_arrays(
        event=dd["event"].values.astype(bool), time=dd["OS"].values.astype(float)
    )

    times = eval_times.copy()
    aucs, mean_auc = cumulative_dynamic_auc(
        y_s, y_s, dd[score_col].values.astype(float), times
    )
    return aucs


def plot_time_auc(aucs, out_png, title):
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(eval_times / 12.0, aucs, marker="o", linewidth=2.5)
    plt.xticks([1, 3, 5], ["1y", "3y", "5y"])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Time horizon (years)")
    plt.ylabel("Time-dependent AUC")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()


def calibration_plot_at_time(
    d, score_col, t_eval_months=36.0, n_bins=10, out_png=None, title=None
):
    dd = d[["OS", "event", score_col]].dropna().copy()
    dd["event"] = dd["event"].astype(int)

    cph = CoxPHFitter()
    cph.fit(dd, duration_col="OS", event_col="event")

    sf = cph.predict_survival_function(dd, times=[t_eval_months])
    pred_surv = sf.values.flatten()
    pred_risk = 1.0 - pred_surv

    dd = dd.assign(pred_surv=pred_surv, pred_risk=pred_risk)

    dd["bin"] = pd.qcut(dd["pred_surv"], q=n_bins, duplicates="drop")

    kmf = KaplanMeierFitter()
    rows = []
    for b, g in dd.groupby("bin"):
        if len(g) < 5:
            continue
        kmf.fit(g["OS"], event_observed=g["event"])

        obs_surv = float(kmf.predict(t_eval_months))
        rows.append(
            {
                "n": len(g),
                "pred_surv_mean": float(g["pred_surv"].mean()),
                "obs_surv_km": obs_surv,
            }
        )
    cal = pd.DataFrame(
        rows, columns=["n", "pred_surv_mean", "obs_surv_km"]
    ).sort_values("pred_surv_mean")
    if cal.empty:
        raise ValueError(
            "No calibration bins contain at least five samples; reduce n_bins."
        )

    plt.figure(figsize=(6.5, 6.5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal")
    plt.plot(
        cal["pred_surv_mean"],
        cal["obs_surv_km"],
        marker="o",
        linewidth=2.5,
        label="Apparent Cox calibration (same cohort)",
    )
    for _, r in cal.iterrows():
        plt.text(r["pred_surv_mean"], r["obs_surv_km"], f"n={int(r['n'])}", fontsize=8)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(f"Predicted S({t_eval_months/12:.0f}y)")
    plt.ylabel(f"Observed S({t_eval_months/12:.0f}y) (KM)")
    plt.title(
        title if title else f"Calibration at {t_eval_months/12:.0f} years ({score_col})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
    plt.show()

    return cal


models_to_eval = ["Clin", "CP", "QP", "MEFF", "PSF", "PLAF"]
summary_rows = []

for m in models_to_eval:
    print("\n" + "=" * 80)
    print("Survival evaluation for model:", m)

    km_png = os.path.join(SURV_OUT_DIR, f"KM_OS_{m}_median.png")
    logrank_p, thr = plot_km_median_split(df_all, m, km_png, title_prefix="")

    hr, hr_lo, hr_hi, cox_p = cox_univariate(df_all, m)

    cidx = calc_cindex(df_all, m)

    aucs = time_dependent_auc_1_3_5(df_all, m)
    auc_png = os.path.join(SURV_OUT_DIR, f"tAUC_1_3_5y_{m}.png")
    plot_time_auc(aucs, auc_png, title=f"Time-dependent AUC (OOF risk) - {m}")

    cal_png = os.path.join(SURV_OUT_DIR, f"Calibration_3y_{m}.png")
    cal_tbl = calibration_plot_at_time(
        df_all,
        m,
        t_eval_months=t3,
        n_bins=10,
        out_png=cal_png,
        title=f"Apparent Cox calibration at 3 years (Cox on risk_score) - {m}",
    )
    cal_csv = os.path.join(SURV_OUT_DIR, f"Calibration_3y_{m}.csv")
    cal_tbl.to_csv(cal_csv, index=False)

    summary_rows.append(
        {
            "Model": m,
            "Aligned_n": len(df_all),
            "KM_median_threshold": thr,
            "Logrank_p": logrank_p,
            "Cox_HR_per_1unit": hr,
            "Cox_HR_95CI_low": hr_lo,
            "Cox_HR_95CI_high": hr_hi,
            "Cox_p": cox_p,
            "C_index": cidx,
            "AUC_1y": float(aucs[0]),
            "AUC_3y": float(aucs[1]),
            "AUC_5y": float(aucs[2]),
        }
    )


df_surv_summary = pd.DataFrame(summary_rows).sort_values("C_index", ascending=False)
summary_csv = os.path.join(SURV_OUT_DIR, "Survival_Summary_OS.csv")
df_surv_summary.to_csv(summary_csv, index=False)

print("\nSaved survival summary:", summary_csv)
print(df_surv_summary)
