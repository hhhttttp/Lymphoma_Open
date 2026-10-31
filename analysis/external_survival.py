"""Evaluate external survival with the saved baseline fold ensemble; no classifier fit."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import data_path, output_path
from model_artifacts import load_baseline, load_external_inputs, predict_saved, MODELS
from workflow_utils import binary_labels, validate_ids
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

SEED = 27
OUTPUT_DIR = output_path("external_survival")

def km_logrank(dd, thr, split_name, model_name, out_png):
    """dd 需含列: OS, event, score；返回 log-rank p 值"""
    d = dd[["OS", "event", "score"]].dropna().copy()
    d["Group"] = np.where(d["score"] >= thr, "High risk", "Low risk")

    d_low = d[d["Group"] == "Low risk"]
    d_high = d[d["Group"] == "High risk"]
    if len(d_low) < 2 or len(d_high) < 2:
        print(f"[警告] {model_name} ({split_name}) 分组样本过少，跳过")
        return np.nan

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(9, 7))
    ax = None
    for g in ["Low risk", "High risk"]:
        dg = d[d["Group"] == g]
        kmf.fit(
            durations=dg["OS"], event_observed=dg["event"], label=f"{g} (n={len(dg)})"
        )
        ax = kmf.plot(ci_show=True, linewidth=2.6, ax=ax)

    lr = logrank_test(
        d_low["OS"],
        d_high["OS"],
        event_observed_A=d_low["event"],
        event_observed_B=d_high["event"],
    )

    ax.set_title(
        f"TCGA OS KM by {model_name} risk ({split_name} split)\n"
        f"threshold={thr:.3f}, log-rank p = {lr.p_value:.4g}",
        fontsize=13,
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[KM 图已保存] {out_png}  log-rank p = {lr.p_value:.4g}")
    return float(lr.p_value)


def cox_univariate(dd, score_col="score"):
    d = dd[["OS", "event", score_col]].dropna().copy()
    d["event"] = d["event"].astype(int)
    cph = CoxPHFitter()
    cph.fit(d, duration_col="OS", event_col="event")
    s = cph.summary.loc[score_col]
    hr = float(np.exp(s["coef"]))
    ci_low = float(np.exp(s["coef lower 95%"]))
    ci_high = float(np.exp(s["coef upper 95%"]))
    return hr, ci_low, ci_high, float(s["p"])


def calc_cindex(dd):
    d = dd[["OS", "event", "score"]].dropna().copy()

    return float(concordance_index(d["OS"], -d["score"], d["event"].astype(int)))


def main():
    saved = load_baseline(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    external = load_external_inputs(
        saved, data_path("external/linba_TCGA_cellprofiler.csv"),
        data_path("external/linba_TCGA_qupath.csv"), data_path("external/TCGA临床1.xlsx"),
    )
    scores = predict_saved(saved, external)
    slide_predictions = external[["Merge_ID", "Patient_ID"]].copy()
    for model in MODELS:
        slide_predictions[model] = scores[model]
    slide_predictions["run_id"] = saved["run_id"]
    slide_predictions.to_csv(os.path.join(OUTPUT_DIR, "TCGA_slide_predictions.csv"), index=False)
    # One row per patient: average slide probabilities before joining survival.
    patient_predictions = slide_predictions.groupby("Patient_ID", as_index=False)[list(MODELS)].mean()
    survival_path = data_path("external/TCGA标签.xlsx")
    survival = pd.read_excel(survival_path, engine="openpyxl")
    if survival["Image"].isna().any():
        raise ValueError("External survival Image IDs must be nonmissing.")
    survival["Patient_ID"] = survival["Image"].astype(str).str.strip().str[:12]
    validate_ids(survival, "Patient_ID", survival_path)
    survival["OS"] = pd.to_numeric(survival["day"], errors="raise")
    survival = survival.dropna(subset=["OS", "生存状态"])
    survival["event"] = binary_labels(survival["生存状态"], survival_path)
    survival = survival.loc[survival.OS > 0, ["Patient_ID", "OS", "event"]]
    cohort = patient_predictions.merge(survival, on="Patient_ID", validate="one_to_one")
    if cohort.empty:
        raise ValueError("No external predictions align with survival records.")
    cohort["run_id"] = saved["run_id"]
    cohort.to_csv(os.path.join(OUTPUT_DIR, "TCGA_sample_level_predictions_survival.csv"), index=False)
    rows = []
    for model in MODELS:
        dd = cohort[["OS", "event", model]].rename(columns={model: "score"})
        cutoff = saved["thresholds"][model]
        median = float(dd.score.median())
        p_youden = km_logrank(dd, cutoff, "development OOF Youden", model,
                             os.path.join(OUTPUT_DIR, f"TCGA_KM_{model}_Youden_split.png"))
        p_median = km_logrank(dd, median, "external median (descriptive)", model,
                             os.path.join(OUTPUT_DIR, f"TCGA_KM_{model}_median_split.png"))
        if dd.score.nunique() < 2 or not dd.event.any():
            hr = lo = hi = cox_p = cindex = np.nan
        else:
            hr, lo, hi, cox_p = cox_univariate(dd)
            cindex = calc_cindex(dd)
        rows.append({"Model": model, "N": len(dd), "Events": int(dd.event.sum()),
                     "Youden_threshold": cutoff, "Logrank_p_Youden": p_youden,
                     "Median_threshold": median, "Logrank_p_median": p_median,
                     "Cox_HR": hr, "Cox_HR_95CI_low": lo, "Cox_HR_95CI_high": hi,
                     "Cox_p": cox_p, "C_index": cindex, "run_id": saved["run_id"]})
    summary = pd.DataFrame(rows).sort_values("C_index", ascending=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "TCGA_Survival_KM_Summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
