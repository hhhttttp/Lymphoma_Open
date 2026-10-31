"""Evaluate the ABC subtype prediction task."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from settings import data_path, output_path
from workflow_utils import measurement_columns
from workflow_utils import validate_feature_pair, validate_ids, impute_for_fold

import os
import random
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


OUTPUT_DIR = output_path("abc_subtype")
CP_PATH = data_path("internal/linba_cellprofiler_full.csv")
QP_PATH = data_path("internal/linba_qupath_full.csv")

validate_feature_pair(CP_PATH, QP_PATH)
CLIN_PATH = data_path("internal/linba_linchuang_with_3new_target.xlsx")


TARGET_COL = "ABC型"
SEED = 72
OUTER_FOLDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_img(path, prefix):
    df = pd.read_csv(path)
    id_col = df.columns[0]
    lbl_col = "label" if "label" in df.columns else df.columns[1]
    df["Merge_ID"] = (
        df[id_col].astype(str).str.replace(".svs", "", regex=False).str.strip()
    )
    feat_cols = measurement_columns(
        sorted([c for c in df.columns if c not in [id_col, lbl_col, "Merge_ID"]])
    )
    df = df.rename(columns={c: f"{prefix}_{c}" for c in feat_cols})
    final_cols = [f"{prefix}_{c}" for c in feat_cols]
    return df[["Merge_ID"] + final_cols], final_cols


def load_clin(path):
    df = pd.read_excel(path, engine="openpyxl")
    col_mapping = {
        "Image": "Merge_ID",
        "Gender": "Clin_Gender",
        "AgeGroup（≥60）": "Clin_AgeGroup",
        "Extranodalinvolvement": "Clin_Extranodal",
        "Stage Group": "Clin_Stage",
    }
    existing = [c for c in col_mapping if c in df.columns]
    df = df.rename(columns=col_mapping)
    clin_cols = [col_mapping[c] for c in existing if col_mapping[c] != "Merge_ID"]
    df["Merge_ID"] = (
        df["Merge_ID"].astype(str).str.replace(".svs", "", regex=False).str.strip()
    )
    for c in clin_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    validate_ids(
        df,
        "Merge_ID"
        if "Merge_ID" in df
        else "Patient_ID"
        if "Patient_ID" in df
        else "Image",
        path,
    )
    return df[["Merge_ID"] + clin_cols + [TARGET_COL]], clin_cols


print("正在读取数据...")
df_cp, cols_cp = load_img(CP_PATH, "CP")
df_qp, cols_qp = load_img(QP_PATH, "QP")
df_clin, cols_clin = load_clin(CLIN_PATH)

df = pd.merge(df_cp, df_qp[["Merge_ID"] + cols_qp], on="Merge_ID", how="inner")
df = pd.merge(df, df_clin, on="Merge_ID", how="inner")
df = df[df[TARGET_COL].notna()].reset_index(drop=True)

X_cp_full = df[cols_cp].copy()
X_qp_full = df[cols_qp].copy()
X_clin_full = df[cols_clin].fillna(0)
y = df[TARGET_COL].astype(int).values
n = len(y)
print(f"Target: {TARGET_COL} | Effective N = {n} | positives = {int(y.sum())}")


def select_features_strict(X_train, y_train, feature_names, seed):
    k = min(5, sum(y_train == 1) - 1, sum(y_train == 0) - 1)
    if k < 1:
        k = 1
    try:
        smote = SMOTE(k_neighbors=k, random_state=seed)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except Exception as exc:
        warnings.warn(f"Feature-selection fallback: {exc}", RuntimeWarning)
        X_res, y_res = X_train, y_train

    X_res = pd.DataFrame(X_res, columns=feature_names)
    scaler = StandardScaler()
    X_res_std = pd.DataFrame(scaler.fit_transform(X_res), columns=feature_names)

    try:
        clf = LogisticRegression(
            solver="liblinear", penalty="l1", C=1, random_state=seed, n_jobs=1
        )
        clf.fit(X_res_std, y_res)
        keep_uni = X_res_std.columns[np.abs(clf.coef_[0]) > 0].tolist()
        if len(keep_uni) < 2:
            keep_uni = feature_names
    except Exception as exc:
        warnings.warn(f"Feature-selection fallback: {exc}", RuntimeWarning)
        keep_uni = feature_names

    X_uni = X_res_std[keep_uni]
    if X_uni.shape[1] > 1:
        corr = X_uni.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > 0.85)]
        keep_spearman = [c for c in X_uni.columns if c not in drop]
    else:
        keep_spearman = keep_uni

    try:
        lasso = LassoCV(
            cv=5, random_state=seed, max_iter=20000, n_jobs=1, selection="cyclic"
        )
        lasso.fit(X_uni[keep_spearman], y_res)
        feats = X_uni[keep_spearman].columns[np.abs(lasso.coef_) > 1e-5].tolist()
        if len(feats) < 2:
            feats = keep_spearman[:5]
    except Exception as exc:
        warnings.warn(f"Feature-selection fallback: {exc}", RuntimeWarning)
        feats = keep_spearman[:5]
    return sorted(feats)


def get_base_model(seed):
    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)


def generate_roc_data(seed):
    seed_everything(seed)
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)

    psf_probs = np.zeros(n)
    true_labels = np.zeros(n)
    fold_curves = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(np.zeros(n), y), 1):
        _fold_X_cp_full = impute_for_fold(X_cp_full, train_idx)
        _fold_X_qp_full = impute_for_fold(X_qp_full, train_idx)
        (y_train, y_test) = (y[train_idx], y[test_idx])
        true_labels[test_idx] = y_test
        scaler_clin = StandardScaler()
        X_clin_train = scaler_clin.fit_transform(X_clin_full.iloc[train_idx])
        X_clin_test = scaler_clin.transform(X_clin_full.iloc[test_idx])
        feats_cp = select_features_strict(
            _fold_X_cp_full.iloc[train_idx], y_train, cols_cp, seed
        )
        scaler_cp = StandardScaler()
        X_cp_train = scaler_cp.fit_transform(_fold_X_cp_full.iloc[train_idx][feats_cp])
        X_cp_test = scaler_cp.transform(_fold_X_cp_full.iloc[test_idx][feats_cp])
        rf_cp = get_base_model(seed)
        rf_cp.fit(X_cp_train, y_train)
        p_cp_train = rf_cp.predict_proba(X_cp_train)[:, 1]
        p_cp_test = rf_cp.predict_proba(X_cp_test)[:, 1]
        feats_qp = select_features_strict(
            _fold_X_qp_full.iloc[train_idx], y_train, cols_qp, seed
        )
        scaler_qp = StandardScaler()
        X_qp_train = scaler_qp.fit_transform(_fold_X_qp_full.iloc[train_idx][feats_qp])
        X_qp_test = scaler_qp.transform(_fold_X_qp_full.iloc[test_idx][feats_qp])
        rf_qp = get_base_model(seed)
        rf_qp.fit(X_qp_train, y_train)
        p_qp_train = rf_qp.predict_proba(X_qp_train)[:, 1]
        p_qp_test = rf_qp.predict_proba(X_qp_test)[:, 1]
        X_psf_train = np.column_stack((p_cp_train, p_qp_train, X_clin_train))
        X_psf_test = np.column_stack((p_cp_test, p_qp_test, X_clin_test))
        lr_psf = LogisticRegression(random_state=seed, n_jobs=1, max_iter=2000)
        lr_psf.fit(X_psf_train, y_train)
        preds_test = lr_psf.predict_proba(X_psf_test)[:, 1]
        psf_probs[test_idx] = preds_test
        (f_fpr, f_tpr, _) = roc_curve(y_test, preds_test)
        fold_curves.append((f_fpr, f_tpr, auc(f_fpr, f_tpr)))
        print(
            f"Fold {fold_idx} done. (CP feats={len(feats_cp)}, QP feats={len(feats_qp)})"
        )

    return true_labels, psf_probs, fold_curves


print(f"\n运行 ABC型（baseline方案, Seed {SEED}）...")
true_labels, psf_probs, fold_curves = generate_roc_data(SEED)
total_auc = roc_auc_score(true_labels, psf_probs)
print(f"Overall PSF AUC = {total_auc:.4f}")


plt.figure(figsize=(7, 6.5), dpi=300)

for idx, (fpr, tpr, fold_auc) in enumerate(fold_curves):
    plt.plot(
        fpr,
        tpr,
        linestyle="--",
        alpha=0.5,
        label=f"Fold {idx + 1} (AUC = {fold_auc:.3f})",
    )

overall_fpr, overall_tpr, _ = roc_curve(true_labels, psf_probs)
plt.plot(
    overall_fpr,
    overall_tpr,
    color="darkred",
    lw=2.5,
    label=f"Overall baseline PSF (AUC = {total_auc:.4f})",
)

plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle=":")

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel(
    "False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=10
)
plt.ylabel(
    "True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", labelpad=10
)
plt.title(f"MCD-like status", fontsize=13, fontweight="bold", pad=15)
plt.legend(
    loc="lower right",
    frameon=True,
    shadow=False,
    facecolor="white",
    edgecolor="gainsboro",
    fontsize=10,
)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, f"ROC_Curve_target_{TARGET_COL}_seed{SEED}.png")
plt.savefig(save_path, bbox_inches="tight")
plt.close()

print(f"ROC 曲线图保存成功！路径: {save_path}")
