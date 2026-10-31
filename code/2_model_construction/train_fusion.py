"""Evaluate baseline multimodal fusion using five-fold predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from settings import data_path, output_path
from model_artifacts import save_baseline
from workflow_utils import measurement_columns
from workflow_utils import (
    binary_labels,
    validate_feature_pair,
    validate_ids,
    impute_for_fold,
    numeric_frame,
)

# Baseline PSF uses in-sample base probabilities within each outer training fold.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import warnings


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from imblearn.over_sampling import SMOTE


OUTPUT_DIR = output_path("fusion_baseline")
SEED = 27
CP_PATH = data_path("internal/linba_cellprofiler_full.csv")
QP_PATH = data_path("internal/linba_qupath_full.csv")

validate_feature_pair(CP_PATH, QP_PATH)
CLIN_PATH = data_path("internal/linba_linchuang_6.xlsx")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


print(f"正在读取数据 (Seed={SEED})...")


def load_img(path, prefix):
    if not os.path.exists(path):
        return None, None, []
    df = pd.read_csv(path)
    id_col = df.columns[0]
    lbl_col = "label" if "label" in df.columns else df.columns[1]

    df["Merge_ID"] = (
        df[id_col].astype(str).str.replace(".svs", "", regex=False).str.strip()
    )
    feat_cols = measurement_columns(
        sorted([c for c in df.columns if c not in [id_col, lbl_col, "Merge_ID"]])
    )
    rename_map = {c: f"{prefix}_{c}" for c in feat_cols}
    df = df.rename(columns=rename_map)
    final_cols = [f"{prefix}_{c}" for c in feat_cols]
    return df, df[["Merge_ID", lbl_col] + final_cols], final_cols


_, df_cp, cols_cp = load_img(CP_PATH, "CP")
_, df_qp, cols_qp = load_img(QP_PATH, "QP")


def load_clin(path):
    if not os.path.exists(path):
        return None, []

    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        try:
            df = pd.read_csv(path, encoding="gbk")
        except (UnicodeDecodeError, pd.errors.ParserError):
            df = pd.read_csv(path, encoding="utf-8")

    col_mapping = {
        "Image": "Merge_ID",
        "Gender": "Clin_Gender",
        "AgeGroup（≥60）": "Clin_AgeGroup",
        "Extranodalinvolvement": "Clin_Extranodal",
        "Stage Group": "Clin_Stage",
    }

    existing_cols = [c for c in col_mapping.keys() if c in df.columns]
    if "Image" not in df.columns:
        print("Error: 临床表缺少 Image 列，无法对齐")
        print("实际列名:", df.columns.tolist())
        return None, []

    df = df.rename(columns=col_mapping)
    clin_cols = [col_mapping[c] for c in existing_cols if col_mapping[c] != "Merge_ID"]

    df["Merge_ID"] = (
        df["Merge_ID"].astype(str).str.replace(".svs", "", regex=False).str.strip()
    )
    for c in clin_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    validate_ids(
        df,
        "Merge_ID"
        if "Merge_ID" in df
        else "Patient_ID"
        if "Patient_ID" in df
        else "Image",
        path,
    )
    return df[["Merge_ID"] + clin_cols], clin_cols


df_clin, cols_clin = load_clin(CLIN_PATH)


df_m1 = pd.merge(df_cp, df_qp[["Merge_ID"] + cols_qp], on="Merge_ID", how="inner")
df_final = pd.merge(df_m1, df_clin, on="Merge_ID", how="inner")

X_cp_full = df_final[cols_cp].copy()
X_qp_full = df_final[cols_qp].copy()
X_clin_full = df_final[cols_clin].fillna(0)

raw_y = df_final[df_cp.columns[1] if "label" not in df_cp else "label"]
y = binary_labels(raw_y)

print(f"最终对齐样本数: {len(y)}")


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

    X_final = X_uni[keep_spearman]

    try:
        lasso = LassoCV(
            cv=5, random_state=seed, max_iter=20000, n_jobs=1, selection="cyclic"
        )
        lasso.fit(X_final, y_res)
        feats = X_final.columns[np.abs(lasso.coef_) > 1e-5].tolist()
        if len(feats) < 2:
            feats = keep_spearman[:5]
    except Exception as exc:
        warnings.warn(f"Feature-selection fallback: {exc}", RuntimeWarning)
        feats = keep_spearman[:5]

    return sorted(feats)


def get_base_model(seed):
    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

probs = {
    "Clin": np.zeros(len(y)),
    "CP": np.zeros(len(y)),
    "QP": np.zeros(len(y)),
    "MEFF": np.zeros(len(y)),
    "PSF": np.zeros(len(y)),
    "PLAF": np.zeros(len(y)),
}

true_labels = np.zeros(len(y))
saved_folds = []
fold_ids = np.zeros(len(y), dtype=int)

desc_map = {
    "Clin": "临床单模态 (Clinical) -> RF",
    "CP": "CellProfiler单模态 -> RF",
    "QP": "QuPath单模态 -> RF",
    "MEFF": "多模态特征早期融合 (Multimodal Early Feature Fusion, MEFF) -> RF",
    "PSF": "概率级堆叠融合 (Probability-level Stacking Fusion, PSF) -> LR",
    "PLAF": "概率级晚期平均融合 (Probability-level Late Average Fusion, PLAF)",
}

print(f"\n开始对比实验 (Seed {SEED})...")

for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), 1):
    _fold_X_cp_full = impute_for_fold(X_cp_full, train_idx)
    _fold_X_qp_full = impute_for_fold(X_qp_full, train_idx)
    (y_train, y_test) = (y[train_idx], y[test_idx])
    true_labels[test_idx] = y_test
    scaler_clin = StandardScaler()
    X_clin_train = scaler_clin.fit_transform(X_clin_full.iloc[train_idx])
    X_clin_test = scaler_clin.transform(X_clin_full.iloc[test_idx])
    rf_clin = get_base_model(SEED)
    rf_clin.fit(X_clin_train, y_train)
    p_clin_train = rf_clin.predict_proba(X_clin_train)[:, 1]
    p_clin_test = rf_clin.predict_proba(X_clin_test)[:, 1]
    probs["Clin"][test_idx] = p_clin_test
    feats_cp = select_features_strict(
        _fold_X_cp_full.iloc[train_idx], y_train, cols_cp, SEED
    )
    scaler_cp = StandardScaler()
    X_cp_train = scaler_cp.fit_transform(_fold_X_cp_full.iloc[train_idx][feats_cp])
    X_cp_test = scaler_cp.transform(_fold_X_cp_full.iloc[test_idx][feats_cp])
    rf_cp = get_base_model(SEED)
    rf_cp.fit(X_cp_train, y_train)
    p_cp_train = rf_cp.predict_proba(X_cp_train)[:, 1]
    p_cp_test = rf_cp.predict_proba(X_cp_test)[:, 1]
    probs["CP"][test_idx] = p_cp_test
    feats_qp = select_features_strict(
        _fold_X_qp_full.iloc[train_idx], y_train, cols_qp, SEED
    )
    scaler_qp = StandardScaler()
    X_qp_train = scaler_qp.fit_transform(_fold_X_qp_full.iloc[train_idx][feats_qp])
    X_qp_test = scaler_qp.transform(_fold_X_qp_full.iloc[test_idx][feats_qp])
    rf_qp = get_base_model(SEED)
    rf_qp.fit(X_qp_train, y_train)
    p_qp_train = rf_qp.predict_proba(X_qp_train)[:, 1]
    p_qp_test = rf_qp.predict_proba(X_qp_test)[:, 1]
    probs["QP"][test_idx] = p_qp_test
    X_meff_train = np.hstack((X_cp_train, X_qp_train, X_clin_train))
    X_meff_test = np.hstack((X_cp_test, X_qp_test, X_clin_test))
    rf_meff = get_base_model(SEED)
    rf_meff.fit(X_meff_train, y_train)
    probs["MEFF"][test_idx] = rf_meff.predict_proba(X_meff_test)[:, 1]
    X_psf_train = np.column_stack((p_cp_train, p_qp_train, X_clin_train))
    X_psf_test = np.column_stack((p_cp_test, p_qp_test, X_clin_test))
    lr_psf = LogisticRegression(random_state=SEED, n_jobs=1, max_iter=2000)
    lr_psf.fit(X_psf_train, y_train)
    probs["PSF"][test_idx] = lr_psf.predict_proba(X_psf_test)[:, 1]
    probs["PLAF"][test_idx] = (p_cp_test + p_qp_test) / 2.0
    fold_ids[test_idx] = fold
    saved_folds.append({
        "fold": fold,
        "train_idx": train_idx.copy(),
        "test_idx": test_idx.copy(),
        "features": {"CP": list(feats_cp), "QP": list(feats_qp), "Clin": list(cols_clin)},
        "means": {
            "CP": numeric_frame(X_cp_full.iloc[train_idx][feats_cp]).mean().fillna(0.0),
            "QP": numeric_frame(X_qp_full.iloc[train_idx][feats_qp]).mean().fillna(0.0),
            # Clinical missing values were filled with zero by the baseline loader.
            "Clin": pd.Series(0.0, index=cols_clin),
        },
        "scalers": {"CP": scaler_cp, "QP": scaler_qp, "Clin": scaler_clin},
        "models": {"CP": rf_cp, "QP": rf_qp, "Clin": rf_clin,
                   "MEFF": rf_meff, "PSF": lr_psf},
    })
    print(f"Fold {fold} done. (CP feats={len(feats_cp)}, QP feats={len(feats_qp)})")


oof_path = os.path.join(OUTPUT_DIR, f"OOF_probs_ROC_seed_{SEED}.npz")
artifact, model_path = save_baseline(
    SEED, df_final, {"CP": cols_cp, "QP": cols_qp, "Clin": cols_clin},
    y, probs, saved_folds,
)
np.savez(oof_path, y_true=true_labels, sample_ids=df_final["Merge_ID"].to_numpy(dtype=str),
         fold_ids=fold_ids, run_id=artifact["run_id"],
         protocol_version="baseline_in_sample_stacking_v3", **probs)
print(f"OOF 预测已保存: {oof_path}")
print(f"每折模型和预处理参数已保存: {model_path}")


print("\n" + "=" * 100)
print(f"{'Model':<10} | {'Description':<75} | {'AUC':<10}")
print("-" * 100)

results = []
for name, p in probs.items():
    auc = roc_auc_score(true_labels, p)
    results.append((name, desc_map.get(name, name), auc))

results.sort(key=lambda x: x[2], reverse=True)

for name, desc, auc in results:
    print(f"{name:<10} | {desc:<75} | {auc:.4f}")

print("=" * 100)


plt.figure(figsize=(10, 8))

for name, desc, auc in results:
    fpr, tpr, _ = roc_curve(true_labels, probs[name])

    if name in ["MEFF", "PSF", "PLAF"]:
        plt.plot(fpr, tpr, lw=2.8, label=f"{name} (AUC={auc:.3f})")
    else:
        plt.plot(fpr, tpr, lw=1.6, linestyle="--", label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title("Experiment: Fusion (MEFF/PSF/PLAF) vs. Single Modalities")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

out_png = os.path.join(
    OUTPUT_DIR, f"Clinical_Experiment_ROC_MEFF_PSF_PLAF_seed_{SEED}.png"
)
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nROC 图已保存: {out_png}")
