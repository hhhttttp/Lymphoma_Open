"""Compare classifiers for individual modalities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from settings import data_path, output_path
from workflow_utils import measurement_columns
from workflow_utils import (
    binary_labels,
    validate_feature_pair,
    validate_ids,
    impute_for_fold,
)

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
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE


from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


OUTPUT_DIR = output_path("classifier_comparison")
SEED = 21
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
    final_cols = [col_mapping[c] for c in existing_cols if col_mapping[c] != "Merge_ID"]

    df["Merge_ID"] = (
        df["Merge_ID"].astype(str).str.replace(".svs", "", regex=False).str.strip()
    )

    for c in final_cols:
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
    return df[["Merge_ID"] + final_cols], final_cols


df_clin, cols_clin = load_clin(CLIN_PATH)


df_m1 = pd.merge(df_cp, df_qp[["Merge_ID"] + cols_qp], on="Merge_ID", how="inner")
df_final = pd.merge(df_m1, df_clin, on="Merge_ID", how="inner")

X_cp_full = df_final[cols_cp].copy()
X_qp_full = df_final[cols_qp].copy()
X_clin_full = df_final[cols_clin].fillna(0)

raw_y = df_final[df_cp.columns[1] if "label" not in df_cp else "label"]
y = binary_labels(raw_y)

print(f"最终对齐样本数: {len(y)}")
print(f"特征概况: CP={len(cols_cp)}, QP={len(cols_qp)}, Clin={len(cols_clin)}")


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


def get_models(seed=21):
    return {
        "LR": LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
        "GNB": GaussianNB(),
        "BNB": BernoulliNB(),
        "SVM": SVC(probability=True, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "XGB": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = get_models(SEED)

single_oof = {}
for modal in ["Clin", "CP", "QP"]:
    for m in models.keys():
        single_oof[(modal, m)] = np.zeros(len(y))

true_labels = np.zeros(len(y))

print(f"\n开始单模态模型对比 (Seed {SEED}, 5-fold OOF)...")

for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), 1):
    _fold_X_cp_full = impute_for_fold(X_cp_full, train_idx)
    _fold_X_qp_full = impute_for_fold(X_qp_full, train_idx)
    (y_train, y_test) = (y[train_idx], y[test_idx])
    true_labels[test_idx] = y_test
    scaler_clin = StandardScaler()
    X_clin_train = scaler_clin.fit_transform(X_clin_full.iloc[train_idx])
    X_clin_test = scaler_clin.transform(X_clin_full.iloc[test_idx])
    feats_cp = select_features_strict(
        _fold_X_cp_full.iloc[train_idx], y_train, cols_cp, SEED
    )
    scaler_cp = StandardScaler()
    X_cp_train = scaler_cp.fit_transform(_fold_X_cp_full.iloc[train_idx][feats_cp])
    X_cp_test = scaler_cp.transform(_fold_X_cp_full.iloc[test_idx][feats_cp])
    feats_qp = select_features_strict(
        _fold_X_qp_full.iloc[train_idx], y_train, cols_qp, SEED
    )
    scaler_qp = StandardScaler()
    X_qp_train = scaler_qp.fit_transform(_fold_X_qp_full.iloc[train_idx][feats_qp])
    X_qp_test = scaler_qp.transform(_fold_X_qp_full.iloc[test_idx][feats_qp])
    for (mname, model) in models.items():
        clf = model
        clf.fit(X_clin_train, y_train)
        single_oof["Clin", mname][test_idx] = clf.predict_proba(X_clin_test)[:, 1]
        clf = model
        clf.fit(X_cp_train, y_train)
        single_oof["CP", mname][test_idx] = clf.predict_proba(X_cp_test)[:, 1]
        clf = model
        clf.fit(X_qp_train, y_train)
        single_oof["QP", mname][test_idx] = clf.predict_proba(X_qp_test)[:, 1]
    print(f"Fold {fold} done. (CP feats={len(feats_cp)}, QP feats={len(feats_qp)})")


rows = []
for (modal, mname), p in single_oof.items():
    auc = roc_auc_score(true_labels, p)
    rows.append({"Modality": modal, "Model": mname, "AUC": auc})

df_single = pd.DataFrame(rows).sort_values(["Modality", "AUC"], ascending=[True, False])

print("\n" + "=" * 70)
print("Single-modality Model Benchmark (5-fold OOF AUC)")
print("=" * 70)
print(df_single.to_string(index=False, formatters={"AUC": lambda x: f"{x:.4f}"}))
print("=" * 70)

best_each = df_single.sort_values("AUC", ascending=False).groupby("Modality").head(1)
print("\nBest model per modality:")
print(best_each.to_string(index=False, formatters={"AUC": lambda x: f"{x:.4f}"}))


out_csv = os.path.join(OUTPUT_DIR, f"single_modality_benchmark_seed_{SEED}.csv")
df_single.to_csv(out_csv, index=False)
print(f"\n结果表已保存: {out_csv}")
