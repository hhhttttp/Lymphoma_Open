"""Check sample alignment across the three modalities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from settings import data_path, output_path
from workflow_utils import validate_feature_pair

import os

import pandas as pd


CP_PATH = data_path("internal/linba_cellprofiler_full.csv")
QP_PATH = data_path("internal/linba_qupath_full.csv")

validate_feature_pair(CP_PATH, QP_PATH)
CLIN_PATH = data_path("internal/linba_linchuang_6.xlsx")


OUT_CSV = output_path("fusion_nested/sample_alignment_report.csv")


def load_image_ids(path):
    df = pd.read_csv(path)
    id_col = df.columns[0]
    ids = df[id_col].astype(str).str.replace(".svs", "", regex=False).str.strip()
    return set(ids)


def load_clin_ids(path):
    df = pd.read_excel(path, engine="openpyxl")
    if "Image" not in df.columns:
        raise ValueError(f"临床表缺少 Image 列，实际列名: {df.columns.tolist()}")
    ids = df["Image"].astype(str).str.replace(".svs", "", regex=False).str.strip()
    return set(ids)


cp_ids = load_image_ids(CP_PATH)
qp_ids = load_image_ids(QP_PATH)
clin_ids = load_clin_ids(CLIN_PATH)

cp_qp = cp_ids & qp_ids
cp_clin = cp_ids & clin_ids
qp_clin = qp_ids & clin_ids
final = cp_ids & qp_ids & clin_ids

all_ids = sorted(cp_ids | qp_ids | clin_ids)
rows = []
for sample in all_ids:
    rows.append(
        {
            "Sample": sample,
            "in_CP": sample in cp_ids,
            "in_QP": sample in qp_ids,
            "in_Clin": sample in clin_ids,
            "in_Final_Inner": sample in final,
        }
    )

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print("========== 三表样本数 ==========")
print(f"CP    : {len(cp_ids)}")
print(f"QP    : {len(qp_ids)}")
print(f"Clin  : {len(clin_ids)}")
print(f"最终对齐: {len(final)}")

print("\n========== 两两交集 ==========")
print(f"CP ∩ QP        : {len(cp_qp)}")
print(f"CP ∩ Clin      : {len(cp_clin)}")
print(f"QP ∩ Clin      : {len(qp_clin)}")
print(f"CP ∩ QP ∩ Clin : {len(final)}")


def show(name, ids):
    print(f"\n{name} ({len(ids)}):")
    for s in sorted(ids):
        print("  " + s)


print("\n========== 未进入最终队列的样本 ==========")
show("CP 有，但 QP 缺", cp_ids - qp_ids)
show("CP 有，但 Clin 缺", cp_ids - clin_ids)
show("QP 有，但 CP 缺", qp_ids - cp_ids)
show("QP 有，但 Clin 缺", qp_ids - clin_ids)
show("Clin 有，但 CP 缺", clin_ids - cp_ids)
show("Clin 有，但 QP 缺", clin_ids - qp_ids)

print(f"\n详细逐样本表已保存: {OUT_CSV}")
