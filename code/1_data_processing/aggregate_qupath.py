"""Aggregate numeric QuPath object measurements across TSV exports by image."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from settings import data_path

import pandas as pd

INPUT_DIR = Path(data_path("raw/qupath"))
OUTPUT_CSV = Path(data_path("external/linba_TCGA_qupath.csv"))
METADATA = {"Image", "Object ID", "Name", "Class", "Parent", "ROI", "Object type"}


def main():
    frames = []
    for path in sorted(INPUT_DIR.glob("*.tsv")):
        frame = pd.read_csv(path, sep="\t")
        if "Image" not in frame or frame["Image"].isna().any():
            raise ValueError(f"{path}: missing Image identifiers.")
        features = [
            c
            for c in frame.select_dtypes(include="number").columns
            if c not in METADATA
        ]
        if not features:
            raise ValueError(f"{path}: no numeric measurement columns.")
        frames.append(frame[["Image"] + features])
    if not frames:
        raise FileNotFoundError(f"No TSV exports in {INPUT_DIR}.")
    # Pool raw observations before aggregation when an image spans multiple exports.
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    grouped = pooled.groupby("Image", sort=True).agg(
        ["mean", "std", "max", "min", "median"]
    )
    grouped.columns = [f"{feature}_{stat}" for feature, stat in grouped.columns]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grouped.reset_index().to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(grouped)} image feature rows: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
