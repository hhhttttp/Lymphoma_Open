"""Pool CellProfiler image measurements and aggregate them by case."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from settings import data_path

import pandas as pd

INPUT_DIR = Path(data_path("raw/cellprofiler"))
OUTPUT_CSV = Path(data_path("external/linba_TCGA_cellprofiler.csv"))
COUNTERS = {
    "ImageNumber",
    "Group_Number",
    "Group_Index",
    "ObjectNumber",
    "Case_ID",
    "FileName_Aggregated",
}


def main():
    frames = []
    for path in sorted(INPUT_DIR.rglob("MyExpt_Image.csv")):
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        urls = [c for c in frame if "URL_HE" in c]
        if len(urls) != 1:
            raise ValueError(f"{path}: expected exactly one URL_HE column.")
        paths = frame[urls[0]].astype("string").str.replace("\\", "/", regex=False)
        case_ids = paths.str.rstrip("/").str.rsplit("/", n=2).str[-2]
        if case_ids.isna().any() or case_ids.eq("").any():
            raise ValueError(f"{path}: cannot extract a case directory from URL_HE.")
        clean = pd.DataFrame({"Case_ID": case_ids})
        for column in frame:
            if (
                column in COUNTERS
                or column == urls[0]
                or column.startswith(("Metadata_", "FileName_", "PathName_", "URL_"))
            ):
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & numeric.isna()
            if not invalid.any() and numeric.notna().any():
                clean[column] = numeric
        if clean.shape[1] == 1:
            raise ValueError(f"{path}: no numeric measurement columns.")
        frames.append(clean)
    if not frames:
        raise FileNotFoundError(f"No MyExpt_Image.csv exports in {INPUT_DIR}.")
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    grouped = pooled.groupby("Case_ID", sort=True).agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{feature}_{stat}" for feature, stat in grouped.columns]
    grouped.index.name = "FileName_Aggregated"
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grouped.reset_index().to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(grouped)} case feature rows: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
