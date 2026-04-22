import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_dataframe(df: pd.DataFrame, name: str):
    print_header(f"{name}: shape and columns")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    for col in df.columns:
        print(f"- {col}")

    print_header(f"{name}: dtypes")
    print(df.dtypes.to_string())

    print_header(f"{name}: first 5 rows")
    print(df.head(5).to_string(index=False))

    print_header(f"{name}: missing values per column")
    print(df.isna().sum().to_string())


def summarize_metadata(df_meta: pd.DataFrame):
    print_header("Metadata summary")

    expected_cols = [
        "object",
        "distance",
        "angle",
        "mic",
        "trial",
        "band",
        "ROI start",
        "ROI end",
        "ROI length",
    ]

    missing_expected = [c for c in expected_cols if c not in df_meta.columns]
    if missing_expected:
        print("Missing expected metadata columns:")
        for c in missing_expected:
            print(f"- {c}")
    else:
        print("All expected metadata columns are present.")

    for col in ["object", "distance", "angle", "mic", "band"]:
        if col in df_meta.columns:
            vals = sorted(df_meta[col].dropna().unique().tolist())
            print_header(f"Unique values for {col}")
            print(vals)

    if "ROI length" in df_meta.columns:
        print_header("ROI length stats")
        print(df_meta["ROI length"].describe().to_string())

    if {"ROI start", "ROI end"}.issubset(df_meta.columns):
        print_header("ROI index consistency")
        span = df_meta["ROI end"] - df_meta["ROI start"]
        print("Computed span (ROI end - ROI start) stats:")
        print(span.describe().to_string())


def summarize_roi_arrays(df_roi: pd.DataFrame):
    if "roi" not in df_roi.columns:
        print_header("ROI array summary")
        print("No 'roi' column found in ROI dataset file.")
        return

    print_header("ROI array summary")
    lengths = df_roi["roi"].apply(lambda x: len(x) if isinstance(x, np.ndarray) else np.nan)
    print("Array length stats:")
    print(lengths.describe().to_string())

    valid = df_roi["roi"].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
    print(f"Valid numpy ROI arrays: {int(valid.sum())} / {len(df_roi)}")

    if valid.any():
        first_idx = int(valid[valid].index[0])
        first_roi = df_roi.loc[first_idx, "roi"]
        print_header("Example ROI sample values (first valid row)")
        print(f"Row index: {first_idx}")
        print(f"First 12 values: {np.asarray(first_roi)[:12]}")


def main():
    ap = argparse.ArgumentParser(description="Inspect ROI dataset and metadata outputs.")
    ap.add_argument("--roi_pkl", default="outputs_roi/roi_dataset.pkl", help="Path to ROI dataset .pkl file.")
    ap.add_argument(
        "--meta_csv",
        default="outputs_roi/roi_dataset_metadata.csv",
        help="Path to ROI metadata .csv file.",
    )
    args = ap.parse_args()

    roi_path = Path(args.roi_pkl)
    meta_path = Path(args.meta_csv)

    if not roi_path.exists():
        raise SystemExit(f"ROI dataset file not found: {roi_path}")
    if not meta_path.exists():
        raise SystemExit(f"Metadata CSV file not found: {meta_path}")

    print_header("Loading files")
    print(f"ROI dataset path: {roi_path}")
    print(f"Metadata CSV path: {meta_path}")

    df_roi = pd.read_pickle(roi_path)
    df_meta = pd.read_csv(meta_path)

    summarize_dataframe(df_roi, "ROI PKL")
    summarize_roi_arrays(df_roi)

    summarize_dataframe(df_meta, "Metadata CSV")
    summarize_metadata(df_meta)

    print_header("Inspection complete")


if __name__ == "__main__":
    main()
