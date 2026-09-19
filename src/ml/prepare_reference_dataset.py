"""
src/ml/prepare_reference_dataset.py
===================================
Prepares and partitions Fang et al.'s (2020) reference NVD dataset
(2013–2018, 60,783 CVE records) into clean processed splits.

Inputs:
  data/nvd_data_2013_2018_with_time_all_exp.csv

Outputs:
  data/processed_reference/ref_dataset_cleaned.csv
  data/processed_reference/ref_train.csv  (2013-2016)
  data/processed_reference/ref_val.csv    (2017)
  data/processed_reference/ref_test.csv   (2018+)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = PROJECT_ROOT / "data" / "nvd_data_2013_2018_with_time_all_exp.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_reference"


def prepare_reference_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  PREPARING REFERENCE DATASET (Fang et al. 2013–2018 NVD)")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    print(f"Loading {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Raw shape: {df.shape}")

    # Parse timestamps
    df["publishedDate"] = pd.to_datetime(df["publishedDate"], utc=True, errors="coerce")
    df["lastModifiedDate"] = pd.to_datetime(df["lastModifiedDate"], utc=True, errors="coerce")
    df["pub_year"] = df["publishedDate"].dt.year

    # Verify target variable E (Exploited)
    df["E"] = df["E"].astype(int)
    pos_count = int(df["E"].sum())
    pos_rate = df["E"].mean() * 100
    print(f"  Total CVEs: {len(df):,d} | Exploited (E=1): {pos_count:,d} ({pos_rate:.2f}%)")

    # Clean description
    df["DESC"] = df["DESC"].fillna("").astype(str)

    # Save full cleaned dataset
    cleaned_path = OUTPUT_DIR / "ref_dataset_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"  Saved cleaned dataset: {cleaned_path}")

    # Chronological partition:
    # Train: 2013–2016 (25,176 CVEs)
    # Val:   2017      (17,058 CVEs)
    # Test:  2018+     (18,549 CVEs)
    train_df = df[df["pub_year"] <= 2016].copy()
    val_df   = df[df["pub_year"] == 2017].copy()
    test_df  = df[df["pub_year"] >= 2018].copy()

    print("\nChronological Partition:")
    for name, split in [("Train (<=2016)", train_df), ("Val (2017)", val_df), ("Test (2018+)", test_df)]:
        pos = int(split["E"].sum())
        rate = split["E"].mean() * 100
        print(f"  {name:15s}: {len(split):,d} CVEs | {pos:,d} positives ({rate:.2f}%)")

    train_path = OUTPUT_DIR / "ref_train.csv"
    val_path   = OUTPUT_DIR / "ref_val.csv"
    test_path  = OUTPUT_DIR / "ref_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[Done] Reference data prepared in: {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_reference_data()
