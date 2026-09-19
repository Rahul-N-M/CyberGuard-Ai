"""
src/ml/inspect_pasted_data.py
Detailed inspection script for data/train.csv, data/validation.csv, data/test.csv.
"""

import os
import pandas as pd
import numpy as np

def inspect_datasets():
    print("=" * 65)
    print("  INSPECTING NEWLY PASTED DATASETS")
    print("=" * 65)

    paths = {
        "Train": "data/train.csv",
        "Validation": "data/validation.csv",
        "Test": "data/test.csv"
    }

    df_sample = pd.read_csv("data/train.csv", nrows=5)
    print(f"\n[Columns Count]: {len(df_sample.columns)}")
    print("[Column Names]:")
    for i, col in enumerate(df_sample.columns, 1):
        print(f"  {i:>2}. {col}")

    total_rows = 0
    total_positives = 0
    all_cves = set()
    pos_cves_all = set()

    split_stats = {}

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"ERROR: {path} not found!")
            continue

        print(f"\n--- Loading {name} ({path}) ---")
        df = pd.read_csv(path)
        shape = df.shape
        total_rows += shape[0]

        target_candidates = [c for c in df.columns if "target" in c.lower() or "kev" in c.lower() or c == "label"]
        t_col = target_candidates[0] if target_candidates else df.columns[-1]

        positives = int(df[t_col].sum())
        total_positives += positives
        prevalence = (positives / shape[0]) if shape[0] > 0 else 0.0

        cve_col = "CVE_ID" if "CVE_ID" in df.columns else ("cve_id" if "cve_id" in df.columns else None)
        n_unique_cves = df[cve_col].nunique() if cve_col else 0

        pos_cve_set = set(df[df[t_col] == 1][cve_col].unique()) if cve_col else set()
        all_cves.update(df[cve_col].unique()) if cve_col else None
        pos_cves_all.update(pos_cve_set)

        date_col = "observation_date" if "observation_date" in df.columns else ("Published_Date" if "Published_Date" in df.columns else None)
        min_date = df[date_col].min() if date_col else "N/A"
        max_date = df[date_col].max() if date_col else "N/A"

        print(f"  Shape              : {shape}")
        print(f"  Target Column      : '{t_col}'")
        print(f"  Positives Count    : {positives:,d} / {shape[0]:,d}")
        print(f"  Prevalence         : {prevalence:.6%}")
        print(f"  Unique CVEs        : {n_unique_cves:,d}")
        print(f"  Unique Pos CVEs    : {len(pos_cve_set):,d}")
        print(f"  Date Range ({date_col}): {min_date} to {max_date}")

        # Missing values check
        null_counts = df.isna().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            print(f"  Missing Values     : {null_cols.to_dict()}")
        else:
            print("  Missing Values     : None (0 nulls)")

        split_stats[name] = {
            "df": df,
            "shape": shape,
            "t_col": t_col,
            "positives": positives,
            "prevalence": prevalence,
            "n_cves": n_unique_cves,
            "n_pos_cves": len(pos_cve_set)
        }

    print("\n" + "=" * 65)
    print("  OVERALL TEMPORAL DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total Rows         : {total_rows:,d}")
    print(f"  Total Positives    : {total_positives:,d}")
    print(f"  Total Unique CVEs  : {len(all_cves):,d}")
    print(f"  Total Pos CVEs     : {len(pos_cves_all):,d}")

    # Calculate scale_pos_weight for Training
    tr_pos = split_stats["Train"]["positives"]
    tr_tot = split_stats["Train"]["shape"][0]
    tr_neg = tr_tot - tr_pos
    scale_pos_weight = tr_neg / tr_pos if tr_pos > 0 else 1.0
    print(f"  Training Positives : {tr_pos:,d}, Negatives: {tr_neg:,d}")
    print(f"  Training Imbalance : 1 : {tr_neg/tr_pos:.2f}")
    print(f"  scale_pos_weight   : {scale_pos_weight:.6f}")
    print("=" * 65)

if __name__ == "__main__":
    inspect_datasets()
