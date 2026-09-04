"""
src/data_processing/preprocess_baseline.py
Preprocessing pipeline for the baseline LightGBM model.

Design & Constraints:
  1. Chronological Split:
     - Positives (KEV=1): Split by KEV Date_Added year (<=2022 -> Train, 2023 -> Val, 2024+ -> Test)
     - Negatives (KEV=0): Split by Published_Date month (Jan -> Train, Feb -> Val, Mar -> Test)
  2. Data Leakage Prevention:
     - Strict exclusion of `kev_epss_interaction` (target leakage).
     - Strict exclusion of `kev` (target column).
     - Strict exclusion of identifiers (`cve_id`, `asset_id`, `asset_name`).
     - Strict exclusion of EPSS snapshot features for the initial baseline per prompt guidance.
  3. Preprocessing Rules:
     - Missing values (e.g. 300 NULL CVSS records) imputed using medians computed ONLY on Training data.
     - Categorical features encoded deterministically using categories learned ONLY from Training data.
     - Identical feature schema enforced across Train, Validation, and Test splits.
  4. Output files:
     - data/processed/baseline_train.csv
     - data/processed/baseline_val.csv
     - data/processed/baseline_test.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_raw_temporal_data():
    """Load master dataset and merge exact NVD Published_Date and KEV Date_Added."""
    master_path = "data/processed/cyberguard_master_enterprise_dataset.csv"
    kev_path = "data/raw/kev_data.csv"
    nvd_path = "data/raw/nvd_data.csv"

    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master dataset not found at {master_path}")

    df_master = pd.read_csv(master_path)
    df_kev = pd.read_csv(kev_path)
    df_nvd = pd.read_csv(nvd_path)

    # Merge date and cvss_version columns safely
    df = df_master.merge(
        df_kev[["CVE_ID", "Date_Added"]],
        left_on="cve_id", right_on="CVE_ID",
        how="left"
    )
    df = df.merge(
        df_nvd[["CVE_ID", "CVSS_Version", "Published_Date"]],
        left_on="cve_id", right_on="CVE_ID",
        how="left",
        suffixes=("", "_nvd")
    )
    df["cvss_version"] = df["CVSS_Version"]

    df["pub_dt"] = pd.to_datetime(df["Published_Date"], format="ISO8601", errors="coerce").dt.tz_localize(None)
    df["kev_dt"] = pd.to_datetime(df["Date_Added"], format="ISO8601", errors="coerce").dt.tz_localize(None)

    # Correct vulnerability_age_days using observation date T = 2022-03-31
    # (Prevents runtime Timestamp.now() leakage across chronological splits)
    observation_date = pd.Timestamp("2022-03-31")
    df["vulnerability_age_days"] = (observation_date - df["pub_dt"]).dt.days.clip(lower=0)

    return df


def assign_chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns split label chronologically:
      - Positives (kev==1): 2021/2022 -> Train, 2023 -> Validation, 2024+ -> Test
      - Negatives (kev==0): Jan 2022 -> Train, Feb 2022 -> Validation, Mar 2022 -> Test
    """
    splits = []
    for idx, row in df.iterrows():
        if row["kev"] == 1:
            yr = row["kev_dt"].year if pd.notna(row["kev_dt"]) else 2022
            if yr <= 2022:
                splits.append("train")
            elif yr == 2023:
                splits.append("validation")
            else:
                splits.append("test")
        else:
            m = row["pub_dt"].month if pd.notna(row["pub_dt"]) else 1
            if m == 1:
                splits.append("train")
            elif m == 2:
                splits.append("validation")
            else:
                splits.append("test")

    df["split"] = splits
    return df


def run_preprocessing():
    print("=" * 65)
    print("  CyberGuard AI -- Baseline Preprocessing Pipeline")
    print("=" * 65)

    df = load_raw_temporal_data()
    df = assign_chronological_split(df)

    # 1. Target Column Verification
    target_col = "kev"
    assert set(df[target_col].unique()).issubset({0, 1}), "Target must be strictly binary {0, 1}"
    print(f"\n[1] Target Column Verified: '{target_col}' (Binary {df[target_col].unique().tolist()})")

    # 2. Categorization of Columns
    identifier_cols = ["cve_id", "asset_id", "asset_name", "CVE_ID_x", "CVE_ID_y", "CVSS_Version", "Date_Added", "Published_Date", "pub_dt", "kev_dt", "status", "split"]
    target_cols = ["kev"]
    leakage_cols = ["kev_epss_interaction"]
    epss_snapshot_cols = ["epss_score", "epss_percentile", "cvss_epss_interaction"]

    # Baseline features selected
    numerical_features = [
        "cvss_score",
        "vulnerability_age_days",
        "business_impact_score",
        "internet_exposure",
        "remediation_hours"
    ]

    categorical_features = [
        "cvss_version",
        "severity",
        "criticality",
        "asset_type",
        "environment",
        "remediation_type"
    ]

    all_excluded = identifier_cols + target_cols + leakage_cols + epss_snapshot_cols
    print(f"\n[2] Feature Classification:")
    print(f"    Numerical Features ({len(numerical_features)}): {numerical_features}")
    print(f"    Categorical Features ({len(categorical_features)}): {categorical_features}")
    print(f"    Excluded / Leakage Columns ({len(all_excluded)}): {all_excluded}")

    # Split into train, val, test
    df_train_raw = df[df["split"] == "train"].copy()
    df_val_raw   = df[df["split"] == "validation"].copy()
    df_test_raw  = df[df["split"] == "test"].copy()

    # 3. Handle Missing Values on Numerical Features (Fit on Train ONLY)
    train_medians = {}
    for num_col in numerical_features:
        med_val = df_train_raw[num_col].median()
        train_medians[num_col] = med_val

    print(f"\n[3] Computed Training Imputation Medians:")
    for col, val in train_medians.items():
        print(f"    • {col}: {val}")

    def impute_numerical(df_sub):
        df_sub = df_sub.copy()
        for num_col, med_val in train_medians.items():
            df_sub[num_col] = df_sub[num_col].fillna(med_val)
        return df_sub

    df_train_imp = impute_numerical(df_train_raw)
    df_val_imp   = impute_numerical(df_val_raw)
    df_test_imp  = impute_numerical(df_test_raw)

    # 4. Handle Categorical Features via One-Hot Encoding (Fit on Train ONLY)
    # Fill categorical NAs with 'Missing'
    for cat_col in categorical_features:
        df_train_imp[cat_col] = df_train_imp[cat_col].fillna("Missing").astype(str)
        df_val_imp[cat_col]   = df_val_imp[cat_col].fillna("Missing").astype(str)
        df_test_imp[cat_col]  = df_test_imp[cat_col].fillna("Missing").astype(str)

    # Learn categories from TRAIN set only
    train_encoded = pd.get_dummies(df_train_imp[categorical_features], drop_first=False)
    feature_columns_cat = train_encoded.columns.tolist()

    def encode_categorical(df_sub):
        encoded = pd.get_dummies(df_sub[categorical_features], drop_first=False)
        # Reindex to match train_encoded exactly, filling missing columns with 0
        encoded = encoded.reindex(columns=feature_columns_cat, fill_value=0)
        return encoded

    train_cat = train_encoded
    val_cat   = encode_categorical(df_val_imp)
    test_cat  = encode_categorical(df_test_imp)

    # Combine Numerical + Categorical Features
    X_train = pd.concat([df_train_imp[numerical_features].reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    X_val   = pd.concat([df_val_imp[numerical_features].reset_index(drop=True), val_cat.reset_index(drop=True)], axis=1)
    X_test  = pd.concat([df_test_imp[numerical_features].reset_index(drop=True), test_cat.reset_index(drop=True)], axis=1)

    y_train = df_train_raw[target_col].values
    y_val   = df_val_raw[target_col].values
    y_test  = df_test_raw[target_col].values

    # 5. Verification Checks
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature count mismatch across splits!"
    assert (X_train.columns == X_val.columns).all(), "Feature column names mismatch!"
    assert (X_val.columns == X_test.columns).all(), "Feature column names mismatch!"
    assert "kev" not in X_train.columns and "kev_epss_interaction" not in X_train.columns, "Leakage column in features!"

    print("\n" + "=" * 65)
    print("  PREPROCESSING SUMMARY REPORT")
    print("=" * 65)

    splits_data = [
        ("Train", df_train_raw, y_train, X_train),
        ("Validation", df_val_raw, y_val, X_val),
        ("Test", df_test_raw, y_test, X_test)
    ]

    for name, df_sub, y_sub, X_sub in splits_data:
        n_total = len(df_sub)
        n_pos = int(y_sub.sum())
        prev = (n_pos / n_total * 100) if n_total > 0 else 0.0
        if name == "Train":
            d_range = f"Published: {df_sub['pub_dt'].min().strftime('%Y-%m-%d')} to {df_sub['pub_dt'].max().strftime('%Y-%m-%d')} | KEV Date: <= 2022-12-31"
        elif name == "Validation":
            d_range = f"Published: {df_sub['pub_dt'].min().strftime('%Y-%m-%d')} to {df_sub['pub_dt'].max().strftime('%Y-%m-%d')} | KEV Date: 2023-01-01 to 2023-12-31"
        else:
            d_range = f"Published: {df_sub['pub_dt'].min().strftime('%Y-%m-%d')} to {df_sub['pub_dt'].max().strftime('%Y-%m-%d')} | KEV Date: 2024-01-01 to 2026-08-26"

        print(f"\n--- {name} Split ---")
        print(f"  Row Count         : {n_total}")
        print(f"  Positive Count    : {n_pos}")
        print(f"  Prevalence        : {prev:.4f}%")
        print(f"  Date Ranges       : {d_range}")
        print(f"  Feature Matrix    : Shape {X_sub.shape}")

    # 6. Save Processed Datasets
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train_out = pd.concat([pd.Series(y_train, name="target"), X_train], axis=1)
    df_val_out   = pd.concat([pd.Series(y_val, name="target"), X_val], axis=1)
    df_test_out  = pd.concat([pd.Series(y_test, name="target"), X_test], axis=1)

    df_train_out.to_csv("data/processed/baseline_train.csv", index=False)
    df_val_out.to_csv("data/processed/baseline_val.csv", index=False)
    df_test_out.to_csv("data/processed/baseline_test.csv", index=False)

    print("\n" + "-" * 65)
    print("  Processed files saved:")
    print("    • data/processed/baseline_train.csv")
    print("    • data/processed/baseline_val.csv")
    print("    • data/processed/baseline_test.csv")
    print("-" * 65)
    print("  [OK] Preprocessing completed & verified successfully.\n")

    return {
        "final_features": X_train.columns.tolist(),
        "excluded_columns": all_excluded,
        "train_rows": len(df_train_raw), "train_pos": int(y_train.sum()), "train_prev": float(y_train.sum() / len(df_train_raw)),
        "val_rows": len(df_val_raw), "val_pos": int(y_val.sum()), "val_prev": float(y_val.sum() / len(df_val_raw)),
        "test_rows": len(df_test_raw), "test_pos": int(y_test.sum()), "test_prev": float(y_test.sum() / len(df_test_raw)),
    }


if __name__ == "__main__":
    run_preprocessing()
