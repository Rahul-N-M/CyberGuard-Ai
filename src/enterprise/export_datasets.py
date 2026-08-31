"""
src/enterprise/export_datasets.py
Exports enriched enterprise datasets to CSV files for Vinod (ML) and Navya (Dashboard).

Exports:
  - data/processed/enterprise_assets.csv
  - data/processed/asset_vulnerabilities.csv
  - data/processed/cyberguard_master_enterprise_dataset.csv  ← PRIMARY ML INPUT

Run:
    python -m src.enterprise.export_datasets
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.database.config import get_assets_output, get_asset_vulns_output, get_master_dataset_output
from src.enterprise.data_access import (
    get_all_assets,
    get_asset_vulnerability_pairs,
    get_ml_training_matrix,
    get_enterprise_risk_summary,
)


def export_enterprise_assets():
    """Export the full enterprise asset inventory."""
    output_path = get_assets_output()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = get_all_assets()
    df.to_csv(output_path, index=False)
    print(f"  [OK] enterprise_assets.csv         -> {len(df)} assets  ({output_path})")
    return df


def export_asset_vulnerabilities():
    """Export all asset-vulnerability mapping pairs."""
    output_path = get_asset_vulns_output()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = get_asset_vulnerability_pairs()
    df.to_csv(output_path, index=False)
    print(f"  [OK] asset_vulnerabilities.csv      -> {len(df)} mappings  ({output_path})")
    return df


def export_master_dataset():
    """
    Export the unified ML-ready master dataset.
    This is the PRIMARY INPUT for Vinod's LightGBM model.
    Contains all vulnerability features + enterprise asset context + remediation hours.
    """
    output_path = get_master_dataset_output()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = get_ml_training_matrix()
    df.to_csv(output_path, index=False)
    print(f"  [OK] cyberguard_master_enterprise_dataset.csv -> {len(df)} rows  ({output_path})")
    return df


def export_risk_summary():
    """Export per-asset risk summary for the dashboard."""
    output_path = "data/processed/enterprise_risk_summary.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = get_enterprise_risk_summary()
    df.to_csv(output_path, index=False)
    print(f"  [OK] enterprise_risk_summary.csv    -> {len(df)} assets  ({output_path})")
    return df


def run_exports():
    print("\n" + "=" * 55)
    print("  CyberGuard AI -- Dataset Export Pipeline")
    print("=" * 55)

    assets_df   = export_enterprise_assets()
    av_df       = export_asset_vulnerabilities()
    master_df   = export_master_dataset()
    summary_df  = export_risk_summary()

    print("\n" + "-" * 55)
    print("  Export Summary:")
    print(f"  Assets                   : {len(assets_df)}")
    print(f"  Asset-CVE Mappings       : {len(av_df)}")
    print(f"  ML Master Dataset rows   : {len(master_df)}")
    print(f"  ML Master Dataset cols   : {len(master_df.columns)}")
    print(f"  Column names:")
    for col in master_df.columns:
        print(f"    • {col}")
    print("-" * 55)
    print("  [CyberGuard] All datasets exported successfully.\n")


if __name__ == "__main__":
    run_exports()
