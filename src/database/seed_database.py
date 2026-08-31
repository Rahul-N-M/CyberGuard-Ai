"""
src/database/seed_database.py
Ingestion pipeline: reads Rahul's risk_features.csv and seeds the database.

Pipeline:
  1. Init database schema
  2. Seed departments
  3. Seed assets (from assets_config.py)
  4. Ingest vulnerabilities from risk_features.csv  ← NULL values PRESERVED
  5. Generate asset_vulnerability mappings (from synthetic_generator.py)
  6. Print ingestion summary report

Run:
    python -m src.database.seed_database
"""

import sys
import os
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
from sqlalchemy.exc import IntegrityError

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.database.connection import init_db, get_session_factory
from src.database.models import Department, Asset, Vulnerability, AssetVulnerability
from src.database.config import get_risk_features_path
from src.enterprise.assets_config import DEPARTMENTS, ASSETS
from src.enterprise.synthetic_generator import generate_asset_vulnerability_mappings


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _nan_to_none(value):
    """Convert NaN / pandas NA to Python None so SQLAlchemy stores NULL."""
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _parse_date(date_str):
    """Parse a date string to datetime, returning None on failure."""
    if pd.isna(date_str) or date_str is None:
        return None
    try:
        return pd.to_datetime(date_str, utc=True).to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# SEED FUNCTIONS
# ─────────────────────────────────────────────────────────────
def seed_departments(session):
    """Insert department records."""
    print("\n[1/5] Seeding departments...")
    count = 0
    for dept_data in DEPARTMENTS:
        existing = session.query(Department).filter_by(
            department_name=dept_data["department_name"]
        ).first()
        if not existing:
            dept = Department(**dept_data)
            session.add(dept)
            count += 1
    session.commit()
    print(f"      -> {count} departments inserted.")


def seed_assets(session):
    """Insert enterprise asset records."""
    print("\n[2/5] Seeding assets...")
    # Build department name -> id lookup
    dept_map = {d.department_name: d.department_id for d in session.query(Department).all()}
    count = 0
    for asset_data in ASSETS:
        existing = session.query(Asset).filter_by(asset_id=asset_data["asset_id"]).first()
        if not existing:
            data = dict(asset_data)
            dept_name = data.pop("department_name", None)
            data["department_id"] = dept_map.get(dept_name)
            asset = Asset(**data)
            session.add(asset)
            count += 1
    session.commit()
    print(f"      -> {count} assets inserted.")


def seed_vulnerabilities(session, csv_path: str):
    """
    Ingest vulnerability records from risk_features.csv.
    NULL values (300 incomplete records) are preserved as NULL -- NOT replaced with 0.
    """
    print(f"\n[3/5] Ingesting vulnerabilities from {csv_path}...")
    df = pd.read_csv(csv_path)
    total = len(df)
    inserted = 0
    skipped  = 0

    for _, row in df.iterrows():
        cve_id = str(row.get("CVE_ID", "")).strip()
        if not cve_id:
            skipped += 1
            continue

        existing = session.query(Vulnerability).filter_by(cve_id=cve_id).first()
        if existing:
            skipped += 1
            continue

        vuln = Vulnerability(
            cve_id                = cve_id,
            cvss_score            = _nan_to_none(row.get("CVSS_Score")),
            cvss_version          = _nan_to_none(row.get("CVSS_Version")),
            severity              = _nan_to_none(row.get("Severity")),
            description           = _nan_to_none(row.get("Description")),
            published_date        = _parse_date(row.get("Published_Date")),
            epss_score            = _nan_to_none(row.get("EPSS_Score")),
            epss_percentile       = _nan_to_none(row.get("EPSS_Percentile")),
            kev                   = int(row.get("KEV", 0) or 0),
            vulnerability_age_days= _nan_to_none(row.get("Vulnerability_Age_Days")),
            severity_encoded      = _nan_to_none(row.get("Severity_Encoded")),
            cvss_epss_interaction = _nan_to_none(row.get("CVSS_EPSS_Interaction")),
            kev_epss_interaction  = _nan_to_none(row.get("KEV_EPSS_Interaction")),
        )
        session.add(vuln)
        inserted += 1

        # Batch commit every 500 rows for performance
        if inserted % 500 == 0:
            session.commit()
            print(f"      -> {inserted}/{total} inserted...", end="\r")

    session.commit()

    # Count NULL records for verification
    null_count = session.query(Vulnerability).filter(
        Vulnerability.cvss_score == None  # noqa
    ).count()

    print(f"      -> {inserted} vulnerabilities inserted, {skipped} skipped.")
    print(f"      -> Records with NULL CVSS (preserved correctly): {null_count}")


def seed_asset_vulnerabilities(session):
    """Generate and insert asset-vulnerability mappings with remediation hours."""
    print("\n[4/5] Generating asset-vulnerability mappings...")
    mappings = generate_asset_vulnerability_mappings(session)
    count = 0
    for mapping in mappings:
        try:
            av = AssetVulnerability(**mapping)
            session.add(av)
            count += 1
        except IntegrityError:
            session.rollback()
    session.commit()
    print(f"      -> {count} asset-vulnerability mappings created.")


def print_summary(session):
    """Print a final data integrity summary."""
    print("\n[5/5] Ingestion Summary")
    print("=" * 50)
    print(f"  Departments        : {session.query(Department).count()}")
    print(f"  Assets             : {session.query(Asset).count()}")
    print(f"  Vulnerabilities    : {session.query(Vulnerability).count()}")
    null_cvss = session.query(Vulnerability).filter(Vulnerability.cvss_score == None).count()  # noqa
    print(f"  |-- with NULL CVSS : {null_cvss}  (300 expected)")
    kev_count = session.query(Vulnerability).filter(Vulnerability.kev == 1).count()
    print(f"  |-- KEV confirmed  : {kev_count}")
    print(f"  Asset-CVE Mappings : {session.query(AssetVulnerability).count()}")
    print("=" * 50)
    print("\n[CyberGuard] Database seeding COMPLETE.\n")


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def run_seed():
    print("\n" + "=" * 50)
    print("  CyberGuard AI -- Database Seeding Pipeline")
    print("=" * 50)

    # 1. Initialise schema
    init_db()

    Session = get_session_factory()
    session = Session()

    try:
        seed_departments(session)
        seed_assets(session)
        seed_vulnerabilities(session, get_risk_features_path())
        seed_asset_vulnerabilities(session)
        print_summary(session)
    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] Seeding failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    run_seed()
