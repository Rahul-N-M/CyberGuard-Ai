"""
tests/test_enterprise_database.py
Automated verification tests for Varun's Enterprise & Database module.

Tests:
  1. Database connection health
  2. All 4 tables created correctly
  3. Correct record counts (6,316 vulnerabilities)
  4. NULL preservation — exactly 300 records with NULL CVSS (NOT 0)
  5. KEV record count
  6. All 30 assets seeded
  7. All 7 departments seeded
  8. Asset-vulnerability mappings exist and have realistic remediation hours
  9. No mapping has remediation_hours outside 0.5–40.0 range
 10. ML training matrix loads without error and has expected columns

Run:
    python -m pytest tests/test_enterprise_database.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd

from src.database.connection import init_db, get_session_factory, test_connection as check_db_connection
from src.database.models import Department, Asset, Vulnerability, AssetVulnerability


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def session():
    """Provides a live database session for tests."""
    init_db()
    Session = get_session_factory()
    s = Session()
    yield s
    s.close()


# ─────────────────────────────────────────────────────────────
# TEST 1: Connection Health
# ─────────────────────────────────────────────────────────────
def test_database_connection():
    """Database connection should be healthy."""
    assert check_db_connection(), "Database connection failed. Check .env settings."


# ─────────────────────────────────────────────────────────────
# TEST 2: Table Creation
# ─────────────────────────────────────────────────────────────
def test_all_tables_exist(session):
    """All 4 tables must exist and be queryable."""
    assert session.query(Department).count() >= 0
    assert session.query(Asset).count() >= 0
    assert session.query(Vulnerability).count() >= 0
    assert session.query(AssetVulnerability).count() >= 0


# ─────────────────────────────────────────────────────────────
# TEST 3: Vulnerability Record Count
# ─────────────────────────────────────────────────────────────
def test_vulnerability_count(session):
    """Should have 6,316 vulnerability records ingested from risk_features.csv."""
    count = session.query(Vulnerability).count()
    assert count == 6316, f"Expected 6316 vulnerabilities, got {count}"


# ─────────────────────────────────────────────────────────────
# TEST 4: NULL Preservation — 300 Incomplete Records
# ─────────────────────────────────────────────────────────────
def test_null_cvss_preservation(session):
    """
    Exactly 300 records should have NULL CVSS score — NOT zero.
    This verifies Rahul's 300 incomplete records were NOT overwritten with 0.
    """
    null_count = session.query(Vulnerability).filter(
        Vulnerability.cvss_score == None  # noqa: E711
    ).count()
    assert null_count == 300, (
        f"Expected 300 records with NULL CVSS, got {null_count}. "
        "Check that NaN values were correctly converted to NULL (not 0)."
    )


def test_no_zero_cvss_for_incomplete_records(session):
    """
    The 300 incomplete records should NOT have CVSS = 0.0.
    Zero would indicate incorrect imputation.
    """
    zero_cvss = session.query(Vulnerability).filter(
        Vulnerability.cvss_score == 0.0
    ).count()
    # Zero CVSS is extremely unusual; if >50 records have it, something went wrong
    assert zero_cvss < 50, (
        f"Found {zero_cvss} records with CVSS=0.0, which suggests "
        "NULL values were incorrectly replaced with 0 during ingestion."
    )


# ─────────────────────────────────────────────────────────────
# TEST 5: KEV Count
# ─────────────────────────────────────────────────────────────
def test_kev_count(session):
    """Should have 46 CISA KEV-confirmed vulnerabilities."""
    kev_count = session.query(Vulnerability).filter(Vulnerability.kev == 1).count()
    assert kev_count == 46, f"Expected 46 KEV vulnerabilities, got {kev_count}"


# ─────────────────────────────────────────────────────────────
# TEST 6: Asset Count
# ─────────────────────────────────────────────────────────────
def test_asset_count(session):
    """Should have exactly 30 enterprise assets seeded."""
    count = session.query(Asset).count()
    assert count == 30, f"Expected 30 assets, got {count}"


# ─────────────────────────────────────────────────────────────
# TEST 7: Department Count
# ─────────────────────────────────────────────────────────────
def test_department_count(session):
    """Should have exactly 7 departments seeded."""
    count = session.query(Department).count()
    assert count == 7, f"Expected 7 departments, got {count}"


# ─────────────────────────────────────────────────────────────
# TEST 8: Asset-Vulnerability Mappings Exist
# ─────────────────────────────────────────────────────────────
def test_asset_vulnerability_mappings_exist(session):
    """Asset-vulnerability mapping table should not be empty."""
    count = session.query(AssetVulnerability).count()
    assert count > 0, "No asset-vulnerability mappings found. Run seed_database.py first."
    # Expect at least 1 mapping per vulnerability on average
    vuln_count = session.query(Vulnerability).count()
    assert count >= vuln_count, (
        f"Mapping count ({count}) is less than vulnerability count ({vuln_count}). "
        "Expected at least one mapping per CVE."
    )


# ─────────────────────────────────────────────────────────────
# TEST 9: Remediation Hours Validity
# ─────────────────────────────────────────────────────────────
def test_remediation_hours_in_valid_range(session):
    """All remediation_hours values must be between 0.5 and 40.0."""
    out_of_range = session.query(AssetVulnerability).filter(
        (AssetVulnerability.remediation_hours < 0.5) |
        (AssetVulnerability.remediation_hours > 40.0)
    ).count()
    assert out_of_range == 0, (
        f"{out_of_range} asset-vulnerability mappings have remediation_hours "
        "outside the valid 0.5–40.0 hour range."
    )


def test_remediation_type_valid(session):
    """All remediation_type values must be one of the defined types."""
    valid_types = {"Patch", "Config Change", "Workaround", "Upgrade"}
    mappings = session.query(AssetVulnerability).all()
    invalid = [m for m in mappings if m.remediation_type not in valid_types]
    assert len(invalid) == 0, (
        f"{len(invalid)} mappings have invalid remediation_type values."
    )


# ─────────────────────────────────────────────────────────────
# TEST 10: High-Criticality Assets Have Vulnerabilities
# ─────────────────────────────────────────────────────────────
def test_critical_assets_have_vulnerabilities(session):
    """CRITICAL/VERY_HIGH assets (Payment Server, Customer DB) must have mapped CVEs."""
    critical_assets = session.query(Asset).filter(
        Asset.criticality.in_(["CRITICAL", "VERY_HIGH"])
    ).all()
    for asset in critical_assets:
        mapping_count = session.query(AssetVulnerability).filter(
            AssetVulnerability.asset_id == asset.asset_id
        ).count()
        assert mapping_count > 0, (
            f"Critical asset '{asset.asset_name}' ({asset.asset_id}) has no "
            "mapped vulnerabilities."
        )


# ─────────────────────────────────────────────────────────────
# TEST 11: ML Training Matrix
# ─────────────────────────────────────────────────────────────
def test_ml_training_matrix_loads():
    """ML training matrix CSV must exist and contain expected columns."""
    from src.enterprise.data_access import get_ml_training_matrix
    df = get_ml_training_matrix()
    assert len(df) > 0, "ML training matrix is empty."

    required_columns = [
        "cve_id", "asset_id", "asset_type", "criticality",
        "internet_exposure", "business_impact_score",
        "cvss_score", "epss_score", "kev",
        "vulnerability_age_days", "severity_encoded",
        "cvss_epss_interaction", "kev_epss_interaction",
        "remediation_hours", "remediation_type",
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing required column '{col}' in ML training matrix."


def test_ml_matrix_no_negative_remediation():
    """ML training matrix must not contain negative remediation hours."""
    from src.enterprise.data_access import get_ml_training_matrix
    df = get_ml_training_matrix()
    assert (df["remediation_hours"] >= 0).all(), "Found negative remediation_hours in ML matrix."
