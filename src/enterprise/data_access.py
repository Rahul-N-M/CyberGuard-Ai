"""
src/enterprise/data_access.py
Clean query API — returns Pandas DataFrames for downstream consumption.

Usage (Vinod — ML):
    from src.enterprise.data_access import get_ml_training_matrix
    df = get_ml_training_matrix()

Usage (Navya — Dashboard):
    from src.enterprise.data_access import get_all_assets, get_enterprise_risk_summary
    assets_df  = get_all_assets()
    summary_df = get_enterprise_risk_summary()
"""

import pandas as pd
from sqlalchemy import text
from src.database.connection import get_session_factory, get_engine


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────
def _query_to_df(sql: str) -> pd.DataFrame:
    """Execute raw SQL and return result as a Pandas DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ─────────────────────────────────────────────────────────────
# ASSET QUERIES
# ─────────────────────────────────────────────────────────────
def get_all_assets() -> pd.DataFrame:
    """
    Returns all enterprise assets with department name.
    Columns: asset_id, asset_name, asset_type, criticality,
             internet_exposure, business_impact_score,
             environment, department_name, installed_software
    """
    sql = """
        SELECT
            a.asset_id,
            a.asset_name,
            a.asset_type,
            a.criticality,
            a.internet_exposure,
            a.business_impact_score,
            a.environment,
            a.ip_address,
            a.installed_software,
            a.description,
            d.department_name
        FROM assets a
        LEFT JOIN departments d ON a.department_id = d.department_id
        ORDER BY a.business_impact_score DESC, a.criticality DESC
    """
    return _query_to_df(sql)


def get_assets_by_criticality(criticality: str) -> pd.DataFrame:
    """Filter assets by criticality level (LOW/MEDIUM/HIGH/VERY_HIGH/CRITICAL)."""
    sql = f"""
        SELECT a.*, d.department_name
        FROM assets a
        LEFT JOIN departments d ON a.department_id = d.department_id
        WHERE UPPER(a.criticality) = '{criticality.upper()}'
        ORDER BY a.business_impact_score DESC
    """
    return _query_to_df(sql)


def get_internet_exposed_assets() -> pd.DataFrame:
    """Returns only internet-facing assets — the highest-priority attack surface."""
    sql = """
        SELECT a.*, d.department_name
        FROM assets a
        LEFT JOIN departments d ON a.department_id = d.department_id
        WHERE a.internet_exposure = 1
        ORDER BY a.business_impact_score DESC
    """
    return _query_to_df(sql)


# ─────────────────────────────────────────────────────────────
# VULNERABILITY QUERIES
# ─────────────────────────────────────────────────────────────
def get_all_vulnerabilities(include_incomplete: bool = True) -> pd.DataFrame:
    """
    Returns all CVE records.
    Args:
        include_incomplete: If False, filters out the 300 records with NULL CVSS/EPSS.
    """
    where_clause = "" if include_incomplete else "WHERE cvss_score IS NOT NULL"
    sql = f"""
        SELECT
            cve_id, cvss_score, cvss_version, severity,
            epss_score, epss_percentile, kev,
            vulnerability_age_days, severity_encoded,
            cvss_epss_interaction, kev_epss_interaction,
            published_date, description
        FROM vulnerabilities
        {where_clause}
        ORDER BY cvss_score DESC NULLS LAST
    """
    return _query_to_df(sql)


def get_kev_vulnerabilities() -> pd.DataFrame:
    """Returns only CISA KEV-confirmed vulnerabilities (actively exploited in the wild)."""
    sql = """
        SELECT * FROM vulnerabilities
        WHERE kev = 1
        ORDER BY epss_score DESC NULLS LAST
    """
    return _query_to_df(sql)


# ─────────────────────────────────────────────────────────────
# ASSET-VULNERABILITY JOINED QUERIES
# ─────────────────────────────────────────────────────────────
def get_asset_vulnerability_pairs(
    asset_id: str = None,
    kev_only: bool = False,
    min_cvss: float = None,
    status: str = "Open",
) -> pd.DataFrame:
    """
    Returns joined asset-vulnerability pairs with all relevant features.
    This is the PRIMARY data source for Vinod's ML model and Navya's dashboard.

    Args:
        asset_id:  Filter to a specific asset (None = all assets)
        kev_only:  If True, return only KEV-confirmed CVEs
        min_cvss:  Minimum CVSS score threshold
        status:    Remediation status filter (Open/In Progress/Mitigated)
    """
    conditions = []
    if asset_id:
        conditions.append(f"av.asset_id = '{asset_id}'")
    if kev_only:
        conditions.append("v.kev = 1")
    if min_cvss is not None:
        conditions.append(f"v.cvss_score >= {min_cvss}")
    if status:
        conditions.append(f"av.status = '{status}'")

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT
            av.id               AS mapping_id,
            av.asset_id,
            a.asset_name,
            a.asset_type,
            a.criticality,
            a.internet_exposure,
            a.business_impact_score,
            a.environment,
            d.department_name,
            av.cve_id,
            v.cvss_score,
            v.cvss_version,
            v.severity,
            v.epss_score,
            v.epss_percentile,
            v.kev,
            v.vulnerability_age_days,
            v.severity_encoded,
            v.cvss_epss_interaction,
            v.kev_epss_interaction,
            av.remediation_hours,
            av.remediation_type,
            av.status,
            av.discovered_date
        FROM asset_vulnerabilities av
        JOIN assets a        ON av.asset_id = a.asset_id
        JOIN vulnerabilities v ON av.cve_id = v.cve_id
        LEFT JOIN departments d ON a.department_id = d.department_id
        {where_clause}
        ORDER BY v.cvss_score DESC NULLS LAST, a.business_impact_score DESC
    """
    return _query_to_df(sql)


# ─────────────────────────────────────────────────────────────
# ML TRAINING MATRIX (for Vinod)
# ─────────────────────────────────────────────────────────────
def get_ml_training_matrix(complete_records_only: bool = False) -> pd.DataFrame:
    """
    Returns the full feature matrix ready for LightGBM training.
    Includes all vulnerability features + all enterprise/asset context features.

    Args:
        complete_records_only: If True, excludes 300 records with NULL CVSS/EPSS.

    Feature columns:
        Vulnerability features: cvss_score, epss_score, epss_percentile, kev,
                                vulnerability_age_days, severity_encoded,
                                cvss_epss_interaction, kev_epss_interaction
        Asset features:         criticality, internet_exposure, business_impact_score,
                                asset_type, environment
        Optimization input:     remediation_hours, remediation_type
    """
    null_filter = "AND v.cvss_score IS NOT NULL" if complete_records_only else ""

    sql = f"""
        SELECT
            av.cve_id,
            av.asset_id,
            a.asset_name,
            a.asset_type,
            a.criticality,
            CAST(a.internet_exposure AS INTEGER)    AS internet_exposure,
            a.business_impact_score,
            a.environment,
            v.cvss_score,
            v.epss_score,
            v.epss_percentile,
            v.kev,
            v.vulnerability_age_days,
            v.severity_encoded,
            v.cvss_epss_interaction,
            v.kev_epss_interaction,
            v.severity,
            av.remediation_hours,
            av.remediation_type,
            av.status
        FROM asset_vulnerabilities av
        JOIN vulnerabilities v ON av.cve_id = v.cve_id
        JOIN assets a          ON av.asset_id = a.asset_id
        WHERE av.status = 'Open'
        {null_filter}
        ORDER BY av.cve_id
    """
    return _query_to_df(sql)


# ─────────────────────────────────────────────────────────────
# DASHBOARD SUMMARY (for Navya)
# ─────────────────────────────────────────────────────────────
def get_enterprise_risk_summary() -> pd.DataFrame:
    """
    Returns a per-asset risk summary for the dashboard.
    Includes: total open CVEs, KEV count, avg CVSS, avg EPSS,
              total remediation hours, and max business impact.
    """
    sql = """
        SELECT
            a.asset_id,
            a.asset_name,
            a.asset_type,
            a.criticality,
            a.internet_exposure,
            a.business_impact_score,
            a.environment,
            COUNT(av.id)                    AS total_open_cves,
            SUM(v.kev)                      AS kev_count,
            ROUND(AVG(v.cvss_score), 2)     AS avg_cvss,
            ROUND(AVG(v.epss_score), 4)     AS avg_epss,
            ROUND(SUM(av.remediation_hours), 1) AS total_remediation_hours
        FROM assets a
        LEFT JOIN asset_vulnerabilities av ON a.asset_id = av.asset_id AND av.status = 'Open'
        LEFT JOIN vulnerabilities v        ON av.cve_id = v.cve_id
        GROUP BY a.asset_id, a.asset_name, a.asset_type,
                 a.criticality, a.internet_exposure,
                 a.business_impact_score, a.environment
        ORDER BY a.business_impact_score DESC, kev_count DESC
    """
    return _query_to_df(sql)


def get_severity_distribution() -> pd.DataFrame:
    """Returns count of CVEs by severity level (for dashboard charts)."""
    sql = """
        SELECT severity, COUNT(*) AS count
        FROM vulnerabilities
        GROUP BY severity
        ORDER BY count DESC
    """
    return _query_to_df(sql)


def get_top_risk_vulnerabilities(limit: int = 20) -> pd.DataFrame:
    """
    Returns the top N highest-risk asset-vulnerability pairs.
    Ranked by: KEV status > EPSS score > CVSS score > business impact.
    """
    sql = f"""
        SELECT
            av.cve_id,
            a.asset_name,
            a.criticality,
            a.internet_exposure,
            a.business_impact_score,
            v.cvss_score,
            v.epss_score,
            v.kev,
            v.severity,
            av.remediation_hours,
            av.remediation_type
        FROM asset_vulnerabilities av
        JOIN vulnerabilities v ON av.cve_id = v.cve_id
        JOIN assets a          ON av.asset_id = a.asset_id
        WHERE av.status = 'Open'
        ORDER BY v.kev DESC, v.epss_score DESC NULLS LAST,
                 v.cvss_score DESC NULLS LAST,
                 a.business_impact_score DESC
        LIMIT {limit}
    """
    return _query_to_df(sql)
