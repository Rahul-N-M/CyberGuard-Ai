"""Local-file data access for the CyberGuard-AI dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class DashboardDataError(RuntimeError):
    """Raised when a required local dashboard data source is unavailable."""


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.is_file():
        if required:
            raise DashboardDataError(f"Required data file was not found: {path.relative_to(PROJECT_ROOT)}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DashboardDataError(f"Could not read {path.relative_to(PROJECT_ROOT)}: {exc}") from exc


@st.cache_data(show_spinner=False)
def load_assets() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "enterprise_assets.csv"
    return _read_csv(path)


@st.cache_data(show_spinner=False)
def load_enterprise_actions() -> pd.DataFrame:
    """Load existing asset/CVE mappings, enriching from the inventory if needed."""
    path = DATA_DIR / "processed" / "asset_vulnerabilities.csv"
    actions = _read_csv(path, required=False)
    if actions.empty:
        actions = _read_csv(DATA_DIR / "processed" / "cyberguard_master_enterprise_dataset.csv")

    assets = load_assets()
    if "asset_name" not in actions.columns and "asset_id" in actions.columns:
        extra_cols = [
            col for col in (
                "asset_id", "asset_name", "asset_type", "criticality",
                "internet_exposure", "business_impact_score", "environment",
                "department_name",
            ) if col in assets.columns
        ]
        actions = actions.merge(assets[extra_cols], on="asset_id", how="left", suffixes=("", "_inventory"))
    if "department_name" not in actions.columns and {"asset_id", "department_name"}.issubset(assets.columns):
        actions = actions.merge(assets[["asset_id", "department_name"]], on="asset_id", how="left")

    required = {
        "asset_id", "cve_id", "cvss_score", "epss_score", "kev",
        "business_impact_score", "internet_exposure", "remediation_hours", "status",
    }
    missing = sorted(required - set(actions.columns))
    if missing:
        raise DashboardDataError(f"Enterprise mapping is missing required columns: {', '.join(missing)}")
    return actions


@st.cache_data(show_spinner=False)
def load_threat_intelligence() -> pd.DataFrame:
    """Join the repository's local NVD, EPSS, and CISA KEV exports."""
    nvd = _read_csv(DATA_DIR / "raw" / "nvd_data.csv", required=False)
    epss = _read_csv(DATA_DIR / "raw" / "epss_data.csv", required=False)
    kev = _read_csv(DATA_DIR / "raw" / "kev_data.csv", required=False)

    if nvd.empty:
        combined = _read_csv(DATA_DIR / "processed" / "cyberguard_dataset.csv")
        combined = combined.rename(columns={
            "CVE_ID": "cve_id",
            "CVSS_Score": "cvss_score",
            "CVSS_Version": "cvss_version",
            "Severity": "severity",
            "Description": "description",
            "Published_Date": "published_date",
            "EPSS_Score": "epss_score",
            "EPSS_Percentile": "epss_percentile",
            "KEV": "kev",
        })
    else:
        nvd = nvd.rename(columns={
            "CVE_ID": "cve_id",
            "CVSS_Score": "cvss_score",
            "CVSS_Version": "cvss_version",
            "Severity": "severity",
            "Description": "description",
            "Published_Date": "published_date",
        })
        combined = nvd
        if not epss.empty:
            epss = epss.rename(columns={"CVE_ID": "cve_id", "EPSS_Score": "epss_score", "EPSS_Percentile": "epss_percentile"})
            combined = combined.merge(epss.drop_duplicates("cve_id"), on="cve_id", how="left")
        if not kev.empty:
            kev = kev.rename(columns={"CVE_ID": "cve_id", "KEV": "kev"})
            kev_cols = [col for col in ("cve_id", "kev", "Date_Added", "Vendor", "Product") if col in kev.columns]
            kev = kev[kev_cols].drop_duplicates("cve_id")
            combined = combined.merge(kev, on="cve_id", how="left", suffixes=("", "_catalog"))
            if "kev_catalog" in combined:
                combined["kev"] = combined["kev_catalog"].fillna(combined.get("kev", 0))
                combined = combined.drop(columns="kev_catalog")

    required = {"cve_id", "cvss_score", "severity"}
    missing = sorted(required - set(combined.columns))
    if missing:
        raise DashboardDataError(f"Threat-intelligence data is missing columns: {', '.join(missing)}")

    for column in ("epss_score", "epss_percentile", "kev"):
        if column not in combined.columns:
            combined[column] = pd.NA
    if "published_date" in combined.columns:
        combined["published_date"] = pd.to_datetime(combined["published_date"], errors="coerce")
    else:
        combined["published_date"] = pd.NaT
    if "Date_Added" in combined.columns:
        combined["kev_date_added"] = pd.to_datetime(combined["Date_Added"], errors="coerce")
    else:
        combined["kev_date_added"] = pd.NaT
    combined["kev"] = pd.to_numeric(combined["kev"], errors="coerce")
    return combined.drop_duplicates("cve_id").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    """Load feature importance rankings from existing training outputs."""
    path = OUTPUTS_DIR / "lightgbm_feature_importance.csv"
    df = _read_csv(path, required=False)
    if not df.empty and "importance_gain" in df.columns:
        total_gain = df["importance_gain"].sum()
        if total_gain > 0:
            df["gain_pct"] = (df["importance_gain"] / total_gain) * 100
        df = df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_cve_level_scores() -> pd.DataFrame:
    """Load pre-aggregated CVE-level Borda rank and calibrated probability scores."""
    path = OUTPUTS_DIR / "v3_cve_level_scores.csv"
    df = _read_csv(path, required=False)
    if not df.empty and "CVE_ID" in df.columns:
        return df.rename(columns={"CVE_ID": "cve_id"})
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_calibrated_predictions() -> pd.DataFrame:
    """Load calibrated predictions using pre-aggregated scores if available, or test predictions."""
    cve_scores = load_cve_level_scores()
    if not cve_scores.empty and "prob_last" in cve_scores.columns:
        result = cve_scores[["cve_id", "prob_last"]].rename(columns={"prob_last": "model_probability"})
        if "borda_mean" in cve_scores.columns:
            result["borda_rank_score"] = cve_scores["borda_mean"]
        return result

    path = OUTPUTS_DIR / "v3_predictions_test.csv"
    predictions = _read_csv(path, required=False)
    expected = {"CVE_ID", "Observation_Date", "predicted_prob"}
    if predictions.empty or not expected.issubset(predictions.columns):
        return pd.DataFrame(columns=["cve_id", "model_probability"])
    predictions["Observation_Date"] = pd.to_datetime(predictions["Observation_Date"], errors="coerce")
    predictions["predicted_prob"] = pd.to_numeric(predictions["predicted_prob"], errors="coerce")
    predictions = predictions.dropna(subset=["predicted_prob"]).sort_values("Observation_Date")
    latest = predictions.drop_duplicates("CVE_ID", keep="last")
    return latest.rename(columns={"CVE_ID": "cve_id", "predicted_prob": "model_probability"})[
        ["cve_id", "model_probability"]
    ]


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    names = (
        "v3_metrics.json",
        "benchmark_2009_2015_comparison.json",
        "cyberguard_reference_metrics.json",
        "fastembed_metrics.json",
        "alperin_metrics.json",
    )
    result = {}
    for name in names:
        path = OUTPUTS_DIR / name
        if not path.is_file():
            continue
        try:
            result[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DashboardDataError(f"Could not load saved metrics {name}: {exc}") from exc
    return result


def latest_data_refresh() -> pd.Timestamp | None:
    paths = [
        DATA_DIR / "raw" / "nvd_data.csv",
        DATA_DIR / "raw" / "epss_data.csv",
        DATA_DIR / "raw" / "kev_data.csv",
        DATA_DIR / "processed" / "asset_vulnerabilities.csv",
        OUTPUTS_DIR / "v3_predictions_test.csv",
        OUTPUTS_DIR / "v3_cve_level_scores.csv",
    ]
    existing = [p.stat().st_mtime for p in paths if p.is_file()]
    return pd.to_datetime(max(existing), unit="s", utc=True) if existing else None

