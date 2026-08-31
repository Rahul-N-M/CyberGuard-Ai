"""
src/database/config.py
Database configuration — reads from .env file.
Supports PostgreSQL (primary) and SQLite (fallback for local dev).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


def get_database_url() -> str:
    """
    Returns the SQLAlchemy database connection URL.
    Reads DB_ENGINE from environment. Defaults to SQLite.
    """
    engine = os.getenv("DB_ENGINE", "sqlite").lower()

    if engine == "postgresql":
        user     = os.getenv("POSTGRES_USER", "cyberguard")
        password = os.getenv("POSTGRES_PASSWORD", "password")
        host     = os.getenv("POSTGRES_HOST", "localhost")
        port     = os.getenv("POSTGRES_PORT", "5432")
        db       = os.getenv("POSTGRES_DB", "cyberguard_db")
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    # SQLite fallback — always works without any installation
    sqlite_path = os.getenv("SQLITE_PATH", "data/cyberguard.db")
    # Ensure parent directory exists
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path}"


def get_risk_features_path() -> str:
    return os.getenv("RISK_FEATURES_PATH", "data/processed/risk_features.csv")


def get_master_dataset_output() -> str:
    return os.getenv(
        "MASTER_DATASET_OUTPUT",
        "data/processed/cyberguard_master_enterprise_dataset.csv"
    )


def get_assets_output() -> str:
    return os.getenv("ASSETS_OUTPUT", "data/processed/enterprise_assets.csv")


def get_asset_vulns_output() -> str:
    return os.getenv(
        "ASSET_VULNS_OUTPUT",
        "data/processed/asset_vulnerabilities.csv"
    )
