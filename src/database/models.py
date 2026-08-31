"""
src/database/models.py
SQLAlchemy ORM models for CyberGuard AI.

Tables:
  - departments        → Business units owning assets
  - assets             → Enterprise asset inventory (servers, databases, laptops…)
  - vulnerabilities    → CVE records from NVD/EPSS/KEV pipeline (Rahul's output)
  - asset_vulnerabilities → Many-to-many mapping: which CVE affects which asset,
                            with remediation hours and status
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    Text, DateTime, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ─────────────────────────────────────────────────────────────
# DEPARTMENTS
# ─────────────────────────────────────────────────────────────
class Department(Base):
    """Business unit / department that owns enterprise assets."""
    __tablename__ = "departments"

    department_id   = Column(Integer, primary_key=True, autoincrement=True)
    department_name = Column(String(100), nullable=False, unique=True)
    lead_contact    = Column(String(100), nullable=True)

    # Relationship
    assets = relationship("Asset", back_populates="department")

    def __repr__(self):
        return f"<Department(id={self.department_id}, name={self.department_name})>"


# ─────────────────────────────────────────────────────────────
# ASSETS
# ─────────────────────────────────────────────────────────────
class Asset(Base):
    """
    Enterprise asset — represents a server, database, laptop, etc.
    Business context fields (criticality, exposure, business impact)
    are what CyberGuard AI uses on top of plain CVSS scores.
    """
    __tablename__ = "assets"

    asset_id             = Column(String(50), primary_key=True)
    asset_name           = Column(String(150), nullable=False)
    asset_type           = Column(String(50), nullable=False)   # Server, Database, Laptop, etc.
    criticality          = Column(String(20), nullable=False)   # LOW / MEDIUM / HIGH / VERY_HIGH / CRITICAL
    internet_exposure    = Column(Boolean, nullable=False, default=False)
    business_impact_score= Column(Integer, nullable=False)      # 1–10 scale
    department_id        = Column(Integer, ForeignKey("departments.department_id"), nullable=True)
    ip_address           = Column(String(45), nullable=True)
    installed_software   = Column(Text, nullable=True)          # Comma-separated software tags
    environment          = Column(String(30), nullable=True)    # Production / Staging / Development / Corporate
    description          = Column(Text, nullable=True)

    # Relationships
    department              = relationship("Department", back_populates="assets")
    asset_vulnerabilities   = relationship("AssetVulnerability", back_populates="asset")

    def __repr__(self):
        return f"<Asset(id={self.asset_id}, name={self.asset_name}, criticality={self.criticality})>"


# ─────────────────────────────────────────────────────────────
# VULNERABILITIES
# ─────────────────────────────────────────────────────────────
class Vulnerability(Base):
    """
    CVE vulnerability record ingested from Rahul's risk_features.csv.

    IMPORTANT: 300 records have NULL CVSS/EPSS/Severity — these are preserved
    as NULL (not filled with 0) per team handoff instructions.
    """
    __tablename__ = "vulnerabilities"

    cve_id                = Column(String(30), primary_key=True)
    cvss_score            = Column(Float,   nullable=True)   # NULL for 300 incomplete records
    cvss_version          = Column(String(10), nullable=True)
    severity              = Column(String(20), nullable=True)  # LOW/MEDIUM/HIGH/CRITICAL or NULL
    description           = Column(Text,    nullable=True)
    published_date        = Column(DateTime, nullable=True)
    epss_score            = Column(Float,   nullable=True)   # NULL for 300 incomplete records
    epss_percentile       = Column(Float,   nullable=True)
    kev                   = Column(Integer, nullable=False, default=0)  # 0 or 1
    vulnerability_age_days= Column(Integer, nullable=True)
    severity_encoded      = Column(Integer, nullable=True)   # 1=LOW,2=MED,3=HIGH,4=CRIT; NULL if severity unknown
    cvss_epss_interaction = Column(Float,   nullable=True)
    kev_epss_interaction  = Column(Float,   nullable=True)

    # Relationship
    asset_vulnerabilities = relationship("AssetVulnerability", back_populates="vulnerability")

    def __repr__(self):
        return f"<Vulnerability(cve_id={self.cve_id}, cvss={self.cvss_score}, kev={self.kev})>"


# ─────────────────────────────────────────────────────────────
# ASSET VULNERABILITIES (Mapping / Junction Table)
# ─────────────────────────────────────────────────────────────
class AssetVulnerability(Base):
    """
    Links a CVE to a specific enterprise asset.
    Contains business-critical remediation context:
      - remediation_hours: estimated engineering time to fix
      - remediation_type:  type of fix required
      - status:            current fix status
    This table is the KEY INPUT for Vinod's OR-Tools Knapsack Optimizer.
    """
    __tablename__ = "asset_vulnerabilities"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    asset_id          = Column(String(50), ForeignKey("assets.asset_id"), nullable=False)
    cve_id            = Column(String(30), ForeignKey("vulnerabilities.cve_id"), nullable=False)
    remediation_hours = Column(Float, nullable=False)       # 1.0–40.0 engineer-hours
    remediation_type  = Column(String(30), nullable=True)   # Patch / Config Change / Workaround / Upgrade
    status            = Column(String(20), nullable=False, default="Open")  # Open / In Progress / Mitigated
    discovered_date   = Column(DateTime, nullable=True, default=datetime.utcnow)

    # Constraints
    __table_args__ = (
        UniqueConstraint("asset_id", "cve_id", name="uq_asset_cve"),
    )

    # Relationships
    asset         = relationship("Asset", back_populates="asset_vulnerabilities")
    vulnerability = relationship("Vulnerability", back_populates="asset_vulnerabilities")

    def __repr__(self):
        return (
            f"<AssetVulnerability(asset={self.asset_id}, "
            f"cve={self.cve_id}, hours={self.remediation_hours})>"
        )
