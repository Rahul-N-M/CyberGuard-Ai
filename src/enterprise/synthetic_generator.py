"""
src/enterprise/synthetic_generator.py
Generates realistic asset-vulnerability mappings.

Approach (NOT random assignment):
  1. For each CVE in the database, scan its description for software keywords.
  2. Compare against each asset's installed_software tags.
  3. Assign the CVE to every asset whose software profile matches.
  4. For CVEs with no keyword match, distribute them proportionally to assets
     based on their criticality (more critical assets get more unmatched CVEs).
  5. Compute realistic remediation_hours based on:
       - CVSS severity (higher = more testing overhead)
       - Asset criticality (critical assets need more change-management overhead)
       - Internet exposure (exposed assets require faster but more careful patching)
       - Remediation type (Patch < Config Change < Upgrade < Architecture Change)

Output: list of dicts ready for AssetVulnerability table insertion.
"""

import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.database.models import Vulnerability, Asset


# ─────────────────────────────────────────────────────────────
# REMEDIATION TYPE MAPPING
# ─────────────────────────────────────────────────────────────
REMEDIATION_TYPES = ["Patch", "Config Change", "Workaround", "Upgrade"]

# Base hours per remediation type
REMEDIATION_BASE_HOURS = {
    "Patch":         1.5,
    "Config Change": 1.0,
    "Workaround":    0.5,
    "Upgrade":       8.0,
}

# Criticality multiplier for remediation effort
# (Critical assets need more testing, change-management, and rollback planning)
CRITICALITY_MULTIPLIER = {
    "LOW":       1.0,
    "MEDIUM":    1.3,
    "HIGH":      1.8,
    "VERY_HIGH": 2.5,
    "CRITICAL":  3.5,
}

# CVSS-based overhead multiplier (higher severity = more careful patching)
def _cvss_multiplier(cvss_score) -> float:
    if cvss_score is None:
        return 1.2   # Unknown severity: treat conservatively
    if cvss_score >= 9.0:
        return 2.5   # Critical — full testing cycle required
    if cvss_score >= 7.0:
        return 1.8   # High
    if cvss_score >= 4.0:
        return 1.2   # Medium
    return 1.0       # Low


def _compute_remediation_hours(
    vuln: Vulnerability,
    asset: Asset,
    remediation_type: str,
) -> float:
    """
    Compute realistic engineer-hours to remediate a CVE on a specific asset.
    Capped between 0.5 and 40.0 hours.
    """
    base = REMEDIATION_BASE_HOURS[remediation_type]
    crit_mult = CRITICALITY_MULTIPLIER.get(asset.criticality, 1.0)
    cvss_mult = _cvss_multiplier(vuln.cvss_score)

    # Internet-exposed assets need faster patches but with extra verification
    exposure_mult = 1.3 if asset.internet_exposure else 1.0

    hours = base * crit_mult * cvss_mult * exposure_mult

    # Add a small random variance (±15%) to make numbers realistic
    variance = random.uniform(0.85, 1.15)
    hours = hours * variance

    # Round to nearest 0.5 hour and clamp
    hours = round(hours * 2) / 2
    return max(0.5, min(40.0, hours))


def _pick_remediation_type(vuln: Vulnerability) -> str:
    """
    Select a remediation type based on vulnerability characteristics.
    - KEV or CVSS>=9: likely requires a Patch or Upgrade
    - Medium severity: Config Change or Patch
    - Low severity: Workaround or Config Change
    """
    cvss = vuln.cvss_score or 5.0
    kev  = bool(vuln.kev)

    if kev or cvss >= 9.0:
        return random.choices(["Patch", "Upgrade"], weights=[0.6, 0.4])[0]
    if cvss >= 7.0:
        return random.choices(["Patch", "Config Change", "Upgrade"], weights=[0.6, 0.3, 0.1])[0]
    if cvss >= 4.0:
        return random.choices(["Config Change", "Patch", "Workaround"], weights=[0.5, 0.35, 0.15])[0]
    return random.choices(["Workaround", "Config Change"], weights=[0.5, 0.5])[0]


# ─────────────────────────────────────────────────────────────
# KEYWORD MATCHING
# ─────────────────────────────────────────────────────────────
def _asset_software_tags(asset: Asset) -> List[str]:
    """Return lowercase software tag list for an asset."""
    if not asset.installed_software:
        return []
    return [tag.strip().lower() for tag in asset.installed_software.split(",")]


def _description_matches_asset(description: str, asset: Asset) -> bool:
    """
    Returns True if any of the asset's software tags appear in the CVE description.
    Case-insensitive keyword search — same approach as CPE matching.
    """
    if not description:
        return False
    desc_lower = description.lower()
    tags = _asset_software_tags(asset)
    return any(tag in desc_lower for tag in tags if len(tag) > 2)


# ─────────────────────────────────────────────────────────────
# CRITICALITY WEIGHT (for distributing unmatched CVEs)
# ─────────────────────────────────────────────────────────────
CRITICALITY_WEIGHT = {
    "LOW":       1,
    "MEDIUM":    2,
    "HIGH":      4,
    "VERY_HIGH": 7,
    "CRITICAL":  10,
}


def _weighted_random_assets(assets: List[Asset], k: int) -> List[Asset]:
    """
    Sample k assets with probability proportional to criticality weight.
    Higher-criticality assets are more likely to receive unmatched CVEs.
    """
    weights = [CRITICALITY_WEIGHT.get(a.criticality, 1) for a in assets]
    total   = sum(weights)
    probs   = [w / total for w in weights]
    selected = set()
    result   = []
    attempts = 0
    while len(result) < k and attempts < k * 10:
        idx = random.choices(range(len(assets)), weights=probs, k=1)[0]
        if assets[idx].asset_id not in selected:
            selected.add(assets[idx].asset_id)
            result.append(assets[idx])
        attempts += 1
    return result


# ─────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_asset_vulnerability_mappings(session) -> List[Dict[str, Any]]:
    """
    Core function called by seed_database.py.
    Returns a list of dicts, each representing one AssetVulnerability row.

    Strategy:
      1. For every CVE: keyword-match description against asset software tags.
      2. If ≥1 match found → assign CVE to all matched assets.
      3. If 0 matches → assign to 1-3 random assets (weighted by criticality).
      4. Compute remediation hours and type for each assignment.
    """
    random.seed(42)  # Reproducible results

    vulns  = session.query(Vulnerability).all()
    assets = session.query(Asset).all()

    print(f"      Matching {len(vulns)} CVEs against {len(assets)} assets...")

    mappings = []
    matched_count   = 0
    unmatched_count = 0
    seen_pairs = set()  # Deduplicate (asset_id, cve_id)

    for vuln in vulns:
        desc = vuln.description or ""

        # Step 1: Keyword matching
        matched_assets = [
            a for a in assets
            if _description_matches_asset(desc, a)
        ]

        # Step 2: Fallback for unmatched CVEs
        if not matched_assets:
            # Assign to 1-3 assets weighted by criticality
            k = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
            matched_assets = _weighted_random_assets(assets, k)
            unmatched_count += 1
        else:
            matched_count += 1

        # Step 3: Create mapping rows
        for asset in matched_assets:
            pair = (asset.asset_id, vuln.cve_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            rem_type  = _pick_remediation_type(vuln)
            rem_hours = _compute_remediation_hours(vuln, asset, rem_type)

            # Randomise discovered_date within the last 12 months
            days_ago = random.randint(0, 365)
            disc_date = datetime.utcnow() - timedelta(days=days_ago)

            mappings.append({
                "asset_id":          asset.asset_id,
                "cve_id":            vuln.cve_id,
                "remediation_hours": rem_hours,
                "remediation_type":  rem_type,
                "status":            "Open",
                "discovered_date":   disc_date,
            })

    print(f"      CVEs matched via keywords  : {matched_count}")
    print(f"      CVEs assigned via fallback : {unmatched_count}")
    print(f"      Total mappings generated   : {len(mappings)}")
    return mappings
