from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "risk_features.csv"
REPORT_PATH = ROOT / "reports" / "ml_initial_audit.md"


def list_repo_files(root: Path) -> dict[str, list[str]]:
    """Return a compact inventory of the top-level project artifacts."""
    inventory = {
        "data_files": sorted(str(p.relative_to(root)) for p in (root / "data").rglob("*") if p.is_file()),
        "python_scripts": sorted(str(p.relative_to(root)) for p in (root / "src").rglob("*.py")),
        "notebooks": sorted(str(p.relative_to(root)) for p in (root / "notebooks").rglob("*.ipynb")) if (root / "notebooks").exists() else [],
        "requirements_files": sorted(str(p.relative_to(root)) for p in root.glob("*requirements*.txt")) + sorted(str(p.relative_to(root)) for p in root.glob("**/*requirements*.txt")),
        "ml_related_files": [],
    }

    ml_markers = ("lightgbm", "ml", "baseline", "feature", "risk", "target", "optimizer")
    for path in (root / "src").rglob("*.py"):
        rel = str(path.relative_to(root)).lower()
        if any(marker in rel for marker in ml_markers):
            inventory["ml_related_files"].append(str(path.relative_to(root)))

    inventory["ml_related_files"] = sorted(set(inventory["ml_related_files"]))
    return inventory


def summarize_existing_feature_engineering() -> str:
    lines = [
        "The current feature-engineering logic lives in `src/data_processing/feature_engineering.py`.",
        "",
        "Observed transformations:",
        "- `Published_Date` is converted to timezone-aware datetime (`utc=True`).",
        "- `Vulnerability_Age_Days` = reference_date - `Published_Date` in days.",
        "- `Severity_Encoded` maps `LOW -> 1`, `MEDIUM -> 2`, `HIGH -> 3`, `CRITICAL -> 4`.",
        "- `CVSS_EPSS_Interaction` = `CVSS_Score * EPSS_Score`.",
        "- `KEV_EPSS_Interaction` = `KEV * EPSS_Score`.",
        "",
        "Important note: the 300 incomplete records are preserved rather than zero-filled or deleted."
    ]
    return "\n".join(lines)


def summarize_target_candidates() -> str:
    candidates = [
        {
            "Candidate": "Future KEV appearance",
            "Meaning": "Whether a CVE becomes a CISA KEV item in a future time window.",
            "Data needed": "Historical CVEs, publication date, exploitation telemetry, and a future KEV label window.",
            "Leakage risk": "High if the target is derived from the same time window as the features or if future exploit knowledge is used in training.",
            "Backtesting": "Yes, with time-based splits and a forward-looking label definition.",
            "Fit": "Strong conceptual fit for real-world prioritization if the label is future-looking and non-leaking.",
        },
        {
            "Candidate": "Binary classification: current KEV flag",
            "Meaning": "A vulnerability is already in KEV (1) or not (0).",
            "Data needed": "The current dataset already contains KEV as a binary label.",
            "Leakage risk": "Moderate to high if used as a target with the same snapshot features, because it is strongly correlated with existing input features and is not a future-risk estimate.",
            "Backtesting": "Limited, because it is a contemporaneous label rather than a forward-looking risk target.",
            "Fit": "Useful for a benchmark but not ideal as the final enterprise prioritization target.",
        },
        {
            "Candidate": "Risk regression",
            "Meaning": "Predict a numeric risk score from technical and business factors.",
            "Data needed": "Target labels like remediation urgency, exploit likelihood, or a business risk score built from historical outcomes.",
            "Leakage risk": "High if the numeric target is constructed from the same fields used as model inputs or if business risk is defined from the same period.",
            "Backtesting": "Possible with a carefully defined historical target.",
            "Fit": "A valid formulation only when the target is clearly defined and temporally aligned.",
        },
        {
            "Candidate": "Ranking formulation",
            "Meaning": "Order vulnerabilities by expected importance rather than predict a single integer label.",
            "Data needed": "A ranked list from historical outcomes or labeled expert reviews.",
            "Leakage risk": "Moderate if ranking is derived from current signals; requires careful temporal splits.",
            "Backtesting": "Yes, via rank-based retrospective evaluation.",
            "Fit": "Good for prioritization, but not a direct supervised target unless a valid historical ranking is defined.",
        },
    ]

    sections = []
    for item in candidates:
        sections.append(f"### {item['Candidate']}")
        for key, value in item.items():
            if key == "Candidate":
                continue
            sections.append(f"- {key}: {value}")
        sections.append("")
    return "\n".join(sections).rstrip()


def summarize_leakage_risks(df: pd.DataFrame) -> str:
    risk_facts = [
        "- The dataset is vulnerability-level and does not yet include enterprise asset context; it is not yet the final per-asset ML dataset.",
        "- `CVSS_Score`, `EPSS_Score`, `Severity`, `KEV`, and interaction features are all derived from the same vulnerability snapshot and should not be treated as a future target when used on the same row.",
        "- Using `Description` or `Published_Date` in the same model as a target label is not appropriate without careful temporal validation, because text and publication time may encode current exposure patterns.",
        "- For future exploitation modeling, any target should be built from a later time window than the feature window to avoid look-ahead bias.",
        "- The `Severity_Encoded` field is a deterministic encoding of `Severity`, so it should not be used as a target if `Severity` also appears in input features.",
        "- If the future enterprise asset dataset is added later, the per-asset mapping must remain one vulnerability per asset row to avoid collapsing multiple assets into a single record.",
    ]
    return "\n".join(risk_facts)


def build_report() -> str:
    df = pd.read_csv(DATA_PATH)

    numeric = df.select_dtypes(include=["number"]).copy()
    categorical = df.select_dtypes(exclude=["number"]).copy()

    missing_summary = df.isna().sum().sort_values(ascending=False)
    categorical_unique = {
        col: {
            "nunique": int(df[col].nunique(dropna=True)),
            "examples": df[col].dropna().astype(str).unique()[:10].tolist(),
        }
        for col in categorical.columns
    }

    sections = [
        "# ML Initial Audit",
        "",
        "## 1. Repository inventory",
        "",
    ]

    repo_inventory = list_repo_files(ROOT)
    for key, value in repo_inventory.items():
        sections.append(f"### {key.replace('_', ' ').title()}")
        if value:
            for item in value:
                sections.append(f"- {item}")
        else:
            sections.append("- None detected")
        sections.append("")

    sections.extend([
        "## 2. Dataset profile",
        "",
        f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"- Columns: {', '.join(df.columns.tolist())}",
        "",
        "### Data types",
        "",
        df.dtypes.to_string(),
        "",
        "### Missing values",
        "",
        missing_summary.to_string(),
        "",
        "### Duplicate records",
        "",
        f"- Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "### Unique categorical values",
        "",
    ])

    for col, info in categorical_unique.items():
        sections.append(f"- {col}: unique={info['nunique']}; examples={info['examples']}")
    sections.append("")

    sections.extend([
        "### Numerical statistics",
        "",
        numeric.describe().transpose().to_string(),
        "",
        "## 3. Existing feature-engineering logic",
        "",
        summarize_existing_feature_engineering(),
        "",
        "## 4. Potential target and label candidates",
        "",
        summarize_target_candidates(),
        "",
        "## 5. Potential data leakage risks",
        "",
        summarize_leakage_risks(df),
        "",
        "## 6. Initial takeaways",
        "",
        "- The current dataset is a vulnerability-centric feature set, not the final enterprise-level row-level ML dataset.",
        "- Missingness is concentrated in `CVSS_Score`, `CVSS_Version`, `Severity`, `EPSS_Score`, `EPSS_Percentile`, `Severity_Encoded`, and interaction terms, exactly matching the 300 incomplete records described by the project documentation.",
        "- The CVSS/EPSS missingness pattern is not random but is consistent with incomplete vulnerability records that lack score metadata.",
        "- There are no duplicate CVEs, which is a good sign for record integrity.",
        "- `KEV` is binary and sparse (46 of 6316 values), which makes it useful as a signal but not a reliable target without temporal logic.",
        "- The key decision is to define a future-looking target before any LightGBM training is attempted.",
        "",
        "## Recommended next step",
        "",
        "- Define and document the approved target using a temporal split and a clearly justified label.",
        "- Confirm the precise input feature set for the approved model after the enterprise context is available.",
        "- Treat the 300 incomplete records as a missing-data problem, not as a zero-fill problem.",
        "- Do not train a final LightGBM model until this target definition is reviewed and approved.",
    ])

    return "\n".join(sections)


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Audit report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
