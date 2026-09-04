# CyberGuard AI

## AI-Driven Business-Aware Cybersecurity Risk Prioritization and Remediation Optimization

CyberGuard AI is an AI-driven cybersecurity system that identifies, prioritizes, and optimizes the remediation of vulnerabilities across enterprise assets — combining real-world threat intelligence with business context and resource-constrained optimization.

---

## Research Novelty

CyberGuard AI makes three specific research contributions:

1. **Shift from Ranking to Constrained Selection** — Existing tools only produce a sorted list. CyberGuard AI solves a 0-1 Knapsack optimization problem (via Google OR-Tools) to select the mathematically optimal *combination* of fixes within a limited engineering budget (5 / 10 / 20 / 40 hours).

2. **Tri-Factor Threat + Business Fusion** — Fuses live threat signals (CVSS, EPSS, CISA KEV) with internal enterprise parameters (asset criticality, internet exposure, business impact, remediation hours) into a unified LightGBM ML model. Neither component alone is sufficient.

3. **Three-Way Explainability** — Every recommendation carries three auditable justifications: model explanation (what threat telemetry drove it), business explanation (why this asset matters), and optimization explanation (why this fix beats other candidates given the budget).

---

## System Architecture

```
NVD + EPSS + CISA KEV
        │
        ▼
[Rahul] Data Engineering Pipeline
        │  data/processed/risk_features.csv
        ▼
[Varun] Enterprise Context & Database Layer
        │  • PostgreSQL / SQLite schema
        │  • 30-asset ShopEasy enterprise simulation
        │  • Realistic CVE-to-asset mapping (keyword matching)
        │  • Remediation hours computation
        │  data/processed/cyberguard_master_enterprise_dataset.csv
        ▼
[Vinod] ML Risk Model + Optimization
        │  • LightGBM business-aware risk scoring
        │  • Google OR-Tools Knapsack optimization
        ▼
[Navya] Streamlit Dashboard
        • Interactive budget slider (5 / 10 / 20 / 40 hours)
        • Baseline comparisons (CVSS-only / EPSS-only / KEV-first / Weighted)
        • Historical backtesting results
```

---

## Project Status

| Module | Owner | Status |
|--------|-------|--------|
| NVD + EPSS + KEV data pipeline + feature engineering | **Rahul** | ✅ Complete |
| Enterprise context, PostgreSQL schema, asset-CVE mapping | **Varun** | ✅ Complete |
| LightGBM ML risk model + OR-Tools optimization | **Vinod** | 🔄 In Progress |
| Streamlit dashboard + evaluation + integration | **Navya** | 🔄 In Progress |

---

## Data Sources

### Real Vulnerability Intelligence
- **NVD** – National Vulnerability Database (CVE records + CVSS scores)
- **EPSS** – Exploit Prediction Scoring System (exploitation probability)
- **CISA KEV** – Known Exploited Vulnerabilities (confirmed real-world attacks)

### Synthetic Enterprise Data (Clearly Labeled)
- **ShopEasy** – Simulated e-commerce enterprise with 30 assets across 7 departments
- Realistic asset software stacks drive CVE-to-asset assignment (not random)
- Remediation hours modeled from CVSS severity, asset criticality, and patch complexity

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| NVD CVEs collected | **6,316** |
| EPSS records matched | **6,016** |
| CVEs without EPSS (NULL preserved) | **300** |
| CISA KEV records | **1,685** |
| CVEs confirmed in KEV | **46** |
| Duplicate CVEs | **0** |
| Enterprise assets (ShopEasy) | **30** |
| Departments | **7** |

---

## Module 3: Enterprise Context & Database (Varun)

### Database Schema

```
departments
  └── department_id (PK), department_name, lead_contact

assets
  └── asset_id (PK), asset_name, asset_type, criticality,
      internet_exposure, business_impact_score (1–10),
      installed_software, environment, department_id (FK)

vulnerabilities
  └── cve_id (PK), cvss_score*, cvss_version*, severity*,
      description, published_date, epss_score*, epss_percentile*,
      kev, vulnerability_age_days, severity_encoded,
      cvss_epss_interaction, kev_epss_interaction
      (* = NULL for 300 incomplete records — preserved intentionally)

asset_vulnerabilities
  └── id (PK), asset_id (FK), cve_id (FK),
      remediation_hours (0.5–40.0), remediation_type,
      status, discovered_date
```

### Enterprise Asset Inventory (ShopEasy)

| Asset | Criticality | Internet Exposed | Business Impact |
|-------|-------------|-----------------|----------------|
| Payment Processing Server | CRITICAL | ✅ | 10/10 |
| Customer Database Server | CRITICAL | ❌ | 10/10 |
| Authentication & SSO Server | HIGH | ✅ | 9/10 |
| Financial Reporting Database | VERY_HIGH | ❌ | 9/10 |
| HR & Payroll System | HIGH | ❌ | 8/10 |
| ERP System (SAP) | HIGH | ❌ | 8/10 |
| Developer Secrets Manager | HIGH | ❌ | 8/10 |
| Cloud Storage Bucket | HIGH | ✅ | 8/10 |
| Load Balancer / API Gateway | HIGH | ✅ | 8/10 |
| Public Web Server | HIGH | ✅ | 8/10 |
| … (30 assets total) | | | |

### CVE-to-Asset Matching Strategy
CVEs are assigned to assets based on keyword matching between the CVE description and each asset's installed software tags (e.g., Nginx/Apache CVEs → Web Server; SQL/PostgreSQL CVEs → Customer Database). Unmatched CVEs are distributed to assets proportionally weighted by criticality score.

---

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Database (Optional — SQLite is default)

```bash
cp .env.example .env
# Edit .env to set DB_ENGINE=postgresql if you have PostgreSQL running
# Leave as DB_ENGINE=sqlite for zero-setup local development
```

### 3. Seed the Database

```bash
python -m src.database.seed_database
```

This will:
- Create all tables (departments, assets, vulnerabilities, asset_vulnerabilities)
- Ingest 6,316 CVEs from `data/processed/risk_features.csv` (preserving 300 NULLs)
- Seed 7 departments and 30 enterprise assets
- Generate realistic CVE-to-asset mappings with remediation hours

### 4. Export Datasets for ML and Dashboard

```bash
python -m src.enterprise.export_datasets
```

Outputs:
- `data/processed/enterprise_assets.csv`
- `data/processed/asset_vulnerabilities.csv`
- `data/processed/cyberguard_master_enterprise_dataset.csv`  ← **Primary ML input**
- `data/processed/enterprise_risk_summary.csv`

### 5. Run Tests

```bash
python -m pytest tests/test_enterprise_database.py -v
```

---

## Generated Datasets

```
data/
├── raw/
│   ├── nvd_data.csv
│   ├── epss_data.csv
│   └── kev_data.csv
│
└── processed/
    ├── cyberguard_dataset.csv                        ← Rahul: merged NVD+EPSS+KEV
    ├── risk_features.csv                             ← Rahul: feature-engineered CVE data
    ├── data_quality_report.txt                       ← Rahul: data quality summary
    ├── enterprise_assets.csv                         ← Varun: 30-asset inventory
    ├── asset_vulnerabilities.csv                     ← Varun: CVE-to-asset mappings
    ├── enterprise_risk_summary.csv                   ← Varun: per-asset risk summary
    └── cyberguard_master_enterprise_dataset.csv      ← Varun: UNIFIED ML FEATURE MATRIX
```

### Master Dataset Columns

| Column | Source | Description |
|--------|--------|-------------|
| `cve_id` | NVD | CVE identifier |
| `cvss_score` | NVD | Technical severity (0–10), NULL for 300 records |
| `epss_score` | EPSS | Exploitation probability (0–1), NULL for 300 records |
| `kev` | CISA | 1 = confirmed exploited in the wild |
| `severity_encoded` | Rahul | 1=LOW, 2=MED, 3=HIGH, 4=CRIT |
| `cvss_epss_interaction` | Rahul | CVSS × EPSS feature |
| `kev_epss_interaction` | Rahul | KEV × EPSS feature |
| `vulnerability_age_days` | Rahul | Days since published |
| `criticality` | Varun | Asset criticality level |
| `internet_exposure` | Varun | 1 = internet-facing asset |
| `business_impact_score` | Varun | 1–10 business impact rating |
| `asset_type` | Varun | Server / Database / Laptop / Network / Cloud |
| `environment` | Varun | Production / Staging / Development / Corporate |
| `remediation_hours` | Varun | Estimated engineer-hours to fix |
| `remediation_type` | Varun | Patch / Config Change / Workaround / Upgrade |

---

## Evaluation & Baseline Comparisons

CyberGuard AI is evaluated against 4 industry-standard approaches:

| Strategy | Business Context | Budget-Aware | Explanation |
|----------|-----------------|-------------|-------------|
| CVSS-Only | ❌ | ❌ | Single severity score |
| EPSS-Only | ❌ | ❌ | Single exploit probability |
| KEV-First | ❌ | ❌ | Known exploits only |
| Weighted Formula | ❌ | ❌ | Static linear combination |
| **CyberGuard AI** | **✅** | **✅** | **ML + Knapsack + Explainability** |

Metrics: Risk Reduction per Engineer-Hour at budgets of 5 / 10 / 20 / 40 hours, Precision@K, Recall@K, NDCG (historical backtesting against CISA KEV additions).

---

## Machine Learning Pipeline

The CyberGuard AI vulnerability prioritization engine utilizes a chronological, leakage-free temporal machine learning pipeline:

```
NVD CVE data
   │
   ▼
EPSS matching (95.68% coverage)
   │
   ▼
Temporal observation construction (monthly panel snapshots)
   │
   ▼
180-day future KEV target definition (Target_KEV_180d)
   │
   ▼
Chronological train/validation/test split (2022 / 2023 / 2024)
   │
   ▼
Preprocessing & feature encoding
   │
   ▼
LightGBM temporal baseline (scale_pos_weight = 1037.25)
   │
   ▼
Probability prediction
   │
   ▼
Threshold / ranking evaluation & calibration diagnostics
   │
   ▼
Future enterprise prioritization / Google OR-Tools integration
```

### Dataset Scale

- **Total CVEs**: 98,084 CVEs covering 2022–2024
- **EPSS Matches**: 93,844 records matched
- **EPSS Coverage**: **95.68%**
- **Temporal Observations**: **1,535,261** total panel observations
- **Positive Observations**: **601** (`Target_KEV_180d = 1`)
- **Unique Positive CVEs**: **215**

### Chronological Split

To ensure zero temporal data leakage, splits are partitioned chronologically by observation year:

- **2022 (Train)**: 141,202 observations | 136 positive observations | 53 unique positive CVEs
- **2023 (Validation)**: 485,969 observations | 217 positive observations | 72 unique positive CVEs
- **2024 (Test)**: 908,090 observations | 248 positive observations | 90 unique positive CVEs

### Model Discrimination & Test Results

- **Training (2022)**: ROC-AUC = `0.689193`, PR-AUC = `0.001922`
- **Validation (2023)**: ROC-AUC = `0.670827`, PR-AUC = `0.001115`
- **Test (2024)**: ROC-AUC = **0.629073**, PR-AUC = **0.000863**

#### Operating Threshold 0.95 Evaluation (2024 Test Set)

- **True Positives (TP)**: 109
- **False Positives (FP)**: 81,851
- **True Negatives (TN)**: 825,991
- **False Negatives (FN)**: 139
- **Precision**: 0.001330 (0.133%)
- **Recall**: **43.95%** (109 / 248)
- **F1-Score**: 0.002652

> **Important Distinction**:  
> "The 43.95% recall is obtained using a threshold of 0.95 and does NOT represent Top-500 recall."

#### Corrected Top-K Evaluation (2024 Test Set)

- **Precision@500**: **0.002000**
- **Recall@500**: **0.004032** (0.4032%, capturing 1 TP out of 248)

#### Probability Saturation & Baseline Limitations

- **Probability Saturation**: 79,693 test observations (8.78% of the test set) received a predicted probability of `1.0000`. This massive tie group creates severe limitations for standalone Top-K ranking.
- **Feature Reliance**: Saturation-group observations have a mean `Vulnerability_Age_Days` of 110.7 days (vs 448.2 days for prob < 1.0) and mean `CVSS_Score` of 8.66 (vs 6.66). The model strongly relies on vulnerability age (77.4% feature gain) and CVSS score (14.7% gain) as cohort proxies.
- **Classification**: This model is a **Temporal LightGBM Baseline**, **NOT production-ready**. Standalone Top-K prioritization is not yet viable without richer technical features, probability calibration, and downstream enterprise asset-risk filtering.

---

## Pending Work

The following next steps are directly supported by the current empirical findings:

1. Investigate and resolve the 2,993 stored-vs-computed vulnerability-age mismatches.
2. Enrich features with NVD CVSS vector components:
   - Attack Vector (AV)
   - Attack Complexity (AC)
   - Privileges Required (PR)
   - Scope (S)
   - CWE classification taxonomy
3. Evaluate relative/percentile vulnerability-age features to avoid cohort proxy saturation.
4. Retrain and compare improved temporal LightGBM models against this baseline.
5. Apply probability calibration (isotonic regression / Platt scaling) and evaluate whether ranking improves.
6. Improve Top-K discrimination within high-probability tie clusters.
7. Integrate enterprise asset context (business impact score, asset criticality, internet exposure) with vulnerability risk.
8. Implement/validate Google OR-Tools remediation-budget optimization (0-1 Knapsack across 5h / 10h / 20h / 40h budgets).
9. Perform additional temporal/generalization validation across multi-year sliding windows.
10. Define final operational threshold based on enterprise remediation capacity and SLA requirements.

---

## Technology Stack

| Purpose | Tool |
|---------|------|
| Programming language | Python |
| Data handling | Pandas, NumPy |
| Database ORM | SQLAlchemy |
| Database | PostgreSQL / SQLite |
| Machine learning | Scikit-learn, LightGBM |
| Optimization | Google OR-Tools |
| Dashboard | Streamlit |
| Charts / visuals | Plotly |
| Testing | pytest |
| Version control | Git / GitHub |

---

## Team

| Member | Responsibility | Status |
|--------|---------------|--------|
| **Rahul** | NVD + EPSS + KEV + Data Pipeline + Feature Engineering | ✅ Complete |
| **Varun** | Enterprise Context + PostgreSQL Schema + Asset Mapping + Data Export | ✅ Complete |
| **Vinod** | LightGBM Risk Model + OR-Tools Optimization | 🔄 In Progress |
| **Navya** | Streamlit Dashboard + Evaluation + Integration | 🔄 In Progress |

> For Vinod: Load `data/processed/cyberguard_master_enterprise_dataset.csv` — see `docs/handoff.md` for full specs.
>
> For Navya: Load `data/processed/enterprise_risk_summary.csv` for asset charts — see `docs/handoff.md` for full specs.
