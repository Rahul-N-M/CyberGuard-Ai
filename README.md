# CyberGuard AI

## AI-Driven Business-Aware Cybersecurity Risk Prioritization and Remediation Optimization

CyberGuard AI is an AI-driven cybersecurity system designed to identify, analyze, prioritize, and optimize the remediation of vulnerabilities in an enterprise environment.

The project combines real-world vulnerability intelligence with enterprise asset information, machine learning, and remediation optimization to prioritize vulnerabilities based on both technical severity and business impact.

---

## Project Overview

Modern enterprises face a large number of software vulnerabilities, making it difficult for security teams to determine which vulnerabilities should be addressed first.

CyberGuard AI addresses this problem by combining:

- National Vulnerability Database (NVD)
- Exploit Prediction Scoring System (EPSS)
- CISA Known Exploited Vulnerabilities (KEV)
- Enterprise asset information
- Business impact and asset criticality
- Machine learning-based risk prioritization
- Remediation optimization
- Interactive security analytics

---

# Current Project Status



### Data Sources

- **NVD** – National Vulnerability Database
- **EPSS** – Exploit Prediction Scoring System
- **CISA KEV** – Known Exploited Vulnerabilities

### Current Dataset

- NVD CVEs collected: **6,316**
- EPSS records matched: **6,016**
- CVEs without EPSS information: **300**
- CISA KEV records: **1,685**
- CVEs identified in KEV: **46**
- Duplicate CVEs: **0**

### Completed Work

- [x] NVD API integration
- [x] NVD vulnerability collection
- [x] EPSS API integration
- [x] EPSS data collection
- [x] CISA KEV data collection
- [x] NVD + EPSS + KEV dataset integration
- [x] Data quality analysis
- [x] Data quality report
- [x] Feature engineering
- [x] GitHub repository setup

### Generated Datasets

```text
data/
├── raw/
│   ├── nvd_data.csv
│   ├── epss_data.csv
│   └── kev_data.csv
│
└── processed/
    ├── cyberguard_dataset.csv
    ├── risk_features.csv
    └── data_quality_report.txt
