# CyberGuard AI

## AI-Driven Enterprise Cybersecurity Risk Detection and Remediation

CyberGuard AI is an AI-driven cybersecurity system designed to identify, analyze, prioritize, and optimize the remediation of vulnerabilities in an enterprise environment.

The project combines real-world vulnerability intelligence with a simulated enterprise environment to provide risk-aware vulnerability prioritization and remediation recommendations.

---

## Project Overview

Modern enterprises face a large number of software vulnerabilities, making it difficult for security teams to determine which vulnerabilities should be addressed first.

CyberGuard AI addresses this problem by combining:

- National Vulnerability Database (NVD)
- Exploit Prediction Scoring System (EPSS)
- CISA Known Exploited Vulnerabilities (KEV)
- Enterprise asset information
- Machine learning-based vulnerability prioritization
- Optimization-based remediation planning
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

### Data Pipeline

```text
NVD API
   |
   v
NVD Vulnerability Dataset
   |
   +------> EPSS API
   |           |
   |           v
   |      EPSS Scores
   |
   +------> CISA KEV
               |
               v
        Known Exploited Status
               |
               v
       Data Merging & Validation
               |
               v
     cyberguard_dataset.csv# CyberGuard-Ai
AI-Driven Business-Aware Cybersecurity Risk Prioritization and Remediation Optimization
