# CyberGuard AI — Enterprise Database Schema Reference

> **Module:** Enterprise Context & Database Architecture (Varun)  
> **Status:** Complete  
> **Last Updated:** August 2026

---

## Overview

This document is the technical reference for the database schema, data dictionary, entity relationships, and the CVE-to-asset matching rules used in CyberGuard AI's Enterprise Layer.

---

## Entity-Relationship Diagram

```
┌─────────────────────┐       ┌────────────────────────────┐
│     departments     │       │         assets             │
├─────────────────────┤       ├────────────────────────────┤
│ department_id  (PK) │◄──────│ department_id         (FK) │
│ department_name     │       │ asset_id              (PK) │
│ lead_contact        │       │ asset_name                 │
└─────────────────────┘       │ asset_type                 │
                              │ criticality                │
                              │ internet_exposure          │
                              │ business_impact_score      │
                              │ installed_software         │
                              │ environment                │
                              │ ip_address                 │
                              │ description                │
                              └────────────┬───────────────┘
                                           │
                              ┌────────────▼───────────────┐
                              │   asset_vulnerabilities    │
                              ├────────────────────────────┤
                              │ id                    (PK) │
                              │ asset_id              (FK) │◄── assets.asset_id
                              │ cve_id                (FK) │◄── vulnerabilities.cve_id
                              │ remediation_hours          │
                              │ remediation_type           │
                              │ status                     │
                              │ discovered_date            │
                              └────────────────────────────┘
                                           ▲
                              ┌────────────┴───────────────┐
                              │      vulnerabilities       │
                              ├────────────────────────────┤
                              │ cve_id                (PK) │
                              │ cvss_score          (NULL?)│
                              │ cvss_version        (NULL?)│
                              │ severity            (NULL?)│
                              │ epss_score          (NULL?)│
                              │ epss_percentile     (NULL?)│
                              │ kev                        │
                              │ vulnerability_age_days     │
                              │ severity_encoded    (NULL?)│
                              │ cvss_epss_interaction      │
                              │ kev_epss_interaction       │
                              │ published_date             │
                              │ description                │
                              └────────────────────────────┘
```

Fields marked `(NULL?)` are NULL for 300 incomplete records — intentionally preserved.

---

## Table: `departments`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `department_id` | INTEGER | NO (PK) | Auto-increment primary key |
| `department_name` | VARCHAR(100) | NO | Unique department name |
| `lead_contact` | VARCHAR(100) | YES | Email of department lead |

### Seeded Departments

| ID | Department | Lead Contact |
|----|-----------|--------------|
| 1 | Payments & Finance | payments-lead@shopeasy.com |
| 2 | Core Engineering | engineering-lead@shopeasy.com |
| 3 | IT Infrastructure | infra-lead@shopeasy.com |
| 4 | Security Operations | security-lead@shopeasy.com |
| 5 | Human Resources | hr-lead@shopeasy.com |
| 6 | Data & Analytics | data-lead@shopeasy.com |
| 7 | DevOps | devops-lead@shopeasy.com |

---

## Table: `assets`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `asset_id` | VARCHAR(50) | NO (PK) | e.g., "ASSET-001" |
| `asset_name` | VARCHAR(150) | NO | Human-readable asset name |
| `asset_type` | VARCHAR(50) | NO | Server / Database / Laptop / Network / Storage / Cloud |
| `criticality` | VARCHAR(20) | NO | LOW / MEDIUM / HIGH / VERY_HIGH / CRITICAL |
| `internet_exposure` | BOOLEAN | NO | True = reachable from internet |
| `business_impact_score` | INTEGER | NO | 1 (minimal) to 10 (catastrophic) |
| `department_id` | INTEGER | YES (FK) | References departments.department_id |
| `ip_address` | VARCHAR(45) | YES | Internal IP or NULL for cloud services |
| `installed_software` | TEXT | YES | Comma-separated technology tags |
| `environment` | VARCHAR(30) | YES | Production / Staging / Development / Corporate |
| `description` | TEXT | YES | Risk significance description |

### Criticality Scale

| Level | Description | Example Asset |
|-------|-------------|---------------|
| CRITICAL | Catastrophic impact if breached | Payment Processing Server |
| VERY_HIGH | Severe impact, near-catastrophic | Financial Reporting Database |
| HIGH | Significant business disruption | Authentication Server, Web Server |
| MEDIUM | Moderate impact, recovery possible | SIEM Server, CI/CD Pipeline |
| LOW | Minimal business impact | Test Server, Employee Laptops |

---

## Table: `vulnerabilities`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `cve_id` | VARCHAR(30) | NO (PK) | e.g., "CVE-2024-12345" |
| `cvss_score` | FLOAT | **YES** | 0.0–10.0; NULL for 300 records |
| `cvss_version` | VARCHAR(10) | YES | "3.1", "3.0", "2.0" |
| `severity` | VARCHAR(20) | **YES** | LOW/MEDIUM/HIGH/CRITICAL; NULL for 300 |
| `description` | TEXT | YES | CVE description text |
| `published_date` | DATETIME | YES | NVD publication date |
| `epss_score` | FLOAT | **YES** | 0.0–1.0 exploitation probability; NULL for 300 |
| `epss_percentile` | FLOAT | YES | EPSS percentile rank |
| `kev` | INTEGER | NO | 0 = not exploited, 1 = CISA KEV confirmed |
| `vulnerability_age_days` | INTEGER | YES | Days since published_date |
| `severity_encoded` | INTEGER | YES | 1=LOW, 2=MED, 3=HIGH, 4=CRIT; NULL for 300 |
| `cvss_epss_interaction` | FLOAT | YES | CVSS_Score × EPSS_Score |
| `kev_epss_interaction` | FLOAT | YES | KEV × EPSS_Score |

> **Important:** The 300 records with NULL CVSS/EPSS/Severity are preserved as NULL — NOT replaced with 0. Vinod should handle these in ML preprocessing (e.g., exclude from training or use median imputation).

---

## Table: `asset_vulnerabilities`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | INTEGER | NO (PK) | Auto-increment |
| `asset_id` | VARCHAR(50) | NO (FK) | References assets.asset_id |
| `cve_id` | VARCHAR(30) | NO (FK) | References vulnerabilities.cve_id |
| `remediation_hours` | FLOAT | NO | Engineer-hours to fix (0.5–40.0) |
| `remediation_type` | VARCHAR(30) | YES | Patch / Config Change / Workaround / Upgrade |
| `status` | VARCHAR(20) | NO | Open / In Progress / Mitigated |
| `discovered_date` | DATETIME | YES | When vulnerability was identified on this asset |

### Remediation Hours Model

Hours are computed (not random) from:

```
hours = base_hours × criticality_multiplier × cvss_multiplier × exposure_multiplier × variance(±15%)

base_hours:
  Patch         = 1.5h
  Config Change = 1.0h
  Workaround    = 0.5h
  Upgrade       = 8.0h

criticality_multiplier:
  LOW       = 1.0×
  MEDIUM    = 1.3×
  HIGH      = 1.8×
  VERY_HIGH = 2.5×
  CRITICAL  = 3.5×

cvss_multiplier:
  CVSS ≥ 9.0 = 2.5×   (full testing cycle)
  CVSS ≥ 7.0 = 1.8×
  CVSS ≥ 4.0 = 1.2×
  CVSS < 4.0 = 1.0×
  NULL CVSS  = 1.2×   (conservative estimate)

exposure_multiplier:
  internet_exposure = True  → 1.3×
  internet_exposure = False → 1.0×

Final: rounded to nearest 0.5h, clamped to [0.5, 40.0]
```

---

## CVE-to-Asset Matching Rules

### Step 1: Keyword Matching (Primary Strategy)
Each CVE description is searched for the asset's software tags (case-insensitive):

| Asset | Software Tags Searched |
|-------|----------------------|
| Payment Processing Server | java, spring, stripe, openssl, tls, nginx, payment, pci, encryption, tomcat |
| Customer Database Server | mysql, postgresql, sql, database, redis, storage, encryption, pii, gdpr |
| Public Web Server | nginx, apache, http, https, react, node, javascript, php, web, html, ssl, tls |
| Authentication & SSO Server | oauth, jwt, keycloak, ldap, saml, sso, kerberos, auth, session, openid, mfa |
| CI/CD Build Server | jenkins, gitlab, github, ci, cd, docker, kubernetes, build, pipeline |
| … | … |

If any tag appears in the CVE description → CVE is assigned to that asset.

### Step 2: Criticality-Weighted Fallback
CVEs with no keyword match are assigned to 1–3 assets selected with probability proportional to criticality weight:

| Criticality | Weight | Relative Probability |
|-------------|--------|---------------------|
| CRITICAL | 10 | ~27% |
| VERY_HIGH | 7 | ~19% |
| HIGH | 4 | ~11% |
| MEDIUM | 2 | ~5% |
| LOW | 1 | ~3% |

---

## Indexes & Constraints

- `asset_vulnerabilities`: Unique constraint on `(asset_id, cve_id)` — no duplicate mappings
- `departments`: Unique constraint on `department_name`
- All FK relationships are enforced at the ORM level (SQLAlchemy)
- SQLite: FKs enforced via `PRAGMA foreign_keys = ON` at connection time
