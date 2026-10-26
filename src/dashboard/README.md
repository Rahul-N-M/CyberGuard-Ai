# CyberGuard-AI: Streamlit Risk Console

### Interactive Web Console for AI-Driven Business-Aware Cybersecurity Risk Prioritization & Remediation Optimization

The **CyberGuard-AI Streamlit Dashboard** (`src/dashboard/app.py`) provides security executives, SecOps analysts, and DevOps patch managers with an interactive, data-driven console to explore threat intelligence, contextualize vulnerability severity against enterprise asset topology, and solve constrained patch remediation optimization under strict engineer-hour budgets.

---

## 1. Quick Start: How to Run

From the repository root in Windows PowerShell:

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run the Streamlit application
python -m streamlit run src/dashboard/app.py
```

The web console will launch at `http://localhost:8501`.

> **Note on Performance:** The dashboard operates entirely on the repository's pre-computed, verified evaluation artifacts and local datasets (`data/processed/` and `outputs/`). It makes **zero live external API calls** and does not retrain models on page load, ensuring sub-second rendering.

---

## 2. Dashboard Architecture & The 4 Focused Workspaces

The console is organized into **4 distinct, non-overlapping workspaces** accessible via the sidebar navigation, structured to support a punchy, 5-minute executive review or an in-depth technical defense:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               CYBERGUARD-AI RISK CONSOLE                               │
├────────────────────────────────────────┬───────────────────────────────────────────────┤
│ 1. Executive Remediation Console       │ ★ CORE PRESENTATION SCREEN ★                  │
│                                        │ Live sprint simulation, absolute risk gain,   │
│                                        │ algorithm charts, patch plan, 3-layer         │
│                                        │ auditable explanation & 1-click CSV export    │
├────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 2. Academic Benchmarks & ML Rigor      │ 4-Quadrant Benchmark Chart vs. FastEmbed      │
│                                        │ (ROC-AUC), Stacking Ensemble Validation,      │
│                                        │ Feature Importance Gain & Diagnostic Curves   │
├────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 3. Vulnerability Prioritization Matrix │ Searchable, filterable 18,666 actionable flaw │
│                                        │ catalog, threat vs CVSS scatter, and formula  │
├────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 4. Enterprise Asset Inventory          │ ShopEasy 30-asset topology, department risk   │
│                                        │ breakdown, and asset vulnerability drill-down │
└────────────────────────────────────────┴───────────────────────────────────────────────┘
```

---

## 3. Detailed Workspace Guide

### Workspace 1: Executive Remediation Console (`views/overview.py`)
> **★ Recommended Starting Point for Mentor / Faculty Evaluation:** This single screen presents the complete problem, empirical solution, mathematical optimization, and operational patch output without needing to switch pages.

* **1. Industry Problem Callout Banner**:
  * Clearly states why CVSS fails (flags 20–30% of flaws as emergencies, but <1% are ever weaponized in the wild; lacks business impact and budget awareness).
* **2. Live Enterprise Sprint Simulation**:
  * **Interactive Sprint Budget Slider**: Choose between `5h` (Emergency Rapid Response), `10h` (DevOps Mini-Sprint), `20h` (Standard Sprint Allocation), or `40h` (Full Engineering Week).
  * **Real-Time Absolute Metric Cards**:
    * **CyberGuard Exact Risk Reduced**: Optimal objective value achieved via Google OR-Tools / SciPy HiGHS.
    * **Naive CVSS Greedy Risk**: Risk addressed by sorting top-down by raw CVSS score alone.
    * **Net Risk Reduction Advantage (Absolute Gain)**: Exact absolute risk units gained over the naive CVSS baseline ($\Delta\text{Risk} = \text{Exact} - \text{CVSS}$, e.g. `+16.043` at 5h, `+22.970` at 10h).
    * **Sprint Hours Used**: Hard constraint enforcement (hours consumed vs remaining capacity).
* **3. Side-by-Side Visualizations**:
  * **Chart 1 (Algorithm Comparison)**: Direct bar chart comparing CyberGuard Exact 0-1 Knapsack vs. Risk/Hour Density Heuristic vs. Naive CVSS Greedy.
  * **Chart 2 (Department Risk Reduction Allocation)**: Visual breakdown of risk reduced and hours spent across enterprise units (Payments, Logistics, Corporate, etc.).
* **4. Actionable Sprint Patch Plan Table**:
  * Displays the selected remediation fixes with CVE ID, affected asset, department, CVSS score, remediation effort, and risk reduction.
  * **📥 Export Sprint Remediation Plan (CSV)** button enables direct download for DevOps execution.
* **5. Auditable 3-Layer Decision Justification**:
  * Selecting any recommended fix displays its 3 auditable justification cards:
    * **Layer 1 (Threat Intelligence)**: EPSS score, CISA KEV status, calibrated model likelihood, and CVSS severity.
    * **Layer 2 (Business Context)**: Asset criticality, department, business impact score (1–10), and 1.5× internet exposure.
    * **Layer 3 (Optimization Rationale)**: Value-to-cost density and branch-and-bound knapsack justification.

---

### Workspace 2: Academic Benchmarks & ML Rigor (`views/performance.py`)
Presents the empirical scientific evaluation against the published state-of-the-art:

* **Tab 1: 4-Quadrant Academic Benchmark**:
  * **The 4-Quadrant Comparative Bar Chart**: Grouped bar chart comparing FastEmbed (*Fang et al., PLOS ONE 2020*) vs. CyberGuard-AI focused strictly on **ROC-AUC (0.50 to 1.00)** across all 4 research scenarios.
  * **The 4-Quadrant Matrix Table**: Direct side-by-side comparison across all 4 benchmark quadrants with absolute score differences:
    * **Quadrant 1 (Table 5)**: Curated 2009–2015 NVD (+0.0147 AUC, +0.0329 PR points).
    * **Quadrant 1 (5-Fold CV)**: Curated 2013–2018 NVD (+0.0387 AUC, +0.0646 PR points).
    * **Quadrants 2 & 3**: Real-world 2022–2024 temporal stream (+0.2546 AUC, +0.000846 PR points where FastEmbed collapses to 0.5174).
    * **Quadrant 3 (CVE-Level)**: Borda count rank aggregation (0.7790 ROC-AUC).
* **Tab 2: CyberGuard v3 Stacking Engine**:
  * Displays test metrics (0.7720 ROC-AUC, 0.7790 CVE Borda ROC-AUC) and individual base learner validation ROC-AUCs (LightGBM, XGBoost, ExtraTrees).
* **Tab 3: Feature Importance Analysis**:
  * Decision gain distribution across top 10 engineered domain features (`Age_Cohort_Pct`, `Is_RCE`, `Is_PrivEsc`, `RCE_CVSS_Interaction`).
* **Tab 4: Generated Diagnostic Curves & Plots**:
  * Interactive viewer for generated pipeline artifacts: ROC curves, PR curves, probability distributions, ranking gain diagnostics, and confusion matrices.

---

### Workspace 3: Vulnerability Prioritization Matrix (`views/prioritization.py`)
Provides SecOps teams with operational flaw triage and granular backlog inspection:

* **1. Enterprise Backlog KPIs**:
  * Real-time counters for Total Actionable Flaws (18,666), Active In Wild / CISA KEV (148), Critical Asset Instances (8,115), and Total Enterprise Risk Surface (4,875.4).
* **2. Threat Likelihood vs. CVSS Severity Scatter Plot**:
  * Directly illustrates why static CVSS fails: flaws with CVSS 9.0–10.0 span both active weaponized zero-days and benign unexploited flaws.
  * Interactive tooltips show CVE ID, affected asset, department, business impact, remediation effort, and EPSS likelihood.
* **3. Search & Advanced Filtering**:
  * Instant search across CVE IDs, asset names, and department tags.
  * Multi-select filters for Severity, Department, CISA KEV status, and CVSS range slider.
* **4. Flaw Catalog & CSV Export**:
  * Complete 18,666 actionable flaw inventory sorted by composite business risk with direct CSV export.
* **5. Flaw Detail Inspector**:
  * Inspects any selected flaw, showing:
    * Threat evidence (EPSS score, CISA KEV confirmation, calibrated probability).
    * Asset context (criticality, business impact score 1–10, 1.5× internet exposure).
    * Exact mathematical step-by-step formula breakdown:
      $$\text{Business Risk} = P(\text{Exploit}) \times \text{Business Impact} \times \text{Exposure} \times \left(\frac{\text{CVSS}}{10}\right)$$

---

### Workspace 4: Enterprise Asset Inventory (`views/assets.py`)
Explores the **ShopEasy Enterprise Archetype** (30 assets across 7 business units).

* **Asset Table**: Displays Asset ID, Asset Name, Department, Type (Server, Database, Gateway, Laptop), Criticality, Internet Exposure, Business Impact (1–10), and Environment (Production, Staging, Corporate).
* **Department Modeled-Risk Chart**: Color-coded by the number of internet-exposed instances.
* **Asset Vulnerability Drill-Down**: Interactive dropdown allowing engineers to select any asset (e.g., `AST-001 Payments Gateway`) and immediately inspect all CVEs mapped to that specific system.

---

## 4. Key Terminology & Mathematical Formulations

| Term | Formulation / Meaning | Role in CyberGuard-AI |
| :--- | :--- | :--- |
| **CVSS** | Common Vulnerability Scoring System ($0.0 - 10.0$) | Baseline technical severity metric. |
| **EPSS** | Exploit Prediction Scoring System ($0.0 - 1.0$) | Daily empirical exploitation probability signal from FIRST.org. |
| **CISA KEV** | Known Exploited Vulnerabilities Catalog | Ground truth evidence of verified exploitation in the wild. |
| **Platt Scaling** | $\hat{P}(Y=1 \mid z) = \frac{1}{1 + \exp(A z + B)}$ | Calibrates decision logits to eliminate saturation ties. |
| **Borda Rank** | Multi-snapshot rank aggregation percentile | Converts monthly temporal predictions into unified CVE ranks. |
| **Business Risk** | $P(\text{Exploit}) \times \text{Impact} \times \text{Exposure} \times \left(\frac{\text{CVSS}}{10}\right)$ | Composite enterprise objective function. |
| **0-1 Knapsack** | Exact combinatorial MILP solved via OR-Tools / HiGHS | Maximizes enterprise risk reduction under hard engineer-hour limits. |

---

## 5. Verification & Testing

To run the automated test suite verifying dashboard data loaders and risk optimization adapters:

```powershell
.\venv\Scripts\pytest tests/test_dashboard.py -v
```

All 4 test cases validate data integrity, threat intelligence joining, knapsack budget enforcement, and feature importance loading.
