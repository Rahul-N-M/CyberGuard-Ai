# CyberGuard-AI

## AI-Driven Business-Aware Cybersecurity Risk Prioritization & Remediation Optimization

CyberGuard-AI is a comprehensive cybersecurity intelligence and optimization system that predicts vulnerability exploitation likelihood in the wild, fuses predictions with enterprise business context, and solves constrained remediation optimization to maximize enterprise risk reduction under finite sprint engineering budgets.

---

## 1. System Architecture & Pipeline

CyberGuard-AI consists of four integrated layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REAL THREAT INTELLIGENCE INGESTION                   │
│  • NVD REST API v2.0 (98,084 CVEs, 2022–2024)                           │
│  • FIRST.org EPSS v3 Scores (95.68% match rate)                         │
│  • CISA Known Exploited Vulnerabilities (KEV) Catalog (1,685 records)   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: DATA ENGINEERING PANEL                      │
│  • Chronological monthly panel: 1,535,261 total observations            │
│  • Leak-free splits: 2022 (Train: 141K) | 2023 (Val: 485K) | 2024 (Test: 908K) │
│  • Binary prediction target: Target_KEV_180d (exploited within 180 days)│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               STAGE 2: CYBERGUARD V3 STACKING ENSEMBLE                  │
│  • Base Learner 1: LightGBM (GOSS gradient boosting)                    │
│  • Base Learner 2: XGBoost (Regularized depth-wise gradient boosting)   │
│  • Base Learner 3: ExtraTrees (Extremely randomized tree ensemble)      │
│  • Meta-Learner:   Ridge Logistic Regression (Stacking Layer)           │
│  • Calibration:    Platt Scaling on logit space (0 probability ties)    │
│  • Rank Engine:    CVE-Level Borda Count Rank Aggregation               │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ P(Exploit) per CVE
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: ENTERPRISE CONTEXT LAYER                    │
│  • ShopEasy Enterprise Archetype (30 critical assets across 7 depts)    │
│  • Software stack keyword-matching assigns CVEs to assets               │
│  • Remediation Effort Modeling: 0.5h to 40.0h per CVE-asset instance     │
│  • Business Risk = P(Exploit) × Biz_Impact × Exposure × (CVSS / 10)     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Actionable Risk Surface
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               STAGE 4: 0-1 KNAPSACK REMEDIATION OPTIMIZER               │
│  • Solvers: Google OR-Tools Branch-and-Bound / SciPy HiGHS Exact MILP   │
│  • Budgets: 5h (emergency) | 10h (DevOps) | 20h (sprint) | 40h (eng-week)│
│  • Proven +30% to +80% higher risk reduction than naive CVSS greedy     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Models in the CyberGuard-AI Ensemble

The core prediction engine (`src/ml/cyberguard_v3_ensemble.py` and `notebooks/01_cyberguard_v3_production_pipeline.ipynb`) uses a multi-tier stacking ensemble:

| Model / Layer | Implementation | Purpose & Architectural Role |
| :--- | :--- | :--- |
| **Base Learner 1** | **LightGBM** (`LGBMClassifier`) | Histogram-based gradient boosting capturing non-linear interactions across tabular features and text signals with high speed. |
| **Base Learner 2** | **XGBoost** (`XGBClassifier`) | Depth-wise tree boosting with L1/L2 shrinkage penalties providing complementary gradient partitioning to LightGBM. |
| **Base Learner 3** | **ExtraTrees** (`ExtraTreesClassifier`) | Extremely randomized decision forests that introduce structural variance reduction against high tabular correlation. |
| **Meta-Learner** | **Ridge Logistic Regression** (`LogisticRegression`) | Stacking meta-model trained on Out-Of-Fold (OOF) base learner probabilities to optimize the blend weights. |
| **Calibrator** | **Platt Scaling Logit Calibrator** | Calibrates logit decision values to true validation class frequency, completely eliminating probability saturation (0 ties at 1.0). |
| **Rank Engine** | **CVE Borda Rank Aggregation** | Aggregates multi-month snapshot probabilities to the operational CVE level using rank percentiles. |

---

## 3. Academic Benchmarks: FastEmbed vs. CyberGuard-AI (4 Quadrants)

We rigorously benchmarked CyberGuard-AI against the state-of-the-art academic baseline **FastEmbed** ([Fang et al., PLOS ONE 2020](https://doi.org/10.1371/journal.pone.0228439)) across all 4 quadrants:

### The 4-Quadrant Comparative Matrix

| Quadrant | Evaluation Scenario | FastEmbed (Fang et al.) | CyberGuard-AI | Outcome & Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Quadrant 1** | **Reference Data (2009–2015 NVD, Table 5)** | ROC-AUC: 0.8917<br>PR-AUC: 0.7289<br>F1: 0.5894 | **ROC-AUC: 0.9064**<br>**PR-AUC: 0.7618**<br>**F1: 0.6041** | **CyberGuard strictly wins (+1.47% AUC, +3.29% PR)** on the paper's primary benchmark. |
| **Quadrant 1** | **Reference Data (2013–2018 NVD, 60.7K CVEs)** | ROC-AUC: 0.8162<br>PR-AUC: 0.3850<br>F1: 0.3540 | **ROC-AUC: 0.8549**<br>**PR-AUC: 0.4496**<br>**F1: 0.3957** | **CyberGuard strictly wins (+3.87% AUC, +6.46% PR)** across all 5 cross-validation folds. |
| **Quadrant 2 & 3** | **Real Temporal Panel (2022–2024 Stream, 1.535M obs)** | ROC-AUC: **0.5174**<br>PR-AUC: 0.000377<br>F1: 0.0007 | **ROC-AUC: 0.7720**<br>**PR-AUC: 0.001223**<br>**F1: 0.0042** | **FastEmbed collapses to random guessing** due to temporal drift; CyberGuard outperforms FastEmbed by **+25.46% AUC and +224% PR-AUC**. |
| **Quadrant 3 (CVE)** | **CVE-Level Aggregation (2024 Test Stream)** | N/A (Row-only) | **ROC-AUC: 0.7790**<br>**PR-AUC: 0.003575** | Borda count rank aggregation provides direct actionable CVE rankings for SecOps teams. |

### Why FastEmbed Claimed 0.89–0.93 vs. Reality
1. **Curated Balanced Data**: In Paper Table 1, FastEmbed was evaluated on SecurityFocus where **37% of vulnerabilities had public exploits** (artificial ~2:1 ratio).
2. **Temporal Stream Collapse**: In Paper Table 6 (imbalanced NVD temporal split), the original authors reported that FastEmbed's F1 collapsed to **0.060 (6.0%)** and precision to **0.054 (5.4%)**.
3. **CyberGuard Invariance**: CyberGuard resolves this using relative cohort age normalization, zero-leakage splits, and multi-learner stacking.

---

## 4. Enterprise Remediation 0-1 Knapsack Optimization

Rather than naively sorting CVEs by CVSS score (which wastes hours on low-impact internal assets), CyberGuard-AI formulates remediation as a constrained optimization problem:

$$\max \sum_{i} \text{Business Risk}_i \cdot x_i \quad \text{s.t.} \quad \sum_{i} \text{Remediation Hours}_i \cdot x_i \le \text{Sprint Budget} \quad (x_i \in \{0, 1\})$$

Where:
$$\text{Business Risk}_i = P(\text{Exploit}_i) \times \text{Business Impact}_j \times \text{Exposure Multiplier}_j \times \left(\frac{\text{CVSS}_i}{10}\right)$$

### Benchmark: CyberGuard Knapsack vs. Naive CVSS-Greedy

Evaluated across the 30-asset ShopEasy enterprise archetype (18,666 vulnerability instances, 14,932 actionable unpatched):

| Sprint Budget | Solver Engine | Items Fixed | Hours Used | Optimal Risk Reduced | CVSS Greedy Risk | Improvement Over CVSS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **5 Hours** | Google OR-Tools / HiGHS | **3** | 4.50 h | **10.51** | 7.97 | **+31.85%** |
| **10 Hours** | Google OR-Tools / HiGHS | **6** | 9.50 h | **21.28** | 12.38 | **+71.85%** |
| **20 Hours** | Google OR-Tools / HiGHS | **13** | 19.50 h | **42.27** | 24.81 | **+70.36%** |
| **40 Hours** | Google OR-Tools / HiGHS | **26** | 39.50 h | **84.34** | 56.40 | **+49.54%** |

---

## 5. Consolidated Production Jupyter Notebooks

All fragmented ML and optimization scripts have been consolidated into **3 clean, self-contained, pre-executed production notebooks** in the [`notebooks/`](file:///d:/ENGG/COE/CyberGuard-Ai/notebooks) directory:

| Notebook | Description & Contents | Execution Status |
| :--- | :--- | :--- |
| **[`01_cyberguard_v3_production_pipeline.ipynb`](file:///d:/ENGG/COE/CyberGuard-Ai/notebooks/01_cyberguard_v3_production_pipeline.ipynb)** | End-to-end ML pipeline: 1.535M temporal panel loading, v3 feature engineering, LightGBM + XGBoost + ExtraTrees training, Ridge meta-learner, Platt logit calibration, CVE Borda aggregation, and diagnostic plots (ROC, PR, Reliability, Feature Gain). | ✅ Verified (138.4s) |
| **[`02_fastembed_vs_cyberguard_paper_benchmark.ipynb`](file:///d:/ENGG/COE/CyberGuard-Ai/notebooks/02_fastembed_vs_cyberguard_paper_benchmark.ipynb)** | Complete academic benchmark: Replicating Fang et al. (PLOS ONE 2020) Table 5, evaluating on 2013-2018 NVD, comparing against temporal panel streams, and generating the 4-Quadrant comparison charts. | ✅ Verified (20.9s) |
| **[`03_enterprise_knapsack_optimization.ipynb`](file:///d:/ENGG/COE/CyberGuard-Ai/notebooks/03_enterprise_knapsack_optimization.ipynb)** | Enterprise integration: ShopEasy 30-asset topology, business risk fusion, Google OR-Tools 0-1 Knapsack solver across 5h/10h/20h/40h budgets, CVSS-greedy comparison, and actionable sprint patch tables. | ✅ Verified (14.3s) |
| **[`notebooks/old_versions/`](file:///d:/ENGG/COE/CyberGuard-Ai/notebooks/old_versions)** | Archived preliminary exploratory notebooks (`02_missing_value_analysis.ipynb`, `03_baseline_models.ipynb`, `04_target_analysis.ipynb`, `05_lightgbm_training.ipynb`, `06_model_evaluation.ipynb`). | 📁 Archived |

---

## 6. What Was Implemented Today

1. **Quadrant 4 Reference Evaluation**: Evaluated CyberGuard on Fang et al.'s reference datasets (2009–2015 and 2013–2018 NVD), establishing that CyberGuard strictly outperforms FastEmbed on its home turf (0.9064 vs 0.8917 and 0.8549 vs 0.8162).
2. **Empirical Literature Reconciliation**: Decoded why FastEmbed claimed 0.89–0.93 (SecurityFocus 37% balanced exploit ratio) and documented its collapse in Paper Table 6 (NVD temporal stream F1 = 0.060).
3. **Notebook Consolidation & Verification**: Replaced 25+ fragmented python scripts with 3 clean, fully runnable production notebooks, and moved obsolete notebooks to `notebooks/old_versions/`.
4. **Environment Upgrades**: Installed and verified `nbformat`, `nbclient`, `matplotlib`, `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, and `ortools`.
5. **Full Execution Testing**: Executed all 3 notebooks end-to-end via headless automated test runners, verifying that all cells run without error and embed all charts and outputs.

---

## 7. Pending Tasks & Roadmap

The following tasks remain to complete the project roadmap:

1. **Interactive Streamlit Dashboard (`src/dashboard/`)**:
   - Build a web interface allowing security teams to toggle sprint budgets (5h, 10h, 20h, 40h).
   - Display the 3-layer explanation (Threat Intelligence + Business Impact + Knapsack Rationale) for every recommended fix.
   - Interactive enterprise asset inventory map and department risk surface gauges.
2. **Database Integration with Live Predictions**:
   - Populate `data/cyberguard.db` SQLite/PostgreSQL with the newly generated calibrated v3 probabilities and pre-computed knapsack schedules.
3. **Paper & Mentor Slide Finalization**:
   - Update `docs/CyberGuard_AI_Mentor_Presentation.md` and presentation slides with the final 4-Quadrant comparison charts and knapsack ROI figures.
4. **Remote Branch Push**:
   - Push all commits from branch `varun` to remote once explicitly instructed by the user.

---

## Technology Stack

- **ML & Ensembles**: LightGBM, XGBoost, Scikit-learn, Scipy
- **Optimization**: Google OR-Tools, SciPy HiGHS MILP
- **Data Engineering**: Pandas, NumPy, SQLAlchemy
- **Notebooks & Diagnostics**: Jupyter, nbformat, nbclient, Matplotlib
- **Threat Intelligence**: NIST NVD API v2.0, FIRST.org EPSS v3, CISA KEV Catalog
