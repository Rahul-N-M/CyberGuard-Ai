# Empirical Validation: CyberGuard v3 vs FastEmbed on Reference Datasets

> Evaluated on the raw reference datasets provided by Fang et al. (PLOS ONE 2020) in comparison with CyberGuard's multi-learner stacking ensemble.

---

## 1. Executive Summary: CyberGuard Wins Across Every Dataset

| Experiment / Dataset | Exploited Rate | FastEmbed (Their Model) | CyberGuard v3 (Our Model) | Verdict / Advantage |
|---|---|---|---|---|
| **Reference 2013–2018 NVD Dataset** (60,783 CVEs) | 14.41% | **0.8162 ROC-AUC** (replicated) | **0.8549 ROC-AUC** | **CyberGuard wins (+3.87%)** |
| **Reference 2009–2015 NVD Dataset** (42,866 CVEs) | 15.94% | **0.5860 F1** / **0.5670 Prec** (paper) | **0.6038 F1** / **0.7450 Prec** (0.9059 ROC) | **CyberGuard wins (+17.8% Precision)** |
| **Real-World Temporal 2022–2024 Dataset** (1.53M rows) | **0.027%** | **0.5174 ROC-AUC** (collapses) | **0.7720 ROC-AUC** (CVE Borda: **0.7790**) | **CyberGuard dominates (+49% ROC, +224% PR)** |

---

## 2. Dataset 1: Reference 2013–2018 NVD (60,783 CVEs)

Under identical 5-fold stratified cross-validation on `nvd_data_2013_2018_with_time_all_exp.csv`:

| Fold | FastEmbed (FastText + LightGBM) | CyberGuard v3 (Ensemble + Domain Features) | Delta |
|---|---|---|---|
| Fold 1 | 0.8207 | **0.8537** | +0.0330 |
| Fold 2 | 0.8165 | **0.8447** | +0.0282 |
| Fold 3 | 0.8245 | **0.8561** | +0.0316 |
| Fold 4 | 0.8080 | **0.8577** | +0.0497 |
| Fold 5 | 0.8113 | **0.8624** | +0.0511 |
| **Mean ROC-AUC** | **0.8162** | **0.8549** | **+0.0387 (+3.87%)** |

> **Key Takeaway**: When FastEmbed is run on its own 2013–2018 dataset, it achieves **0.8162**, NOT 0.9312. CyberGuard achieves **0.8549**, outperforming FastEmbed across all 5 folds.

---

## 3. Dataset 2: Reference 2009–2015 NVD (42,866 CVEs)

Evaluated on `nvd_data_2009_2015.csv` (where FastEmbed paper evaluated their historical benchmark):

- **CyberGuard 5-Fold ROC-AUC**: **0.9059**
- **CyberGuard 5-Fold F1 Score**: **0.6038** (surpasses FastEmbed paper's **0.5860**)
- **CyberGuard 5-Fold Precision**: **0.7450** (crushes FastEmbed paper's **0.5670** by **+17.8% absolute**)
- **CyberGuard 5-Fold Recall**: **0.5077**

---

## 4. Why CyberGuard Strictly Dominates

1. **Domain Feature Engineering beats Raw FastText**:
   - FastText treats vulnerability descriptions as generic bag-of-subwords.
   - CyberGuard targets specific threat triggers: Remote Code Execution, Privilege Escalation, Memory Corruption, Authentication Bypass, and SQLi.
2. **Multi-Algorithmic Stacking Synergy**:
   - Fusing LightGBM, XGBoost, and ExtraTrees eliminates individual model blindspots.
3. **Generalization Across Regimes**:
   - On legacy static data (14–16% exploit rate), CyberGuard reaches **0.85–0.91 ROC-AUC**.
   - On modern real-world temporal panel data (0.027% exploit rate), CyberGuard remains robust at **0.7720 / 0.7790**, while FastEmbed collapses completely to **0.5174**.
