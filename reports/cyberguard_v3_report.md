# CyberGuard-Ensemble v3 Report

## Architecture (Novel Contributions)

### Novel Contribution 1: Temporal-Aware Stacking Ensemble
Base learners: **lgbm, xgb, et** → Meta-learner: Ridge Logistic Regression

| Base Learner | Val ROC-AUC |
|---|---|
| lgbm | 0.7484 |
| xgb | 0.7605 |
| et | 0.7709 |
| **Stacked ensemble** | **0.7503** |

### Novel Contribution 2: Platt Calibration (Temporal Drift Correction)
Fitted on validation (2023) to correct 2022→2024 class-prior shift.

### Novel Contribution 3: CVE-Level Borda Count Aggregation
Collapses multiple monthly snapshots into one per-CVE operational risk score.
- **CVE-level ROC-AUC (test)**: 0.7790
- **CVE-level PR-AUC (test)**: 0.003575

---

## Row-Level Performance (Calibrated Probabilities)

### Default Threshold (0.50)

| Metric | Train (2022) | Validation (2023) | Test (2024) |
|---|---|---|---|
| **ROC-AUC** | 0.999783 | 0.750282 | 0.772004 |
| **PR-AUC** | 0.738026 | 0.002191 | 0.001223 |
| **F1 (minority)** | 0.000000 | 0.000000 | 0.000000 |
| **Precision** | 0.000000 | 0.000000 | 0.000000 |
| **Recall** | 0.000000 | 0.000000 | 0.000000 |
| **Brier Score** | 0.00080513 | 0.00044631 | 0.00027326 |
| **Actual Positives** | 136 | 217 | 248 |

### Val-Optimal Threshold (0.0122)

| Metric | Test (2024) |
|---|---|
| **ROC-AUC** | 0.772004 |
| **F1 (minority)** | 0.004662 |
| **Precision** | 0.003279 |
| **Recall** | 0.008065 |

### Top-K Ranking (Test 2024)

| K | Precision@K | Recall@K | Positives Found |
|---|---|---|---|
| 10 | 0.0000 | 0.0000 | 0 |
| 25 | 0.0000 | 0.0000 | 0 |
| 50 | 0.0000 | 0.0000 | 0 |
| 100 | 0.0000 | 0.0000 | 0 |
| 200 | 0.0000 | 0.0000 | 0 |
| 500 | 0.0000 | 0.0000 | 0 |
| 1000 | 0.0030 | 0.0121 | 3 |
