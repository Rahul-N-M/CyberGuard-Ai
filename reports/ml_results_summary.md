# CyberGuard-Ai: Machine Learning Temporal Model Results Summary

> **Status**: Validated Temporal LightGBM Baseline  
> **Evaluation Mode**: Post-Training Evaluation on Existing Models and Prediction Outputs (No Retraining)  
> **Source of Truth**: `reports/lightgbm_training_report.md`, `reports/lightgbm_threshold_analysis_corrected.md`, `reports/lightgbm_ranking_diagnostic.md`

---

## 1. Dataset & Pipeline Summary

The machine learning pipeline predicts future real-world vulnerability exploitation by formulating a temporally clean panel dataset from public vulnerability intelligence and chronological observation snapshots.

```
NVD CVE Records
      │
      ▼
EPSS Exploitation Probability Matching (95.68% coverage)
      │
      ▼
Temporal Observation Construction (Monthly Snapshots)
      │
      ▼
180-Day Future CISA KEV Target Definition (Target_KEV_180d)
      │
      ▼
Chronological Train / Validation / Test Split (2022 / 2023 / 2024)
      │
      ▼
Feature Preprocessing & Encoding (13 features)
      │
      ▼
LightGBM Temporal Baseline Model (scale_pos_weight = 1037.25)
      │
      ▼
Probability Prediction & Calibration Diagnostics
      │
      ▼
Threshold & Top-K Ranking Evaluation
      │
      ▼
Enterprise Asset Fusion & Downstream Google OR-Tools Knapsack Optimization
```

### Dataset Scale
- **Total Unique CVEs**: 98,084 CVEs (spanning 2022–2024)
- **EPSS Matched Records**: 93,844 CVEs (**95.68%** match rate)
- **Temporal Observations**: **1,535,261** panel rows
- **Total Positive Observations**: **601** (`Target_KEV_180d = 1`)
- **Total Unique Exploited CVEs**: **215**

---

## 2. Target Definition & Temporal Framing

- **Target Variable**: `Target_KEV_180d`
- **Definition**: Binary indicator set to `1` if a CVE appears on the CISA Known Exploited Vulnerabilities (KEV) catalog within 180 days following observation date $T$ ($\text{Obs\_Date} < \text{KEV\_Date} \le \text{Obs\_Date} + 180\,\text{days}$), and `0` otherwise.
- **Temporal Integrity**: The observation date strictly precedes the target horizon. Target leakage verification confirmed 0 instances of future date contamination and no direct inclusion of KEV status, EPSS score, or future remediation data in the model feature set.

---

## 3. Chronological Train / Validation / Test Split

To simulate realistic production conditions and prevent data leakage across time, data was split strictly along annual chronological boundaries:

| Split | Observation Period | Total Rows | Actual Positives | Positive Rate | Unique CVEs | Unique Positive CVEs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 2022 (Jan 1 – Dec 1) | 141,202 | 136 | 0.0963% | 23,531 | 53 |
| **Validation** | 2023 (Jan 1 – Dec 1) | 485,969 | 217 | 0.0447% | 54,694 | 72 |
| **Test** | 2024 (Jan 1 – Dec 1) | 908,090 | 248 | 0.0273% | 94,636 | 90 |
| **Total** | **2022–2024** | **1,535,261** | **601** | **0.0391%** | **94,636** | **215** |

---

## 4. Model Configuration & Feature Architecture

- **Model Class**: LightGBM Classifier (`LGBMClassifier`)
- **Trained Artifacts**: `models/lightgbm_temporal_baseline.pkl`, `models/lightgbm_temporal_baseline.txt`
- **Trained Iterations**: 100 boosting trees, early stopping selected **Best Iteration = 8**
- **Imbalance Handling**: `scale_pos_weight = 1037.25` (calculated from training positive imbalance)
- **Input Features (13 total)**:
  1. `Vulnerability_Age_Days` (Numerical)
  2. `CVSS_Score` (Numerical, 0.0–10.0)
  3. `Severity_Encoded` (Ordinal: 1=Low, 2=Medium, 3=High, 4=Critical)
  4. `CVSS_Version_2.0` (One-hot binary)
  5. `CVSS_Version_3.0` (One-hot binary)
  6. `CVSS_Version_3.1` (One-hot binary)
  7. `CVSS_Version_4.0` (One-hot binary)
  8. `CVSS_Version_Missing` (One-hot binary)
  9. `Severity_CRITICAL` (One-hot binary)
  10. `Severity_HIGH` (One-hot binary)
  11. `Severity_LOW` (One-hot binary)
  12. `Severity_MEDIUM` (One-hot binary)
  13. `Severity_Missing` (One-hot binary)

### Feature Importance (Top Features by Gain)

| Rank | Feature | Splits | Importance Gain | Gain Share |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `Vulnerability_Age_Days` | 102 | 152,818,336,456.93 | 77.4% |
| 2 | `CVSS_Score` | 95 | 28,974,267,528.18 | 14.7% |
| 3 | `CVSS_Version_3.0` | 1 | 6,613,179,904.00 | 3.4% |
| 4 | `CVSS_Version_3.1` | 6 | 5,040,097,729.00 | 2.6% |
| 5 | `Severity_Encoded` | 10 | 3,682,606,980.40 | 1.9% |
| 6 | `CVSS_Version_Missing` | 1 | 296,305,984.00 | 0.1% |
| 7 | `Severity_HIGH` | 1 | 4,534,350.00 | <0.01% |

---

## 5. Model Validation & Test Results

### Discrimination Performance

| Evaluation Metric | Training (2022) | Validation (2023) | Test (2024) |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.689193** | **0.670827** | **0.629073** |
| **PR-AUC / Average Precision** | **0.001922** | **0.001115** | **0.000863** |
| **Precision (threshold = 0.50)** | 0.001592 | 0.001077 | 0.000905 |
| **Recall (threshold = 0.50)** | 0.845588 | 0.626728 | 0.548387 |
| **F1-Score (threshold = 0.50)** | 0.003178 | 0.002150 | 0.001808 |

---

## 6. Threshold Analysis (2024 Test Set)

Operating thresholds evaluated on the 2024 Test Set ($N = 908,090$):

| Threshold | TP | FP | TN | FN | Precision | Recall | F1 | FPR | Flagged Volume |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.95** | **109** | **81,851** | **825,991** | **139** | **0.001330** | **43.9516%** | **0.002652** | **9.02%** | **81,960** |
| 0.90 | 110 | 83,333 | 824,509 | 138 | 0.001318 | 44.3548% | 0.002629 | 9.18% | 83,443 |
| 0.80 | 118 | 106,600 | 801,242 | 130 | 0.001106 | 47.5806% | 0.002206 | 11.74% | 106,718 |
| 0.50 | 136 | 150,078 | 757,764 | 112 | 0.000905 | 54.8387% | 0.001808 | 16.53% | 150,214 |
| 0.25 | 139 | 172,398 | 735,444 | 109 | 0.000806 | 56.0484% | 0.001609 | 18.99% | 172,537 |
| 0.10 | 141 | 202,471 | 705,371 | 107 | 0.000696 | 56.8548% | 0.001390 | 22.30% | 202,612 |

---

## 7. Corrected Top-K Ranking Performance

> **Critical Correction**: The 43.95% recall figure is derived from decision threshold `0.95` (which selects ~81,960 observations) and **does NOT represent Top-500 recall**.

### Corrected Top-K Evaluation (2024 Test Set)

| K Cutoff | TP in Top-K | Precision@K | Recall@K | Total Positives |
| :--- | :--- | :--- | :--- | :--- |
| **Top-50** | 1 | 0.020000 | 0.004032 (0.40%) | 248 |
| **Top-100** | 1 | 0.010000 | 0.004032 (0.40%) | 248 |
| **Top-200** | 1 | 0.005000 | 0.004032 (0.40%) | 248 |
| **Top-500** | **1** | **0.002000** | **0.004032 (0.40%)** | **248** |
| **Top-1000** | 1 | 0.001000 | 0.004032 (0.40%) | 248 |

---

## 8. Probability Saturation Diagnostic

A ranking diagnostic revealed significant probability saturation:
- **79,693 test observations (8.78% of the entire 2024 test set)** received a hard predicted probability of `1.0000`.
- This creates enormous tie groups, rendering standard unweighted sorting ineffective for small Top-K cutoffs ($K \le 1000$).
- **Characteristics of the Saturation Group (`prob = 1.0`)**:
  - Mean `CVSS_Score`: **8.66** (vs 6.66 for `prob < 1.0`)
  - Mean `Vulnerability_Age_Days`: **110.7 days** (vs 448.2 days for `prob < 1.0`)
  - Positive capture: 109 positives (43.95% of test positives) reside inside this 79,693-observation tie group.
- **Root Cause**: The model relies predominantly on `Vulnerability_Age_Days` (77.4% gain) and `CVSS_Score` (14.7% gain) as cohort proxies. CVEs with high CVSS within specific age windows are pushed to the same saturated terminal leaf nodes.

---

## 9. Baseline Status & Current Limitations

This model is a **Temporal LightGBM Baseline**, **NOT a production-ready scoring system**.

### Limitations
1. **Saturation & Ties**: Large clusters of tied probabilities at 1.0 limit standalone Top-K prioritization.
2. **Feature Depth**: Limited to technical CVSS and vulnerability age without granular vulnerability characteristics (e.g. CWE, CVSS vector attack metrics, exploit availability feeds).
3. **High False-Positive Volume**: At threshold 0.95, capturing 43.95% recall requires reviewing 81,851 false positives, which necessitates downstream filtering via enterprise asset context and mathematical budget optimization.

---

## 10. Pending Work

The following items are directly supported by the validated findings and pending implementation:

1. **Vulnerability Age Calibration**: Investigate and resolve the 2,993 stored-vs-computed vulnerability-age mismatches.
2. **NVD CVSS Vector Enrichment**: Extract and incorporate granular vector components:
   - Attack Vector (AV: Network, Adjacent, Local, Physical)
   - Attack Complexity (AC: Low, High)
   - Privileges Required (PR: None, Low, High)
   - Scope (S: Unchanged, Changed)
   - CWE classification taxonomy
3. **Relative & Percentile Age Features**: Replace or supplement raw age days with percentile / relative age to prevent cohort-proxy leaf saturation.
4. **Iterative Model Retraining**: Train and compare improved temporal LightGBM models against this baseline.
5. **Probability Calibration**: Apply isotonic regression or Platt scaling and evaluate whether post-calibration ranking improves.
6. **Top-K Discrimination Enhancement**: Refine sorting criteria within tied leaf clusters.
7. **Enterprise Asset Fusion**: Integrate ShopEasy asset context (criticality, internet exposure, business impact score) with vulnerability exploit probabilities.
8. **Resource-Constrained Optimization**: Implement and validate Google OR-Tools 0-1 Knapsack solver across engineering budgets (5h, 10h, 20h, 40h).
9. **Temporal Generalization Validation**: Conduct multi-year rolling cross-validation.
10. **Operational Threshold Formulation**: Establish final operating threshold aligned with enterprise SLA capacity.
