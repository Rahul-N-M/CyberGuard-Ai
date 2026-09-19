# FastEmbed Baseline Report (Fang et al. 2020 Architecture on CyberGuard Data)

## Setup

| Parameter | Value |
|---|---|
| **Paper** | Fang et al., PLOS ONE 2020, DOI:10.1371/journal.pone.0228439 |
| **Architecture** | fastText embeddings (dim=100) + LightGBM |
| **FastText trained on** | Train descriptions ONLY (no val/test leakage) |
| **Total features** | 114 (100 embed + CVSS structural) |
| **Imbalance handling** | scale_pos_weight = 32.2064 (from train only) |
| **Threshold selection** | Optimal F1 on validation = 0.8176 (applied once to test) |
| **LightGBM training time** | 4.76s |
| **Best iteration (early stopping)** | 2 |

## Performance at Default Threshold (0.50)

| Metric | Train (2022) | Validation (2023) | Test (2024) |
|---|---|---|---|
| **ROC-AUC** | 0.874703 | 0.634119 | 0.517135 |
| **PR-AUC** | 0.503478 | 0.014027 | 0.000377 |
| **F1 (minority)** | 0.477178 | 0.041783 | 0.002308 |
| **Precision** | 0.332370 | 0.023232 | 0.001203 |
| **Recall** | 0.845588 | 0.207373 | 0.028226 |
| **Brier Score** | 0.00177770 | 0.00423909 | 0.00666140 |
| **Actual Positives** | 136 | 217 | 248 |

## Performance at Val-Optimal Threshold (0.8176)

| Metric | Test (2024) |
|---|---|
| **ROC-AUC** | 0.517135 |
| **PR-AUC** | 0.000377 |
| **F1 (minority)** | 0.002312 |
| **Precision** | 0.001205 |
| **Recall** | 0.028226 |

## Top-K Ranking Performance (Test Set, 2024)

| K | Precision@K | Recall@K | Positives Found |
|---|---|---|---|
| 10 | 0.0000 | 0.0000 | 0 |
| 25 | 0.0000 | 0.0000 | 0 |
| 50 | 0.0000 | 0.0000 | 0 |
| 100 | 0.0000 | 0.0000 | 0 |
| 200 | 0.0000 | 0.0000 | 0 |
| 500 | 0.0000 | 0.0000 | 0 |
| 1000 | 0.0010 | 0.0040 | 1 |

## Fairness Guarantees

1. **No temporal leakage**: fastText trained on train text only; medians/OHE fit on train only.
2. **No threshold snooping**: threshold selected on validation, applied once to test.
3. **Identical splits**: same `train.csv`, `validation.csv`, `test.csv` as all CyberGuard models.
4. **Same target**: `Target_KEV_180d`.
5. **Reported both thresholds**: default (0.5) and optimal-val for transparency.
