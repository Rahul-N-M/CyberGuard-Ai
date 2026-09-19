# CyberGuard-AI: Notebooks Directory

This directory contains the consolidated, production-verified Jupyter notebooks for **CyberGuard-AI**, alongside the archived early exploratory versions.

---

## 1. Production Notebooks Overview

All 25+ fragmented experimental scripts from `src/ml/` have been consolidated into **3 clean, self-contained, pre-executed production notebooks**. Each notebook contains complete markdown documentation, embedded visualizations, and verified code cells.

| # | Notebook | Focus Area | Runtime | Key Output |
| :---: | :--- | :--- | :---: | :--- |
| **01** | [**`01_cyberguard_v3_production_pipeline.ipynb`**](./01_cyberguard_v3_production_pipeline.ipynb) | Production ML Pipeline | ~138s | Full stacking ensemble (LightGBM + XGBoost + ExtraTrees + Ridge Meta-Learner + Platt Calibration + Borda Ranking). Test ROC-AUC: **0.7720**, PR-AUC: **0.001223**, CVE ROC-AUC: **0.7790**. Diagnostic curves (ROC, PR, Calibration Reliability, Feature Gain). |
| **02** | [**`02_fastembed_vs_cyberguard_paper_benchmark.ipynb`**](./02_fastembed_vs_cyberguard_paper_benchmark.ipynb) | Academic Benchmark & Replication | ~21s | Rigorous 4-Quadrant comparison against Fang et al. (PLOS ONE 2020). Replicates Table 5 (2009–2015) where CyberGuard achieves **0.9064 vs 0.8917**, evaluates 2013–2018 NVD (**0.8549 vs 0.8162**), and demonstrates FastEmbed's collapse on 2022–2024 temporal streams (**0.5174**). |
| **03** | [**`03_enterprise_knapsack_optimization.ipynb`**](./03_enterprise_knapsack_optimization.ipynb) | Enterprise Remediation Optimization | ~14s | ShopEasy 30-asset enterprise integration. Fuses exploit probability with business impact, runs Google OR-Tools / SciPy HiGHS 0-1 Knapsack across 5h, 10h, 20h, 40h sprint budgets (+31.8% to +71.8% higher risk reduction than CVSS-Greedy), and generates actionable patch lists. |

---

## 2. Archived Versions (`old_versions/`)

Early exploratory and preliminary notebooks have been archived to [**`old_versions/`**](./old_versions/):

- `old_02_missing_value_analysis.ipynb` — Initial missing value diagnostics.
- `old_03_baseline_models.ipynb` — Early non-temporal baseline heuristics.
- `old_04_target_analysis.ipynb` — Preliminary label distribution analysis.
- `old_05_lightgbm_training.ipynb` — Prototype v1 LightGBM notebook.
- `old_06_model_evaluation.ipynb` — Preliminary ranking evaluation checks.

---

## 3. How to Run

All notebooks are designed to run from the repository root or the `notebooks/` directory.

### Prerequisites
Ensure the environment dependencies are installed:
```bash
pip install -r requirements.txt
pip install nbformat nbclient matplotlib scipy scikit-learn lightgbm xgboost ortools
```

### Execution via CLI
You can execute and verify any notebook headlessly using `nbclient`:
```bash
python -m nbclient notebooks/01_cyberguard_v3_production_pipeline.ipynb
python -m nbclient notebooks/02_fastembed_vs_cyberguard_paper_benchmark.ipynb
python -m nbclient notebooks/03_enterprise_knapsack_optimization.ipynb
```
Or open via Jupyter Lab / Notebook:
```bash
jupyter lab notebooks/
```
