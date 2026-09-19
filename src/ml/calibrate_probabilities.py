"""
src/ml/calibrate_probabilities.py
Probability calibration module for CyberGuard-AI temporal predictions.
Fits Platt Scaling (Sigmoid) and Isotonic Regression on the Validation set (2023)
and evaluates on the untouched Test set (2024).

Analyses:
  - Brier Score Loss before & after calibration
  - Expected Calibration Error (ECE) with 10 bins
  - Impact on Top-K Ranking & Spearman rank correlation (empirical check of whether ranking changes)
  - Saves outputs/calibrated_predictions_test.csv and reports/probability_calibration_report.md
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def compute_topk_metrics(y_true: np.ndarray, y_prob: np.ndarray, k_values=[10, 25, 50, 100, 200, 500, 1000]):
    """Computes precision and recall at K."""
    sorted_indices = np.argsort(y_prob)[::-1]
    total_positives = int((y_true == 1).sum())
    results = {}
    for k in k_values:
        k_clamped = min(k, len(y_true))
        top_k_indices = sorted_indices[:k_clamped]
        top_k_positives = int((y_true[top_k_indices] == 1).sum())
        p_at_k = float(top_k_positives / k_clamped) if k_clamped > 0 else 0.0
        r_at_k = float(top_k_positives / total_positives) if total_positives > 0 else 0.0
        results[k] = {"precision": round(p_at_k, 6), "recall": round(r_at_k, 6), "positives": top_k_positives}
    return results


def run_calibration():
    print("=" * 70)
    print("  CYBERGUARD-AI: PROBABILITY CALIBRATION ANALYSIS (PLATT VS ISOTONIC)")
    print("=" * 70)
    
    val_path = "outputs/improved_predictions_validation.csv"
    test_path = "outputs/improved_predictions_test.csv"
    
    print("[Loading] Loading validation and test predictions...")
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    print(f"  Validation rows: {len(val_df):,d} | Test rows: {len(test_df):,d}")
    
    y_val = val_df["target"].values
    prob_val = val_df["predicted_prob"].values
    
    y_test = test_df["target"].values
    prob_test = test_df["predicted_prob"].values
    
    # 1. Platt Scaling (Logistic Regression on logit / probability)
    print("\n[Fitting] Fitting Platt Scaling (Logistic Regression) on Validation set...")
    eps = 1e-7
    clipped_val_prob = np.clip(prob_val, eps, 1 - eps)
    logit_val = np.log(clipped_val_prob / (1 - clipped_val_prob)).reshape(-1, 1)
    
    platt_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    platt_model.fit(logit_val, y_val)
    
    clipped_test_prob = np.clip(prob_test, eps, 1 - eps)
    logit_test = np.log(clipped_test_prob / (1 - clipped_test_prob)).reshape(-1, 1)
    prob_val_platt = platt_model.predict_proba(logit_val)[:, 1]
    prob_test_platt = platt_model.predict_proba(logit_test)[:, 1]
    
    # 2. Isotonic Regression
    print("[Fitting] Fitting Isotonic Regression on Validation set...")
    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(prob_val, y_val)
    
    prob_val_iso = iso_model.predict(prob_val)
    prob_test_iso = iso_model.predict(prob_test)
    
    # 3. Calculate Calibration Metrics (Brier, ECE, AUC)
    models = {
        "Uncalibrated": (prob_val, prob_test),
        "Platt Scaling": (prob_val_platt, prob_test_platt),
        "Isotonic Regression": (prob_val_iso, prob_test_iso)
    }
    
    results = {}
    print("\n" + "=" * 70)
    print("  CALIBRATION METRICS COMPARISON:")
    print(f"  {'Method':<20} | {'Val Brier':<11} | {'Val ECE':<10} | {'Test Brier':<11} | {'Test ECE':<10} | {'Test ROC-AUC':<12}")
    print("  " + "-" * 82)
    
    for name, (p_val, p_te) in models.items():
        brier_val = brier_score_loss(y_val, p_val)
        ece_val = compute_ece(y_val, p_val)
        brier_te = brier_score_loss(y_test, p_te)
        ece_te = compute_ece(y_test, p_te)
        roc_te = roc_auc_score(y_test, p_te)
        
        results[name] = {
            "val_brier": brier_val,
            "val_ece": ece_val,
            "test_brier": brier_te,
            "test_ece": ece_te,
            "test_roc": roc_te
        }
        print(f"  {name:<20} | {brier_val:.8f} | {ece_val:.6f}   | {brier_te:.8f} | {ece_te:.6f}   | {roc_te:.6f}")
    print("=" * 70)
    
    # 4. Ranking Impact & Spearman Correlation
    print("\n[Ranking Impact] Evaluating if Calibration Changes Ranking...")
    spearman_platt, _ = spearmanr(prob_test, prob_test_platt)
    spearman_iso, _ = spearmanr(prob_test, prob_test_iso)
    print(f"  Spearman Rank Correlation (Uncalibrated vs Platt):    {spearman_platt:.8f}")
    print(f"  Spearman Rank Correlation (Uncalibrated vs Isotonic): {spearman_iso:.8f}")
    
    # Check Top-K metrics across all three
    topk_uncal = compute_topk_metrics(y_test, prob_test)
    topk_platt = compute_topk_metrics(y_test, prob_test_platt)
    topk_iso = compute_topk_metrics(y_test, prob_test_iso)
    
    # 5. Save Calibrated Predictions CSV
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    calibrated_df = pd.DataFrame({
        "CVE_ID": test_df["CVE_ID"],
        "Observation_Date": test_df["Observation_Date"],
        "target": y_test,
        "prob_uncalibrated": prob_test,
        "prob_platt": prob_test_platt,
        "prob_isotonic": prob_test_iso
    })
    calibrated_df.to_csv("outputs/calibrated_predictions_test.csv", index=False)
    print(f"[Saved] outputs/calibrated_predictions_test.csv ({len(calibrated_df):,d} rows)")
    
    joblib.dump(platt_model, "models/platt_calibrator.pkl")
    joblib.dump(iso_model, "models/isotonic_calibrator.pkl")
    print(f"[Saved] Calibrator models saved to models/")
    
    # 6. Generate Report
    report_text = f"""# Probability Calibration Report (Validation 2023 -> Test 2024)

## Executive Summary

- **Objective**: Calibrate raw LightGBM probability estimates on the Validation set (2023) and evaluate calibration quality (Brier Score, Expected Calibration Error) and ranking stability on the untouched Test set (2024).
- **Methods Evaluated**:
  1. **Uncalibrated**: Raw probabilities from Improved LightGBM (`scale_pos_weight = 1.0`).
  2. **Platt Scaling**: Logistic Regression fit on Validation log-odds.
  3. **Isotonic Regression**: Non-parametric monotonic piecewise-constant fit on Validation probabilities.

---

## 1. Calibration Quality Comparison

| Method | Validation Brier Score | Validation ECE | Test Brier Score | Test ECE | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Uncalibrated** | {results['Uncalibrated']['val_brier']:.8f} | {results['Uncalibrated']['val_ece']:.6f} | {results['Uncalibrated']['test_brier']:.8f} | {results['Uncalibrated']['test_ece']:.6f} | {results['Uncalibrated']['test_roc']:.6f} |
| **Platt Scaling** | {results['Platt Scaling']['val_brier']:.8f} | {results['Platt Scaling']['val_ece']:.6f} | {results['Platt Scaling']['test_brier']:.8f} | {results['Platt Scaling']['test_ece']:.6f} | {results['Platt Scaling']['test_roc']:.6f} |
| **Isotonic Regression** | {results['Isotonic Regression']['val_brier']:.8f} | {results['Isotonic Regression']['val_ece']:.6f} | {results['Isotonic Regression']['test_brier']:.8f} | {results['Isotonic Regression']['test_ece']:.6f} | {results['Isotonic Regression']['test_roc']:.6f} |

---

## 2. Does Probability Calibration Change Ranking?

- **Mathematical Principle**: Platt scaling is a strictly monotonic sigmoid transformation $P_\\text{{Platt}} = \\sigma(a \\cdot \\text{{logit}} + b)$ where $a > 0$. Therefore, for all pairs with distinct probabilities, the relative rank order is **strictly invariant**.
- **Isotonic Regression Principle**: Isotonic regression is a non-decreasing step function. While it preserves weak monotonicity, it collapses distinct intermediate probabilities into discrete flat bins, creating large tie groups that can alter fine-grained Top-K ordering.
- **Empirical Rank Correlation (Spearman $\\rho$ on Test Set)**:
  - **Uncalibrated vs Platt Scaling**: $\\rho = {spearman_platt:.8f}$ (near-perfect rank preservation)
  - **Uncalibrated vs Isotonic Regression**: $\\rho = {spearman_iso:.8f}$

---

## 3. Top-K Ranking Impact on Test Set (2024)

| Top-K | Uncalibrated (Positives / Prec / Rec) | Platt Scaling (Positives / Prec / Rec) | Isotonic Regression (Positives / Prec / Rec) |
| :--- | :--- | :--- | :--- |
| **Top-10** | {topk_uncal[10]['positives']} / {topk_uncal[10]['precision']:.4f} / {topk_uncal[10]['recall']:.4f} | {topk_platt[10]['positives']} / {topk_platt[10]['precision']:.4f} / {topk_platt[10]['recall']:.4f} | {topk_iso[10]['positives']} / {topk_iso[10]['precision']:.4f} / {topk_iso[10]['recall']:.4f} |
| **Top-25** | {topk_uncal[25]['positives']} / {topk_uncal[25]['precision']:.4f} / {topk_uncal[25]['recall']:.4f} | {topk_platt[25]['positives']} / {topk_platt[25]['precision']:.4f} / {topk_platt[25]['recall']:.4f} | {topk_iso[25]['positives']} / {topk_iso[25]['precision']:.4f} / {topk_iso[25]['recall']:.4f} |
| **Top-50** | {topk_uncal[50]['positives']} / {topk_uncal[50]['precision']:.4f} / {topk_uncal[50]['recall']:.4f} | {topk_platt[50]['positives']} / {topk_platt[50]['precision']:.4f} / {topk_platt[50]['recall']:.4f} | {topk_iso[50]['positives']} / {topk_iso[50]['precision']:.4f} / {topk_iso[50]['recall']:.4f} |
| **Top-100** | {topk_uncal[100]['positives']} / {topk_uncal[100]['precision']:.4f} / {topk_uncal[100]['recall']:.4f} | {topk_platt[100]['positives']} / {topk_platt[100]['precision']:.4f} / {topk_platt[100]['recall']:.4f} | {topk_iso[100]['positives']} / {topk_iso[100]['precision']:.4f} / {topk_iso[100]['recall']:.4f} |
| **Top-200** | {topk_uncal[200]['positives']} / {topk_uncal[200]['precision']:.4f} / {topk_uncal[200]['recall']:.4f} | {topk_platt[200]['positives']} / {topk_platt[200]['precision']:.4f} / {topk_platt[200]['recall']:.4f} | {topk_iso[200]['positives']} / {topk_iso[200]['precision']:.4f} / {topk_iso[200]['recall']:.4f} |
| **Top-500** | {topk_uncal[500]['positives']} / {topk_uncal[500]['precision']:.4f} / {topk_uncal[500]['recall']:.4f} | {topk_platt[500]['positives']} / {topk_platt[500]['precision']:.4f} / {topk_platt[500]['recall']:.4f} | {topk_iso[500]['positives']} / {topk_iso[500]['precision']:.4f} / {topk_iso[500]['recall']:.4f} |
| **Top-1000** | {topk_uncal[1000]['positives']} / {topk_uncal[1000]['precision']:.4f} / {topk_uncal[1000]['recall']:.4f} | {topk_platt[1000]['positives']} / {topk_platt[1000]['precision']:.4f} / {topk_platt[1000]['recall']:.4f} | {topk_iso[1000]['positives']} / {topk_iso[1000]['precision']:.4f} / {topk_iso[1000]['recall']:.4f} |

---

## 4. Key Takeaway & Recommendation

- **Platt Scaling** preserves monotonic ranking while providing calibrated probabilities suitable for risk scoring and cost-benefit calculation.
- **Isotonic Regression** introduces binning plateaus in extreme tails where positive events are sparse, which makes fine-grained tie-breaking worse.
- **Recommendation**: Use **Platt Scaling** for downstream enterprise risk calculation and knapsack remediation optimization.
"""
    with open("reports/probability_calibration_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("[Saved] reports/probability_calibration_report.md generated successfully!")


if __name__ == "__main__":
    run_calibration()
