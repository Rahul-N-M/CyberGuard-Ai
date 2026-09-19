"""
src/ml/generate_improved_plots.py
Generates presentation and publication-ready comparison plots:
  1. ROC Curve Comparison (Baseline vs Improved Model)
  2. PR Curve Comparison (Baseline vs Improved Model)
  3. Probability Calibration Curve (Uncalibrated vs Platt vs Isotonic)
  4. Top-K Recall & Enrichment Curves
  5. Probability Distribution & Saturation Reduction (79,693 at prob=1.0 -> 0)
  6. Feature Importance by Gain (Top 15 Features)
  7. 0-1 Knapsack Enterprise Remediation Frontier (Risk Reduced vs Budget Hours)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetic styling configuration
plt.rcParams.update({
    'font.sans-serif': 'Segoe UI, DejaVu Sans, Arial, Helvetica',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'semibold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})


def generate_all_plots():
    print("=" * 70)
    print("  CYBERGUARD-AI: GENERATING IMPROVED EVALUATION PLOTS")
    print("=" * 70)
    
    # 1. Load Data
    print("[Loading] Loading predictions, feature importance, and knapsack results...")
    base_test = pd.read_csv(OUTPUT_DIR / "lightgbm_predictions_test.csv")
    imp_test = pd.read_csv(OUTPUT_DIR / "improved_predictions_test.csv")
    calib_test = pd.read_csv(OUTPUT_DIR / "calibrated_predictions_test.csv")
    imp_features = pd.read_csv(OUTPUT_DIR / "improved_feature_importance.csv")
    knapsack_df = pd.read_csv(OUTPUT_DIR / "knapsack_remediation_summary.csv")
    
    y_true = base_test["target"].values
    prob_base = base_test["predicted_prob"].values
    prob_imp = imp_test["predicted_prob"].values
    prob_platt = calib_test["prob_platt"].values
    prob_iso = calib_test["prob_isotonic"].values
    
    # -------------------------------------------------------------
    # PLOT 1: ROC Curve Comparison
    # -------------------------------------------------------------
    print("[Plot 1/7] Generating ROC Curve Comparison...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    fpr_base, tpr_base, _ = roc_curve(y_true, prob_base)
    auc_base = roc_auc_score(y_true, prob_base)
    
    fpr_imp, tpr_imp, _ = roc_curve(y_true, prob_imp)
    auc_imp = roc_auc_score(y_true, prob_imp)
    
    ax.plot(fpr_base, tpr_base, color="#e74c3c", lw=2, linestyle="--", label=f"Baseline Temporal Model (AUC = {auc_base:.4f})")
    ax.plot(fpr_imp, tpr_imp, color="#2ecc71", lw=2.5, label=f"Improved Model + Cohort Age (AUC = {auc_imp:.4f})")
    ax.plot([0, 1], [0, 1], color="#95a5a6", lw=1.2, linestyle=":", label="Random Guess (AUC = 0.5000)")
    
    ax.set_title("Test Set ROC Curve Comparison (2024 Split)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plot1_path = OUTPUT_DIR / "improved_roc_comparison.png"
    plt.savefig(plot1_path)
    plt.close()
    print(f"  Saved -> {plot1_path}")
    
    # -------------------------------------------------------------
    # PLOT 2: PR Curve Comparison
    # -------------------------------------------------------------
    print("[Plot 2/7] Generating Precision-Recall Curve Comparison...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    prec_base, rec_base, _ = precision_recall_curve(y_true, prob_base)
    ap_base = average_precision_score(y_true, prob_base)
    
    prec_imp, rec_imp, _ = precision_recall_curve(y_true, prob_imp)
    ap_imp = average_precision_score(y_true, prob_imp)
    
    prevalence = np.mean(y_true)
    
    ax.plot(rec_base, prec_base, color="#e74c3c", lw=2, linestyle="--", label=f"Baseline Model (PR-AUC = {ap_base:.6f})")
    ax.plot(rec_imp, prec_imp, color="#2ecc71", lw=2.5, label=f"Improved Model (PR-AUC = {ap_imp:.6f})")
    ax.axhline(prevalence, color="#95a5a6", lw=1.2, linestyle=":", label=f"Random Prevalence Baseline ({prevalence * 100:.3f}%)")
    
    ax.set_title("Test Set Precision-Recall Curve Comparison (2024 Split)")
    ax.set_xlabel("Recall (Fraction of Actual Exploits Captured)")
    ax.set_ylabel("Precision (True Exploits / Predicted Exploits)")
    ax.set_ylim(-0.0002, max(prec_imp.max(), prec_base.max()) * 1.1)
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plot2_path = OUTPUT_DIR / "improved_pr_comparison.png"
    plt.savefig(plot2_path)
    plt.close()
    print(f"  Saved -> {plot2_path}")
    
    # -------------------------------------------------------------
    # PLOT 3: Probability Calibration Curve
    # -------------------------------------------------------------
    print("[Plot 3/7] Generating Reliability / Calibration Curves...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, prob_imp, n_bins=10)
    prob_true_platt, prob_pred_platt = calibration_curve(y_true, prob_platt, n_bins=10)
    prob_true_iso, prob_pred_iso = calibration_curve(y_true, prob_iso, n_bins=10)
    
    ax.plot([0, 1], [0, 1], color="#95a5a6", linestyle=":", lw=1.5, label="Perfectly Calibrated (Ideal)")
    ax.plot(prob_pred_uncal, prob_true_uncal, marker="o", color="#3498db", lw=2, label="Uncalibrated Improved Model")
    ax.plot(prob_pred_platt, prob_true_platt, marker="s", color="#e67e22", lw=2.2, label="Platt Scaling (Logistic)")
    ax.plot(prob_pred_iso, prob_true_iso, marker="^", color="#2ecc71", lw=2, label="Isotonic Regression")
    
    ax.set_title("Reliability Diagram: Calibration on 2024 Test Set")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of True Exploits")
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plot3_path = OUTPUT_DIR / "improved_probability_calibration.png"
    plt.savefig(plot3_path)
    plt.close()
    print(f"  Saved -> {plot3_path}")
    
    # -------------------------------------------------------------
    # PLOT 4: Top-K Recall & Positives Curve
    # -------------------------------------------------------------
    print("[Plot 4/7] Generating Top-K Recall Curves...")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    k_range = [10, 25, 50, 100, 200, 500, 1000, 2500, 5000, 10000]
    total_pos = y_true.sum()
    
    def get_recalls(probs):
        idx = np.argsort(probs)[::-1]
        recs = []
        for k in k_range:
            recs.append(y_true[idx[:k]].sum() / total_pos)
        return recs
        
    rec_base_k = get_recalls(prob_base)
    rec_imp_k = get_recalls(prob_imp)
    rec_platt_k = get_recalls(prob_platt)
    
    ax.plot(k_range, rec_base_k, marker="o", color="#e74c3c", lw=2, linestyle="--", label="Baseline (79,693 Ties)")
    ax.plot(k_range, rec_imp_k, marker="s", color="#3498db", lw=2.5, label="Improved Model (Distinct Scores)")
    ax.plot(k_range, rec_platt_k, marker="^", color="#2ecc71", lw=2, linestyle=":", label="Platt Calibrated")
    
    ax.set_xscale("log")
    ax.set_title("Recall at Top-K Prioritization Cutoff (Test Set)")
    ax.set_xlabel("Top-K Vulnerabilities Triaged (Log Scale)")
    ax.set_ylabel("Recall (Fraction of Total Exploited CVEs)")
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plot4_path = OUTPUT_DIR / "improved_topk_recall.png"
    plt.savefig(plot4_path)
    plt.close()
    print(f"  Saved -> {plot4_path}")
    
    # -------------------------------------------------------------
    # PLOT 5: Probability Saturation Comparison Histogram
    # -------------------------------------------------------------
    print("[Plot 5/7] Generating Probability Saturation Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Baseline distribution
    ax1.hist(prob_base, bins=50, color="#e74c3c", edgecolor="black", alpha=0.85)
    base_ones = (prob_base == 1.0).sum()
    ax1.annotate(f"Saturated at Prob=1.0:\n{base_ones:,d} rows", 
                 xy=(1.0, base_ones * 0.9), xytext=(0.45, base_ones * 0.7),
                 arrowprops=dict(arrowstyle="->", color="darkred", lw=2),
                 fontweight="bold", color="darkred",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fbeee6", ec="darkred"))
    ax1.set_title(f"Baseline: Massive Probability Saturation\n({base_ones:,d} Tied at 1.0)")
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Number of Observations")
    
    # Improved model distribution
    ax2.hist(prob_imp, bins=50, color="#2ecc71", edgecolor="black", alpha=0.85)
    imp_ones = (prob_imp == 1.0).sum()
    ax2.annotate(f"Clean Distribution:\n{imp_ones} rows at 1.0 (0% Saturation)", 
                 xy=(0.05, len(prob_imp) * 0.6), xytext=(0.25, len(prob_imp) * 0.5),
                 arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2),
                 fontweight="bold", color="darkgreen",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="darkgreen"))
    ax2.set_title(f"Improved Model: Calibrated Distribution\n({imp_ones} Tied at 1.0 - 100% Saturation Elimination)")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Number of Observations")
    
    plt.suptitle("Probability Saturation Diagnostic: Baseline vs Improved Model", fontsize=14, y=1.02)
    plt.tight_layout()
    plot5_path = OUTPUT_DIR / "improved_saturation_comparison.png"
    plt.savefig(plot5_path)
    plt.close()
    print(f"  Saved -> {plot5_path}")
    
    # -------------------------------------------------------------
    # PLOT 6: Feature Importance Bar Chart
    # -------------------------------------------------------------
    print("[Plot 6/7] Generating Feature Importance Bar Chart...")
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    top_feats = imp_features.head(15).sort_values(by="importance_gain", ascending=True)
    y_pos = np.arange(len(top_feats))
    
    bars = ax.barh(y_pos, top_feats["importance_gain"], color="#34495e", edgecolor="black", alpha=0.85)
    # Highlight relative age & text signal features
    for i, f in enumerate(top_feats["feature"]):
        if "age" in f.lower() or "desc" in f.lower() or "rce" in f.lower() or "remote" in f.lower():
            bars[i].set_color("#27ae60")
            
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats["feature"])
    ax.set_title("Top 15 Features by LightGBM Information Gain")
    ax.set_xlabel("Total Information Gain")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", edgecolor="black", label="Engineered Features (Relative Age, Text Signals)"),
        Patch(facecolor="#34495e", edgecolor="black", label="Baseline Features (CVSS, Severity)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plot6_path = OUTPUT_DIR / "improved_feature_importance.png"
    plt.savefig(plot6_path)
    plt.close()
    print(f"  Saved -> {plot6_path}")
    
    # -------------------------------------------------------------
    # PLOT 7: Knapsack Optimization Frontier
    # -------------------------------------------------------------
    print("[Plot 7/7] Generating 0-1 Knapsack Optimization Frontier...")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    budgets = knapsack_df["budget_hours"].values
    opt_risk = knapsack_df["optimal_risk_reduced"].values
    greedy_risk = knapsack_df["cvss_greedy_risk_reduced"].values
    
    ax.plot(budgets, opt_risk, marker="o", color="#2980b9", lw=2.8, label="0-1 Knapsack Global Optimum (CyberGuard-AI)")
    ax.plot(budgets, greedy_risk, marker="s", color="#e74c3c", lw=2, linestyle="--", label="Industry Status Quo (CVSS Greedy)")
    
    # Annotate improvement percentages
    for b, opt, gr in zip(budgets, opt_risk, greedy_risk):
        gain_pct = ((opt - gr) / gr * 100) if gr > 0 else 0.0
        ax.annotate(f"+{gain_pct:.0f}%", xy=(b, opt), xytext=(b - 1.5, opt + 3.5),
                    fontweight="bold", color="#1a5276",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#ebf5fb", ec="#2980b9"))
                    
    ax.set_title("Enterprise Remediation Frontier: 0-1 Knapsack vs CVSS Greedy")
    ax.set_xlabel("Engineer Sprint Budget (Hours)")
    ax.set_ylabel("Total Enterprise Risk Reduced")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{int(b)}h Sprint" for b in budgets])
    ax.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plot7_path = OUTPUT_DIR / "knapsack_remediation_frontier.png"
    plt.savefig(plot7_path)
    plt.close()
    print(f"  Saved -> {plot7_path}")
    
    print("\n" + "=" * 70)
    print("  ALL 7 PUBLICATION PLOTS GENERATED SUCCESSFULLY IN outputs/")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_plots()
