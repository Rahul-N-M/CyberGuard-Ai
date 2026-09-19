"""
src/ml/generate_model_plots.py

Generates publication/presentation-quality plots from EXISTING prediction files.
Rules:
  - NO synthetic or invented values.
  - NO model retraining.
  - Loads existing predictions, evaluation files, and test features.
  - Outputs saved to outputs/ directory.
"""

import os
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

# Project paths
PROJECT_ROOT = Path("C:/Users/VINOD/OneDrive/Desktop/antygravity/CyberGuard-AI")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling defaults
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
    'figure.titleweight': 'bold'
})

print("[1/9] Loading prediction files...")
train_preds = pd.read_csv(OUTPUT_DIR / "lightgbm_predictions_train.csv")
val_preds = pd.read_csv(OUTPUT_DIR / "lightgbm_predictions_validation.csv")
test_preds = pd.read_csv(OUTPUT_DIR / "lightgbm_predictions_test.csv")

y_train, p_train = train_preds['target'].values, train_preds['predicted_prob'].values
y_val, p_val = val_preds['target'].values, val_preds['predicted_prob'].values
y_test, p_test = test_preds['target'].values, test_preds['predicted_prob'].values

print("[2/9] Generating Plot 1: ROC-AUC Comparison...")
# 1. ROC-AUC Comparison
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

fpr_train, tpr_train, _ = roc_curve(y_train, p_train)
fpr_val, tpr_val, _ = roc_curve(y_val, p_val)
fpr_test, tpr_test, _ = roc_curve(y_test, p_test)

auc_train = roc_auc_score(y_train, p_train)
auc_val = roc_auc_score(y_val, p_val)
auc_test = roc_auc_score(y_test, p_test)

ax.plot(fpr_train, tpr_train, label=f"Train (2022)  ROC-AUC = {auc_train:.6f}", color="#1f77b4", lw=2)
ax.plot(fpr_val, tpr_val, label=f"Val (2023)    ROC-AUC = {auc_val:.6f}", color="#2ca02c", lw=2)
ax.plot(fpr_test, tpr_test, label=f"Test (2024)   ROC-AUC = {auc_test:.6f}", color="#d62728", lw=2.5)
ax.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", lw=1.5, label="Random Chance (AUC = 0.5000)")

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR / Recall)")
ax.set_title("LightGBM Temporal Baseline: ROC Curves Across Chronological Splits")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

plt.tight_layout()
roc_path = OUTPUT_DIR / "model_roc_auc_comparison.png"
fig.savefig(roc_path)
plt.close(fig)
print(f"Saved: {roc_path}")

print("[3/9] Generating Plot 2: PR-AUC Comparison...")
# 2. PR-AUC Comparison (Dual panel: full view + operational zoom)
pr_train, re_train, _ = precision_recall_curve(y_train, p_train)
pr_val, re_val, _ = precision_recall_curve(y_val, p_val)
pr_test, re_test, _ = precision_recall_curve(y_test, p_test)

ap_train = average_precision_score(y_train, p_train)
ap_val = average_precision_score(y_val, p_val)
ap_test = average_precision_score(y_test, p_test)

base_train = y_train.mean()
base_val = y_val.mean()
base_test = y_test.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# Full scale
ax1.plot(re_train, pr_train, label=f"Train (2022)  PR-AUC = {ap_train:.6f}", color="#1f77b4", lw=2)
ax1.plot(re_val, pr_val, label=f"Val (2023)    PR-AUC = {ap_val:.6f}", color="#2ca02c", lw=2)
ax1.plot(re_test, pr_test, label=f"Test (2024)   PR-AUC = {ap_test:.6f}", color="#d62728", lw=2.5)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.set_title("Full PR Curves (0.0 to 1.0)")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

# Operational zoom (showing actual non-boundary precision values)
ax2.plot(re_train, pr_train, label=f"Train PR (Base = {base_train:.6f})", color="#1f77b4", lw=2)
ax2.plot(re_val, pr_val, label=f"Val PR (Base = {base_val:.6f})", color="#2ca02c", lw=2)
ax2.plot(re_test, pr_test, label=f"Test PR (Base = {base_test:.6f})", color="#d62728", lw=2.5)
ax2.axhline(base_test, color="#d62728", linestyle=":", lw=1.2, alpha=0.7)
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([0, 0.004])
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision (Zoomed: 0 to 0.004)")
ax2.set_title("Operational Range Zoom (Actual Empirical Precision)")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

plt.suptitle("LightGBM Temporal Baseline: PR-AUC Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
pr_comp_path = OUTPUT_DIR / "model_pr_auc_comparison.png"
fig.savefig(pr_comp_path)
plt.close(fig)
print(f"Saved: {pr_comp_path}")

print("[4/9] Generating Plot 3: Precision-Recall Curve (2024 Test)...")
# 3. Precision-Recall Curve for 2024 Test Set (Dual view: full + detailed operational region)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

p_95 = 0.00132992
r_95 = 0.43951613
p_50 = 0.00090537
r_50 = 0.54838710

# Left: Full View
ax1.plot(re_test, pr_test, color="#d62728", lw=2.5, label=f"Test PR Curve (AP = {ap_test:.6f})")
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.set_title("Full Test PR Curve")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

# Right: Zoomed view showing operating points
ax2.plot(re_test, pr_test, color="#d62728", lw=2.5, label=f"Test PR Curve")
ax2.axhline(base_test, color="#7f7f7f", linestyle="--", lw=1.5, label=f"Random Baseline ({base_test:.6f})")
ax2.scatter([r_95], [p_95], color="#1f77b4", s=110, zorder=5, label=f"Threshold 0.95 (R=43.95%, P=0.133%)")
ax2.scatter([r_50], [p_50], color="#2ca02c", s=90, zorder=5, label=f"Threshold 0.50 (R=54.84%, P=0.091%)")

ax2.annotate("Threshold = 0.95\nRecall: 43.95% (109 TP)\nPrecision: 0.133% (81,851 FP)",
             xy=(r_95, p_95), xytext=(r_95 - 0.35, 0.0022),
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#1f77b4", alpha=0.9))

ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([0, 0.0035])
ax2.set_xlabel("Recall (Positives Captured / 248)")
ax2.set_ylabel("Precision (TP / Flagged Observations)")
ax2.set_title("2024 Test Set: Operational Precision vs Recall")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

plt.suptitle("2024 Test Set: Precision-Recall Curve & Operating Points", fontsize=14, fontweight="bold")
plt.tight_layout()
pr_test_path = OUTPUT_DIR / "model_precision_recall_test.png"
fig.savefig(pr_test_path)
plt.close(fig)
print(f"Saved: {pr_test_path}")

print("[5/9] Generating Plot 4: Threshold vs Metrics...")
# 4. Threshold vs Recall / Precision / F1
thresholds = np.linspace(0.05, 0.99, 95)
prec_list, rec_list, f1_list = [], [], []

for t in thresholds:
    pred_bin = (p_test >= t).astype(int)
    tp = ((pred_bin == 1) & (y_test == 1)).sum()
    fp = ((pred_bin == 1) & (y_test == 0)).sum()
    fn = ((pred_bin == 0) & (y_test == 1)).sum()
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)

fig, ax1 = plt.subplots(figsize=(8.5, 6), dpi=300)
color1 = "#1f77b4"
color2 = "#d62728"
color3 = "#2ca02c"

ax1.plot(thresholds, rec_list, color=color1, lw=2.5, label="Recall")
ax1.set_xlabel("Decision Threshold")
ax1.set_ylabel("Recall", color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim([0, 0.70])

ax2 = ax1.twinx()
ax2.plot(thresholds, prec_list, color=color2, lw=2, linestyle="-.", label="Precision")
ax2.plot(thresholds, f1_list, color=color3, lw=2, linestyle="--", label="F1-Score")
ax2.set_ylabel("Precision & F1-Score", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim([0, 0.005])

ax1.axvline(0.95, color="#ff7f0e", linestyle=":", lw=2, label="Threshold 0.95")
ax1.annotate("Operating Threshold = 0.95\nRecall: 43.95% | FP: 81,851\nPrecision: 0.00133 | F1: 0.00265",
             xy=(0.95, 0.4395), xytext=(0.35, 0.52),
             arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8dc", edgecolor="#ff7f0e", alpha=0.9))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")

plt.title("2024 Test Set: Precision, Recall, and F1 across Decision Thresholds")
ax1.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
thresh_path = OUTPUT_DIR / "model_threshold_metrics.png"
fig.savefig(thresh_path)
plt.close(fig)
print(f"Saved: {thresh_path}")

print("[6/9] Generating Plot 5: Test Confusion Matrix (Threshold 0.95)...")
# 5. Confusion Matrix at Threshold 0.95
cm = np.array([[825991, 81851],
               [139,    109]])

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
fig.colorbar(cax, fraction=0.046, pad=0.04)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted Negative (0)", "Predicted Positive (1)"], fontweight="semibold")
ax.set_yticklabels(["Actual Negative (0)", "Actual Positive (1)"], fontweight="semibold")

for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct = count / cm.sum() * 100
        text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
        label = f"{count:,}\n({pct:.2f}%)"
        if i == 0 and j == 0: label += "\n[TN]"
        elif i == 0 and j == 1: label += "\n[FP]"
        elif i == 1 and j == 0: label += "\n[FN]"
        elif i == 1 and j == 1: label += "\n[TP]"
        ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")

ax.set_title("2024 Test Confusion Matrix at Decision Threshold = 0.95\n(Total Observations = 908,090)", pad=12)

metrics_text = (
    "Evaluation Metrics at Threshold 0.95:\n"
    "• True Positives (TP): 109   • False Positives (FP): 81,851\n"
    "• False Negatives (FN): 139  • True Negatives (TN): 825,991\n"
    "• Recall: 43.95% (109 / 248) • Precision: 0.133% (109 / 81,960)\n"
    "• F1-Score: 0.002652         • False Positive Rate: 9.02%"
)
fig.text(0.12, 0.02, metrics_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbbbbb"))

plt.subplots_adjust(bottom=0.22)
cm_path = OUTPUT_DIR / "model_test_confusion_matrix.png"
fig.savefig(cm_path)
plt.close(fig)
print(f"Saved: {cm_path}")

print("[7/9] Generating Plot 6: Predicted Probability Distribution...")
# 6. Predicted Probability Distribution (Saturation Analysis)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

bins = np.linspace(0, 1.0, 50)
counts, _, patches = ax1.hist(p_test, bins=bins, color="#1f77b4", edgecolor="black", log=True, alpha=0.75)
ax1.set_xlabel("Predicted Probability")
ax1.set_ylabel("Observation Count (Log Scale)")
ax1.set_title("Test Set Probability Distribution (All Rows)")
ax1.grid(True, linestyle=":", alpha=0.6)

sat_count = int((p_test >= 0.9999).sum())
sat_pct = sat_count / len(p_test) * 100
ax1.annotate(f"Saturation Peak at Prob = 1.0\nCount: {sat_count:,} ({sat_pct:.2f}% of test)",
             xy=(1.0, sat_count), xytext=(0.35, sat_count * 0.4),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=2),
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee", edgecolor="#d62728"))

# Probability distribution by target class
p_test_pos = p_test[y_test == 1]
p_test_neg = p_test[y_test == 0]

ax2.hist(p_test_neg, bins=bins, density=True, alpha=0.5, color="#1f77b4", label=f"Target = 0 (Negatives, N={len(p_test_neg):,})")
ax2.hist(p_test_pos, bins=bins, density=True, alpha=0.6, color="#d62728", label=f"Target = 1 (Positives, N={len(p_test_pos):,})")
ax2.set_xlabel("Predicted Probability")
ax2.set_ylabel("Density")
ax2.set_title("Probability Density by Class (Positives vs Negatives)")
ax2.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="#cccccc")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
dist_path = OUTPUT_DIR / "model_probability_distribution.png"
fig.savefig(dist_path)
plt.close(fig)
print(f"Saved: {dist_path}")

print("[8/9] Generating Plot 7: Top Feature Importance...")
# 7. Top Feature Importance
fi_df = pd.read_csv(OUTPUT_DIR / "lightgbm_feature_importance.csv")
fi_df_sorted = fi_df.sort_values(by="importance_gain", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300)

gain_billions = fi_df_sorted["importance_gain"] / 1e9
ax1.barh(fi_df_sorted["feature"], gain_billions, color="#1f77b4", edgecolor="black", alpha=0.85)
ax1.set_xlabel("Total Gain (Billions)")
ax1.set_title("Feature Importance by Total Gain")
ax1.grid(True, linestyle=":", alpha=0.6, axis="x")
for i, v in enumerate(gain_billions):
    if v > 0.1:
        ax1.text(v + 1.0, i, f"{v:.1f}B", va="center", fontsize=9, fontweight="bold")

fi_df_split_sorted = fi_df.sort_values(by="importance_split", ascending=True)
ax2.barh(fi_df_split_sorted["feature"], fi_df_split_sorted["importance_split"], color="#2ca02c", edgecolor="black", alpha=0.85)
ax2.set_xlabel("Number of Splits")
ax2.set_title("Feature Importance by Split Count")
ax2.grid(True, linestyle=":", alpha=0.6, axis="x")
for i, v in enumerate(fi_df_split_sorted["importance_split"]):
    if v > 0:
        ax2.text(v + 1.0, i, f"{v}", va="center", fontsize=9, fontweight="bold")

plt.suptitle("LightGBM Temporal Baseline: Feature Importance Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
fi_path = OUTPUT_DIR / "model_feature_importance.png"
fig.savefig(fi_path)
plt.close(fig)
print(f"Saved: {fi_path}")

print("[9/9] Generating Plot 8: Vulnerability Age vs Predicted Probability...")
# 8. Vulnerability Age vs Predicted Probability
test_features = pd.read_csv(DATA_DIR / "test.csv", usecols=["Vulnerability_Age_Days", "Target_KEV_180d"])
age_days = test_features["Vulnerability_Age_Days"].values

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

hb = ax.hexbin(age_days, p_test, gridsize=45, cmap="Blues", mincnt=1, bins="log", edgecolors="none")
cb = fig.colorbar(hb, ax=ax, pad=0.02)
cb.set_label("Log10(Observation Count)")

pos_mask = (y_test == 1)
ax.scatter(age_days[pos_mask], p_test[pos_mask], color="#d62728", s=35, alpha=0.85, label=f"Exploited Positives (N={pos_mask.sum()})", zorder=5)

ax.set_xlabel("Vulnerability Age at Observation Time (Days)")
ax.set_ylabel("Predicted Probability")
ax.set_title("Vulnerability Age vs Predicted Probability (2024 Test Set)")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")

ax.annotate("Saturation Group (Prob=1.0):\nMean Age = 110.7 days\n79,693 observations flagged",
            xy=(110.7, 1.0), xytext=(350, 0.82),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=2),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3e0", edgecolor="#d62728"))

plt.tight_layout()
age_prob_path = OUTPUT_DIR / "model_age_vs_probability.png"
fig.savefig(age_prob_path)
plt.close(fig)
print(f"Saved: {age_prob_path}")

print("[Bonus] Generating Plot 9: Threshold 0.95 vs Top-K Comparison...")
# 9. Threshold vs Top-K Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

strategies = ["Top-500 Ranking", "Threshold = 0.95"]
flagged_counts = [500, 81960]
tp_counts = [1, 109]
recalls = [1 / 248 * 100, 109 / 248 * 100]

colors = ["#1f77b4", "#ff7f0e"]

bars1 = ax1.bar(strategies, flagged_counts, color=colors, edgecolor="black", width=0.5)
ax1.set_yscale("log")
ax1.set_ylabel("Total Flagged Observations (Log Scale)")
ax1.set_title("Total Observations Flagged for Remediation")
ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
ax1.set_ylim([100, 200000])

for bar, count in zip(bars1, flagged_counts):
    ax1.text(bar.get_x() + bar.get_width()/2, count * 1.3, f"{count:,}", ha="center", fontweight="bold", fontsize=11)

bars2 = ax2.bar(strategies, recalls, color=colors, edgecolor="black", width=0.5)
ax2.set_ylabel("Exploit Recall (%)")
ax2.set_title("Exploit Recall Captured (Total Test Positives = 248)")
ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
ax2.set_ylim([0, 55])

for bar, rec, tp in zip(bars2, recalls, tp_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, rec + 2, f"{rec:.2f}%\n({tp} TP)", ha="center", fontweight="bold", fontsize=11)

plt.suptitle("Crucial Distinction: Top-500 Ranking vs Threshold 0.95 Selection", fontsize=13, fontweight="bold")
plt.tight_layout()
thresh_topk_path = OUTPUT_DIR / "model_threshold_vs_topk.png"
fig.savefig(thresh_topk_path)
plt.close(fig)
print(f"Saved: {thresh_topk_path}")

print("All 9 plots generated successfully in outputs/ directory!")
