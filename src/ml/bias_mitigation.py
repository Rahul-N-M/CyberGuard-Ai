import json
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
)

# Reuse helper functions from the Alperin baseline
from src.ml.alperin_baseline import (
    load_dataset,
    build_preprocessor,
    compute_metrics,
    save_feature_importance,
    plot_roc_pr_curves,
    evaluate_topk,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Class‑weight experiments
EXPERIMENTS = {
    "original": "balanced",  # Experiment A
    "w10": {0: 1, 1: 10},      # Experiment B
    "w25": {0: 1, 1: 25},      # Experiment C
    "w50": {0: 1, 1: 50},      # Experiment D
    "w100": {0: 1, 1: 100},    # Experiment E
}

THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
TOPK_LIST = [100, 500]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train_df = load_dataset("train")
val_df = load_dataset("validation")
test_df = load_dataset("test")

TARGET_COL = "Target_KEV_180d"
FEATURE_COLS = [
    "CVE_ID",
    "Observation_Date",
    "CVSS_Score",
    "CVSS_Version",
    "Severity",
    "Vulnerability_Age_Days",
    "Description",
]

X_train = train_df[FEATURE_COLS].copy()
y_train = train_df[TARGET_COL].values
X_val = val_df[FEATURE_COLS].copy()
y_val = val_df[TARGET_COL].values
X_test = test_df[FEATURE_COLS].copy()
y_test = test_df[TARGET_COL].values

# ---------------------------------------------------------------------------
# Preprocess – fit ONLY on training data
# ---------------------------------------------------------------------------
preprocess = build_preprocessor()
preprocess.fit(X_train)
X_train_proc = preprocess.transform(X_train)
X_val_proc = preprocess.transform(X_val)
X_test_proc = preprocess.transform(X_test)

# ---------------------------------------------------------------------------
# Storage for results
# ---------------------------------------------------------------------------
metrics_rows = []
threshold_rows = []
feature_imp_rows = []
test_pred_frames = []
roc_curves = {}
pr_curves = {}
# top‑k will be computed later using deterministic sorting

def deterministic_sort(df, prob_col="predicted_prob"):
    # sort by probability descending, then CVE_ID, then Observation_Date
    return df.sort_values(
        by=[prob_col, "CVE_ID", "Observation_Date"],
        ascending=[False, True, True],
        kind="mergesort",
    )

for exp_name, cw in EXPERIMENTS.items():
    # -------------------------------------------------------------------
    # Train model with specified class weight
    # -------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        random_state=42,
        class_weight=cw,
        n_jobs=-1,
    )
    rf.fit(X_train_proc, y_train)
    # Save feature importance
    feat_imp_path = OUTPUT_DIR / f"bias_mitigation_feature_importance_{exp_name}.csv"
    importances = rf.feature_importances_
    cat_features = preprocess.named_transformers_["cat"].get_feature_names_out()
    num_features = preprocess.named_transformers_["num"].get_feature_names_out()
    lsa_features = [f"lsa_{i}" for i in range(preprocess.named_transformers_["txt"].named_steps["svd"].n_components)]
    all_features = list(cat_features) + list(num_features) + lsa_features
    df_imp = pd.DataFrame({"feature": all_features, "importance": importances})
    df_imp.sort_values("importance", ascending=False, inplace=True)
    df_imp.to_csv(feat_imp_path, index=False)
    if exp_name == "original":
        df_imp.to_csv(OUTPUT_DIR / "bias_mitigation_feature_importance.csv", index=False)
    feature_imp_rows.append({"experiment": exp_name, "path": str(feat_imp_path)})

    # -------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------
    def get_preds(X, split_name):
        probs = rf.predict_proba(X)[:, 1]
        labels = (probs > 0.5).astype(int)
        df = pd.DataFrame({
            "CVE_ID": globals()[f"{split_name}_df"]["CVE_ID"],
            "Observation_Date": globals()[f"{split_name}_df"]["Observation_Date"],
            "target": globals()[f"y_{split_name}"],
            "predicted_prob": probs,
            "predicted_label": labels,
        })
        return df

    pred_train = get_preds(X_train_proc, "train")
    pred_val = get_preds(X_val_proc, "val")
    pred_test = get_preds(X_test_proc, "test")

    test_pred_frames.append(pd.DataFrame({
        "CVE_ID": test_df["CVE_ID"],
        "Observation_Date": test_df["Observation_Date"],
        "target": y_test,
        f"predicted_prob_{exp_name}": pred_test["predicted_prob"],
        f"predicted_label_{exp_name}": pred_test["predicted_label"],
    }))

    # -------------------------------------------------------------------
    # Compute basic metrics (ROC, PR, etc.) via compute_metrics helper for consistency
    # -------------------------------------------------------------------
    for split_name, df in [("train", pred_train), ("validation", pred_val), ("test", pred_test)]:
        metrics = compute_metrics(df["target"].values, df["predicted_prob"].values, df["predicted_label"].values)
        metrics_row = {
            "experiment": exp_name,
            "split": split_name,
            **metrics,
        }
        metrics_rows.append(metrics_row)
        # Store curves for ROC/PR (only once per experiment for validation set)
        if split_name == "validation":
            from sklearn.metrics import roc_curve, precision_recall_curve
            fpr, tpr, _ = roc_curve(df["target"].values, df["predicted_prob"].values)
            precision, recall, _ = precision_recall_curve(df["target"].values, df["predicted_prob"].values)
            roc_curves[exp_name] = (fpr, tpr)
            pr_curves[exp_name] = (recall, precision)

    # -------------------------------------------------------------------
    # Threshold analysis on validation set
    # -------------------------------------------------------------------
    for thr in THRESHOLDS:
        pred_labels = (pred_val["predicted_prob"] > thr).astype(int)
        tp = int(((pred_labels == 1) & (pred_val["target"] == 1)).sum())
        fp = int(((pred_labels == 1) & (pred_val["target"] == 0)).sum())
        tn = int(((pred_labels == 0) & (pred_val["target"] == 0)).sum())
        fn = int(((pred_labels == 0) & (pred_val["target"] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        threshold_rows.append({
            "experiment": exp_name,
            "threshold": thr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        })

    # -------------------------------------------------------------------
    # Top‑K evaluation on validation and test (deterministic sorting)
    # -------------------------------------------------------------------
    for split_name, df in [("validation", pred_val), ("test", pred_test)]:
        df_sorted = deterministic_sort(df, prob_col="predicted_prob")
        # count ties for reporting
        tie_counts = df_sorted["predicted_prob"].value_counts()
        max_tie = tie_counts.max()
        # Precision@K and Recall@K
        for K in TOPK_LIST:
            top_k = df_sorted.head(K)
            pos_k = top_k[top_k["target"] == 1].shape[0]
            prec_k = pos_k / K
            rec_k = pos_k / df_sorted["target"].sum()
            metrics_rows.append({
                "experiment": exp_name,
                "split": f"{split_name}_topk_{K}",
                "Precision@K": prec_k,
                "Recall@K": rec_k,
            })

# ---------------------------------------------------------------------------
# Save aggregated metrics CSV
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame(metrics_rows)
metrics_path = OUTPUT_DIR / "bias_mitigation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

# Save threshold analysis CSV (validation only)
thresholds_df = pd.DataFrame(threshold_rows)
thresholds_path = OUTPUT_DIR / "bias_mitigation_thresholds.csv"
thresholds_df.to_csv(thresholds_path, index=False)

# ---------------------------------------------------------------------------
# Plot ROC curves (validation) for all experiments
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
for exp, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=exp)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves (validation) – class‑weight experiments")
plt.legend()
roc_path = OUTPUT_DIR / "bias_mitigation_comparison.png"
plt.savefig(roc_path, dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Plot PR curves (validation)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
for exp, (rec, prec) in pr_curves.items():
    plt.plot(rec, prec, label=exp)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR curves (validation) – class‑weight experiments")
plt.legend()
pr_path = OUTPUT_DIR / "bias_mitigation_pr_curves.png"
plt.savefig(pr_path, dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Plot F1 vs threshold (validation) for each experiment
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
for exp in EXPERIMENTS.keys():
    df_thr = thresholds_df[thresholds_df["experiment"] == exp]
    plt.plot(df_thr["threshold"], df_thr["f1"], marker="o", label=exp)
plt.xlabel("Threshold")
plt.ylabel("F1")
plt.title("F1 vs threshold (validation) – class‑weight experiments")
plt.legend()
thr_path = OUTPUT_DIR / "bias_mitigation_threshold_curves.png"
plt.savefig(thr_path, dpi=300)
plt.close()

test_pred_path = OUTPUT_DIR / "bias_mitigation_test_predictions.csv"
from functools import reduce
merged_test = reduce(lambda left, right: pd.merge(left, right, on=["CVE_ID", "Observation_Date", "target"], how="inner"), test_pred_frames)
merged_test.to_csv(test_pred_path, index=False)

# ---------------------------------------------------------------------------
# Write a brief report (Markdown) – placeholder, to be filled later
# ---------------------------------------------------------------------------
report_path = REPORT_DIR / "bias_mitigation_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Bias‑mitigation class‑weight experiments\n\n")
    f.write("This report contains the results of the class‑weight sensitivity analysis.\n")
    f.write("See the CSV files and plots in the `outputs/` directory for full details.\n")

print(f"[Saved] bias mitigation metrics -> {metrics_path}")
print(f"[Saved] bias mitigation thresholds -> {thresholds_path}")
print(f"[Saved] bias mitigation comparison plot -> {roc_path}")
print(f"[Saved] bias mitigation PR curves -> {pr_path}")
print(f"[Saved] bias mitigation threshold curves -> {thr_path}")
print(f"[Saved] bias mitigation test predictions -> {test_pred_path}")
print(f"[Saved] bias mitigation report -> {report_path}")
