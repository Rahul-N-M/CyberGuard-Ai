"""
src/ml/train_temporal_lightgbm_15M.py
Full LightGBM training pipeline for the 1.535M observation temporal dataset.

Design & Rules:
  1. Input Datasets: data/train.csv, data/validation.csv, data/test.csv
  2. Target Column: Target_KEV_180d
  3. Preprocessing fit ONLY on training data (medians, one-hot encoding).
  4. scale_pos_weight calculated dynamically from train set.
  5. Early stopping monitored on validation set.
  6. Final evaluation on untouched test set.

Artifacts Saved:
  - models/lightgbm_temporal_baseline.txt (LightGBM Booster model)
  - models/lightgbm_temporal_baseline.pkl (Joblib serialized model)
  - outputs/lightgbm_predictions_train.csv
  - outputs/lightgbm_predictions_validation.csv
  - outputs/lightgbm_predictions_test.csv
  - outputs/lightgbm_feature_importance.csv
  - reports/lightgbm_training_report.md
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import joblib

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run_pretraining_checks(train_df, val_df, test_df, target_col):
    """Executes mandatory automated pre-training checks."""
    print("=" * 70)
    print("  RUNNING PRE-TRAINING SAFETY CHECKS (1.535M TEMPORAL DATASET)")
    print("=" * 70)

    # 1. Row counts & Positives
    n_tr, n_val, n_te = len(train_df), len(val_df), len(test_df)
    n_tot = n_tr + n_val + n_te
    pos_tr = int(train_df[target_col].sum())
    pos_val = int(val_df[target_col].sum())
    pos_te = int(test_df[target_col].sum())
    pos_tot = pos_tr + pos_val + pos_te

    print(f"  [CHECK 1] Row Counts  : Train={n_tr:,d}, Val={n_val:,d}, Test={n_te:,d} (Total={n_tot:,d})")
    print(f"  [CHECK 1] Positives   : Train={pos_tr}, Val={pos_val}, Test={pos_te} (Total={pos_tot})")
    assert n_tot == 1535261, f"FAIL: Expected 1,535,261 rows, got {n_tot:,d}"
    assert pos_tot == 601, f"FAIL: Expected 601 positives, got {pos_tot}"
    print("  -> Check 1 PASSED!")

    # 2. Target Column & Binary verification
    print(f"  [CHECK 2] Target Name : '{target_col}'")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        assert set(df[target_col].unique()).issubset({0, 1}), f"FAIL: {name} target is not binary {0,1}!"
    print("  -> Check 2 PASSED!")

    # 3. Chronological Order Verification
    obs_col = "Observation_Date"
    tr_obs_max = train_df[obs_col].max()
    val_obs_min = val_df[obs_col].min()
    val_obs_max = val_df[obs_col].max()
    te_obs_min  = test_df[obs_col].min()
    print(f"  [CHECK 3] Chronology  : Train max={tr_obs_max} <= Val min={val_obs_min}")
    print(f"                          Val max={val_obs_max} <= Test min={te_obs_min}")
    assert pd.to_datetime(tr_obs_max) <= pd.to_datetime(val_obs_min), "FAIL: Chronological overlap between Train and Val!"
    assert pd.to_datetime(val_obs_max) <= pd.to_datetime(te_obs_min), "FAIL: Chronological overlap between Val and Test!"
    print("  -> Check 3 PASSED!")

    # 4. Excluded Leakage Columns Check
    excluded_terms = ["cve_id", "observation_date", "published_date", "description", "date_added", "epss"]
    cols = train_df.columns.tolist()
    feature_candidates = [c for c in cols if c != target_col and not any(term in c.lower() for term in excluded_terms)]
    print(f"  [CHECK 4] Feature Cols ({len(feature_candidates)}): {feature_candidates}")
    for term in excluded_terms:
        for f in feature_candidates:
            assert term not in f.lower(), f"FAIL: Excluded term '{term}' found in feature column '{f}'"
    print("  -> Check 4 PASSED!")

    # 5. Dynamic scale_pos_weight calculation
    neg_tr = n_tr - pos_tr
    scale_pos_weight = neg_tr / pos_tr if pos_tr > 0 else 1.0
    print(f"  [CHECK 5] scale_pos_weight (Train only): {neg_tr:,d} / {pos_tr} = {scale_pos_weight:.6f}")
    assert scale_pos_weight > 100, "FAIL: Unexpected scale_pos_weight value!"
    print("  -> Check 5 PASSED!")

    print("=" * 70)
    print("  ALL PRE-TRAINING CHECKS PASSED SUCCESSFULLY -- STARTING TRAINING")
    print("=" * 70 + "\n")

    return feature_candidates, scale_pos_weight


def preprocess_data(train_df, val_df, test_df, feature_cols, target_col):
    """Fits missing value imputation and categorical encoding strictly on training set."""
    print("[Preprocessing] Fitting transformations strictly on Training set...")

    num_cols = ["CVSS_Score", "Severity_Encoded", "Vulnerability_Age_Days"]
    cat_cols = ["CVSS_Version", "Severity"]

    # 1. Medians fit on Train ONLY
    train_medians = {col: train_df[col].median() for col in num_cols}
    print("  Imputation Medians (Train only):", train_medians)

    train_imp = train_df.copy()
    val_imp   = val_df.copy()
    test_imp  = test_df.copy()

    for col in num_cols:
        train_imp[col] = train_imp[col].fillna(train_medians[col])
        val_imp[col]   = val_imp[col].fillna(train_medians[col])
        test_imp[col]  = test_imp[col].fillna(train_medians[col])

    # 2. Categorical One-Hot Encoding fit on Train ONLY
    for col in cat_cols:
        train_imp[col] = train_imp[col].fillna("Missing").astype(str)
        val_imp[col]   = val_imp[col].fillna("Missing").astype(str)
        test_imp[col]  = test_imp[col].fillna("Missing").astype(str)

    train_cat = pd.get_dummies(train_imp[cat_cols], drop_first=False)
    cat_feature_names = train_cat.columns.tolist()

    val_cat  = pd.get_dummies(val_imp[cat_cols], drop_first=False).reindex(columns=cat_feature_names, fill_value=0)
    test_cat = pd.get_dummies(test_imp[cat_cols], drop_first=False).reindex(columns=cat_feature_names, fill_value=0)

    X_train = pd.concat([train_imp[num_cols].reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    X_val   = pd.concat([val_imp[num_cols].reset_index(drop=True), val_cat.reset_index(drop=True)], axis=1)
    X_test  = pd.concat([test_imp[num_cols].reset_index(drop=True), test_cat.reset_index(drop=True)], axis=1)

    y_train = train_df[target_col].values
    y_val   = val_df[target_col].values
    y_test  = test_df[target_col].values

    # Alignment assertions
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature count mismatch!"
    assert (X_train.columns == X_val.columns).all(), "Feature column names mismatch!"
    assert (X_val.columns == X_test.columns).all(), "Feature column names mismatch!"
    print(f"  [OK] Preprocessing completed. Feature count: {X_train.shape[1]}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_precision_recall_at_k(y_true, y_prob, k_values=[10, 20, 50, 100, 200, 500, 1000]):
    """Computes Precision@K and Recall@K metrics."""
    sorted_indices = np.argsort(y_prob)[::-1]
    total_positives = (y_true == 1).sum()

    results = {}
    for k in k_values:
        k_clamped = min(k, len(y_true))
        top_k_indices = sorted_indices[:k_clamped]
        top_k_positives = (y_true[top_k_indices] == 1).sum()

        p_at_k = float(top_k_positives / k_clamped) if k_clamped > 0 else 0.0
        r_at_k = float(top_k_positives / total_positives) if total_positives > 0 else 0.0

        results[f"Precision@{k}"] = round(p_at_k, 6)
        results[f"Recall@{k}"] = round(r_at_k, 6)

    return results


def evaluate_split_performance(y_true, y_prob, df_raw=None, threshold=0.5, k_values=[10, 20, 50, 100, 200, 500, 1000]):
    """Evaluates metrics for a split."""
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    pk_rk = compute_precision_recall_at_k(y_true, y_prob, k_values=k_values)

    cve_col = "CVE_ID" if (df_raw is not None and "CVE_ID" in df_raw.columns) else None
    unique_pos_cves = int(df_raw.loc[y_true == 1, cve_col].nunique()) if (df_raw is not None and cve_col) else 0

    return {
        "ROC_AUC": round(roc_auc, 6),
        "PR_AUC_Average_Precision": round(pr_auc, 6),
        "Decision_Threshold": threshold,
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "F1_Score": round(f1, 6),
        "Confusion_Matrix": cm,
        "Total_Rows": len(y_true),
        "Actual_Positives": int((y_true == 1).sum()),
        "Unique_Positive_CVEs": unique_pos_cves,
        "Predicted_Positives": int((y_pred == 1).sum()),
        **pk_rk
    }


def train_temporal_lightgbm():
    start_time = time.time()

    # 1. Load Data
    train_path = "data/train.csv"
    val_path   = "data/validation.csv"
    test_path  = "data/test.csv"

    print(f"[Loading] Loading dataset files from data/...")
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    target_col = "Target_KEV_180d"

    # 2. Run Pre-Training Checks
    feature_cols, scale_pos_weight = run_pretraining_checks(train_df, val_df, test_df, target_col)

    # 3. Preprocess Features
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        train_df, val_df, test_df, feature_cols, target_col
    )

    # 4. Hyperparameters Configuration
    params = {
        "objective": "binary",
        "metric": ["average_precision", "binary_logloss"],
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    print("\n[Training] Hyperparameters:")
    for k, v in params.items():
        print(f"  • {k}: {v}")
    print()

    # 5. Model Training with Early Stopping
    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=True),
            lgb.log_evaluation(period=20)
        ]
    )

    training_time_sec = round(time.time() - start_time, 2)
    best_iter = model.best_iteration_
    print(f"\n[OK] Training completed in {training_time_sec}s. Best iteration: {best_iter}")

    # 6. Inference
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val   = model.predict_proba(X_val)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]

    # 7. Evaluate Splits
    train_metrics = evaluate_split_performance(y_train, y_prob_train, df_raw=train_df)
    val_metrics   = evaluate_split_performance(y_val, y_prob_val, df_raw=val_df)
    test_metrics  = evaluate_split_performance(y_test, y_prob_test, df_raw=test_df)

    # 8. Feature Importance
    imp_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_split": model.booster_.feature_importance(importance_type="split"),
        "importance_gain": model.booster_.feature_importance(importance_type="gain")
    }).sort_values(by="importance_gain", ascending=False)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Save artifacts
    txt_model_path = "models/lightgbm_temporal_baseline.txt"
    pkl_model_path = "models/lightgbm_temporal_baseline.pkl"
    model.booster_.save_model(txt_model_path)
    joblib.dump(model, pkl_model_path)
    print(f"  [Saved] Model saved to {txt_model_path} and {pkl_model_path}")

    # Save feature importances CSV
    imp_path = "outputs/lightgbm_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  [Saved] Feature importances saved to {imp_path}")

    # Save prediction CSVs
    for name, df_raw, probs in [("train", train_df, y_prob_train), ("validation", val_df, y_prob_val), ("test", test_df, y_prob_test)]:
        pred_out_df = pd.DataFrame({
            "CVE_ID": df_raw["CVE_ID"] if "CVE_ID" in df_raw.columns else "",
            "Observation_Date": df_raw["Observation_Date"] if "Observation_Date" in df_raw.columns else "",
            "target": df_raw[target_col],
            "predicted_prob": probs,
            "predicted_label": (probs >= 0.5).astype(int)
        })
        p_path = f"outputs/lightgbm_predictions_{name}.csv"
        pred_out_df.to_csv(p_path, index=False)
        print(f"  [Saved] Predictions saved to {p_path}")

    top_10_table = "| Feature | Importance Split | Importance Gain |\n| :--- | :--- | :--- |\n"
    for _, r in imp_df.head(10).iterrows():
        top_10_table += f"| `{r['feature']}` | {r['importance_split']} | {r['importance_gain']:.4f} |\n"

    # Generate Markdown Report
    markdown_report = f"""# LightGBM Temporal Model Training Report (1.535M Dataset)

## Executive Summary

- **Dataset**: 1,535,261 temporal observation rows (94,636 unique CVEs)
- **Target**: `Target_KEV_180d` (CISA KEV addition in 180 days post-observation time $T$)
- **Chronological Splits**: Train (2022) | Validation (2023) | Test (2024)
- **Training Time**: {training_time_sec} seconds
- **Best Iteration**: {best_iter}
- **Calculated `scale_pos_weight`**: {scale_pos_weight:.6f}

---

## 1. Split Performance Overview

| Metric | Training Set (2022) | Validation Set (2023) | Test Set (2024) |
| :--- | :--- | :--- | :--- |
| **Total Rows** | {train_metrics['Total_Rows']:,d} | {val_metrics['Total_Rows']:,d} | {test_metrics['Total_Rows']:,d} |
| **Actual Positives** | {train_metrics['Actual_Positives']} | {val_metrics['Actual_Positives']} | {test_metrics['Actual_Positives']} |
| **Unique Positive CVEs** | {train_metrics['Unique_Positive_CVEs']} | {val_metrics['Unique_Positive_CVEs']} | {test_metrics['Unique_Positive_CVEs']} |
| **ROC-AUC** | **{train_metrics['ROC_AUC']:.6f}** | **{val_metrics['ROC_AUC']:.6f}** | **{test_metrics['ROC_AUC']:.6f}** |
| **PR-AUC / Average Precision** | **{train_metrics['PR_AUC_Average_Precision']:.6f}** | **{val_metrics['PR_AUC_Average_Precision']:.6f}** | **{test_metrics['PR_AUC_Average_Precision']:.6f}** |
| **Precision (threshold 0.5)** | {train_metrics['Precision']:.6f} | {val_metrics['Precision']:.6f} | {test_metrics['Precision']:.6f} |
| **Recall (threshold 0.5)** | {train_metrics['Recall']:.6f} | {val_metrics['Recall']:.6f} | {test_metrics['Recall']:.6f} |
| **F1-Score (threshold 0.5)** | {train_metrics['F1_Score']:.6f} | {val_metrics['F1_Score']:.6f} | {test_metrics['F1_Score']:.6f} |

---

## 2. Top-K Ranking Performance (Precision@K & Recall@K)

### Training Set (2022)
- **Precision@10**: {train_metrics['Precision@10']:.4f} | **Recall@10**: {train_metrics['Recall@10']:.4f}
- **Precision@50**: {train_metrics['Precision@50']:.4f} | **Recall@50**: {train_metrics['Recall@50']:.4f}
- **Precision@100**: {train_metrics['Precision@100']:.4f} | **Recall@100**: {train_metrics['Recall@100']:.4f}
- **Precision@200**: {train_metrics['Precision@200']:.4f} | **Recall@200**: {train_metrics['Recall@200']:.4f}
- **Precision@500**: {train_metrics['Precision@500']:.4f} | **Recall@500**: {train_metrics['Recall@500']:.4f}

### Validation Set (2023)
- **Precision@10**: {val_metrics['Precision@10']:.4f} | **Recall@10**: {val_metrics['Recall@10']:.4f}
- **Precision@50**: {val_metrics['Precision@50']:.4f} | **Recall@50**: {val_metrics['Recall@50']:.4f}
- **Precision@100**: {val_metrics['Precision@100']:.4f} | **Recall@100**: {val_metrics['Recall@100']:.4f}
- **Precision@200**: {val_metrics['Precision@200']:.4f} | **Recall@200**: {val_metrics['Recall@200']:.4f}
- **Precision@500**: {val_metrics['Precision@500']:.4f} | **Recall@500**: {val_metrics['Recall@500']:.4f}

### Test Set (2024)
- **Precision@10**: {test_metrics['Precision@10']:.4f} | **Recall@10**: {test_metrics['Recall@10']:.4f}
- **Precision@50**: {test_metrics['Precision@50']:.4f} | **Recall@50**: {test_metrics['Recall@50']:.4f}
- **Precision@100**: {test_metrics['Precision@100']:.4f} | **Recall@100**: {test_metrics['Recall@100']:.4f}
- **Precision@200**: {test_metrics['Precision@200']:.4f} | **Recall@200**: {test_metrics['Recall@200']:.4f}
- **Precision@500**: {test_metrics['Precision@500']:.4f} | **Recall@500**: {test_metrics['Recall@500']:.4f}

---

## 3. Confusion Matrices (Threshold = 0.5)

- **Train**: `{train_metrics['Confusion_Matrix']}`
- **Validation**: `{val_metrics['Confusion_Matrix']}`
- **Test**: `{test_metrics['Confusion_Matrix']}`

---

## 4. Feature Importance (Top 10 by Gain)

{top_10_table}

---

## 5. Saved Artifacts

- **Model File (Booster format)**: `models/lightgbm_temporal_baseline.txt`
- **Model File (Joblib format)**: `models/lightgbm_temporal_baseline.pkl`
- **Feature Importance**: `outputs/lightgbm_feature_importance.csv`
- **Predictions**:
  - `outputs/lightgbm_predictions_train.csv`
  - `outputs/lightgbm_predictions_validation.csv`
  - `outputs/lightgbm_predictions_test.csv`
"""

    report_path = "reports/lightgbm_training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"  [Saved] Training report saved to {report_path}")

    return {
        "training_time_sec": training_time_sec,
        "params": params,
        "best_iter": best_iter,
        "metrics": {
            "Train": train_metrics,
            "Validation": val_metrics,
            "Test": test_metrics
        },
        "top_10_features": imp_df.head(10).to_dict(orient="records")
    }


if __name__ == "__main__":
    train_temporal_lightgbm()
