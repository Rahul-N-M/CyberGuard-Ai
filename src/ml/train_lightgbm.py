"""
src/ml/train_lightgbm.py
Training script for the Baseline LightGBM model on CyberGuard AI.

Safety Checks & Strict Constraints:
  1. No SMOTE or synthetic oversampling.
  2. No EPSS features.
  3. No KEV, Date_Added, target-derived, or identifier features.
  4. scale_pos_weight computed ONLY from the training dataset.
  5. Early stopping based ONLY on the validation set.
  6. Final evaluation ONCE on the untouched test set.

Outputs:
  - Model saved to models/lightgbm_baseline.pkl
  - Predictions saved to outputs/baseline_predictions.csv
  - Metrics report saved to reports/baseline_metrics.json
"""

import os
import sys
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


def run_safety_checks(X_train, y_train, X_val, y_val, X_test, y_test):
    """Executes mandatory safety checks prior to training."""
    print("=" * 65)
    print("  RUNNING PRE-TRAINING SAFETY CHECKS")
    print("=" * 65)

    # 1. Feature columns match
    assert (X_train.columns == X_val.columns).all(), "FAIL: Train and Val feature columns mismatch!"
    assert (X_val.columns == X_test.columns).all(), "FAIL: Val and Test feature columns mismatch!"
    print("  [PASS] Feature columns and order match 100% across all splits (32 features).")

    # 2. No leakage columns
    leakage_terms = ["kev", "epss", "cve_id", "asset_id", "date_added", "published_date"]
    found_leakage = [c for c in X_train.columns if any(term in c.lower() for term in leakage_terms)]
    assert len(found_leakage) == 0, f"FAIL: Leakage columns detected in features: {found_leakage}"
    print("  [PASS] Zero leakage or EPSS-derived columns present in X.")

    # 3. Binary target
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        assert set(np.unique(y)).issubset({0, 1}), f"FAIL: {name} target is not binary!"
    print("  [PASS] Target is strictly binary {0, 1} across all splits.")

    # 4. scale_pos_weight computed ONLY from train
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  [PASS] scale_pos_weight computed strictly from Train: {n_neg} / {n_pos} = {scale_pos_weight:.6f}")

    print("=" * 65)
    print("  ALL SAFETY CHECKS PASSED SUCCESSFULLY -- PROCEEDING TO TRAIN")
    print("=" * 65 + "\n")

    return scale_pos_weight


def compute_precision_recall_at_k(y_true, y_prob, k_values=[10, 20, 50, 100, 200]):
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


def evaluate_split(y_true, y_prob, threshold=0.5, k_values=[10, 20, 50, 100, 200]):
    """Evaluates a split and returns a dictionary of metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    # Handle single class edge cases safely for metrics
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()
    pk_rk = compute_precision_recall_at_k(y_true, y_prob, k_values=k_values)

    return {
        "ROC_AUC": round(roc_auc, 6),
        "PR_AUC_Average_Precision": round(pr_auc, 6),
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "F1_Score": round(f1, 6),
        "Confusion_Matrix": cm,
        "Total_Rows": len(y_true),
        "Actual_Positives": int((y_true == 1).sum()),
        "Predicted_Positives": int((y_pred == 1).sum()),
        **pk_rk
    }


def train_baseline_lightgbm():
    # 1. Load Data
    train_path = "data/processed/baseline_train.csv"
    val_path   = "data/processed/baseline_val.csv"
    test_path  = "data/processed/baseline_test.csv"

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_test  = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"].values

    X_val = df_val.drop(columns=["target"])
    y_val = df_val["target"].values

    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"].values

    # 2. Run Pre-Training Safety Checks
    scale_pos_weight = run_safety_checks(X_train, y_train, X_val, y_val, X_test, y_test)

    # 3. LightGBM Hyperparameters
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    print("LightGBM Hyperparameters:")
    for k, v in params.items():
        print(f"  • {k}: {v}")
    print()

    # 4. Train Model
    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, first_metric_only=True, verbose=True)]
    )

    best_iter = model.best_iteration_
    print(f"\n[OK] Training completed. Best iteration: {best_iter}")

    # 5. Model Inference
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val   = model.predict_proba(X_val)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]

    # 6. Evaluate Splits
    train_metrics = evaluate_split(y_train, y_prob_train)
    val_metrics   = evaluate_split(y_val, y_prob_val)
    test_metrics  = evaluate_split(y_test, y_prob_test)

    # 7. Extract Feature Importances
    imp_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_split": model.booster_.feature_importance(importance_type="split"),
        "importance_gain": model.booster_.feature_importance(importance_type="gain")
    }).sort_values(by="importance_gain", ascending=False)

    top_features = imp_df.head(15).to_dict(orient="records")

    # 8. Save Outputs & Metrics
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    model_path = "models/lightgbm_baseline.pkl"
    joblib.dump(model, model_path)
    print(f"  [Saved] Model weights saved to: {model_path}")

    # Save predictions
    preds_df = pd.DataFrame({
        "split": ["train"]*len(df_train) + ["val"]*len(df_val) + ["test"]*len(df_test),
        "target": np.concatenate([y_train, y_val, y_test]),
        "predicted_prob": np.concatenate([y_prob_train, y_prob_val, y_prob_test]),
        "predicted_label": (np.concatenate([y_prob_train, y_prob_val, y_prob_test]) >= 0.5).astype(int)
    })
    preds_path = "outputs/baseline_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"  [Saved] Predictions saved to: {preds_path}")

    # Save metrics JSON report
    report_dict = {
        "model_name": "LightGBM_Baseline",
        "best_iteration": int(best_iter),
        "parameters": params,
        "metrics": {
            "Train": train_metrics,
            "Validation": val_metrics,
            "Test": test_metrics
        },
        "top_feature_importance": top_features
    }

    report_path = "reports/baseline_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  [Saved] Evaluation report saved to: {report_path}")

    return report_dict, imp_df


if __name__ == "__main__":
    train_baseline_lightgbm()