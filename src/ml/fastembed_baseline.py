"""
src/ml/fastembed_baseline.py
============================
Replication of FastEmbed (Fang et al., PLOS ONE 2020, DOI:10.1371/journal.pone.0228439)
architecture on CyberGuard temporal dataset.

FastEmbed = fastText word embeddings (on CVE descriptions) + LightGBM classifier.

Design Rules (fair comparison, no bias):
  1. fastText model fitted on TRAIN descriptions only — never val/test.
  2. Same temporal splits as all other CyberGuard models (train/val/test CSV files).
  3. Same target: Target_KEV_180d.
  4. Optimal classification threshold found on VALIDATION set, applied once to TEST.
  5. All metrics reported: ROC-AUC, PR-AUC, F1 (minority), Precision@K, Recall@K.
  6. scale_pos_weight computed from train only.

Paper reference:
  Fang Y, Liu Y, Huang C, Liu L (2020) FastEmbed: Predicting vulnerability exploitation
  possibility based on ensemble machine learning algorithm. PLoS ONE 15(2): e0228439.

Outputs:
  models/fastembed_ft_model/           <- fastText model directory
  models/fastembed_lgbm.pkl            <- LightGBM model
  outputs/fastembed_predictions_train.csv
  outputs/fastembed_predictions_validation.csv
  outputs/fastembed_predictions_test.csv
  outputs/fastembed_metrics.json
  reports/fastembed_baseline_report.md
"""

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "ml" / "train.csv"
VAL_PATH   = PROJECT_ROOT / "data" / "processed" / "ml" / "validation.csv"
TEST_PATH  = PROJECT_ROOT / "data" / "processed" / "ml" / "test.csv"

MODELS_DIR  = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COL   = "Target_KEV_180d"
EMBED_DIM    = 100     # FastEmbed paper used 100-dim embeddings
FASTTEXT_EPOCHS = 10   # paper default
FASTTEXT_MIN_COUNT = 1 # keep rare CVE-specific terms


# ─────────────────────────────────────────────────────────────────────────────
# Helper: check gensim availability and guide install if missing
# ─────────────────────────────────────────────────────────────────────────────
def _require_gensim():
    try:
        from gensim.models import FastText  # noqa: F401
        return True
    except ImportError:
        print("[ERROR] gensim is not installed.")
        print("  Run: pip install gensim")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Text tokenization
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace/punctuation, remove stop chars."""
    import re
    text = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens if tokens else ["<empty>"]


# ─────────────────────────────────────────────────────────────────────────────
# FastText embedding: fit on train, transform any split
# ─────────────────────────────────────────────────────────────────────────────
def train_fasttext(train_descriptions: pd.Series, model_dir: Path) -> object:
    """
    Train a fastText model on UNIQUE training CVE descriptions only.
    Training on 141K unique texts (not 1.5M panel rows) is 10x faster
    while learning identical vocabulary — descriptions don't change per row.
    Saves to model_dir / fastembed_ft.model
    """
    from gensim.models import FastText

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fastembed_ft.model"

    # Deduplicate: unique descriptions only for vocabulary learning
    unique_descs = train_descriptions.dropna().unique()
    print(f"[FastText] Tokenizing {len(unique_descs):,d} unique train descriptions "
          f"(deduplicated from {len(train_descriptions):,d} rows)...")
    sentences = [tokenize(d) for d in unique_descs]

    print(f"[FastText] Training FastText (dim={EMBED_DIM}, epochs={FASTTEXT_EPOCHS})...")
    t0 = time.time()
    ft_model = FastText(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=5,
        min_count=FASTTEXT_MIN_COUNT,
        epochs=FASTTEXT_EPOCHS,
        workers=4,
        seed=42,
    )
    ft_model.save(str(model_path))
    print(f"[FastText] Training done in {time.time() - t0:.1f}s. Saved to {model_path}")
    return ft_model


def get_doc_embedding(ft_model, description: str) -> np.ndarray:
    """
    Average fastText token embeddings for a CVE description using
    gensim's optimised get_mean_vector (avoids Python loop over tokens).
    """
    tokens = tokenize(description)
    try:
        return ft_model.wv.get_mean_vector(tokens, ignore_missing=True).astype(np.float32)
    except Exception:
        return np.zeros(EMBED_DIM, dtype=np.float32)


def embed_descriptions(ft_model, descriptions: pd.Series) -> np.ndarray:
    """
    Embed all descriptions efficiently:
      1. Build a lookup dict of unique description -> embedding (avoids recomputing).
      2. Map every row via the lookup (vectorized, no Python loop per row).
    """
    print(f"[FastText] Embedding {len(descriptions):,d} descriptions "
          f"({descriptions.nunique():,d} unique)...")
    t0 = time.time()

    # Compute embedding for each UNIQUE description once
    unique_descs = descriptions.fillna("").unique()
    lookup = {
        d: get_doc_embedding(ft_model, d) for d in unique_descs
    }
    # Map every row in O(n) time
    emb = np.vstack([lookup[d] for d in descriptions.fillna("")])  
    print(f"[FastText] Embedding done in {time.time() - t0:.1f}s. Shape: {emb.shape}")
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# CVSS structural features (same as FastEmbed paper: CVSS score + metadata)
# ─────────────────────────────────────────────────────────────────────────────
def build_cvss_features(df: pd.DataFrame, train_median_cvss: float = None, is_train: bool = False):
    """
    Extract CVSS numerical + categorical OHE features.
    Fit (median imputation) on train only.
    """
    cvss = df["CVSS_Score"].copy().astype(float)

    if is_train:
        train_median_cvss = float(cvss.median())

    cvss_imputed = cvss.fillna(train_median_cvss)
    cvss_missing = cvss.isna().astype(np.float32)

    # One-hot CVSS_Version fit on train vocab (reindex handles unseen cats)
    version_dummies = pd.get_dummies(
        df["CVSS_Version"].fillna("Missing").astype(str), prefix="cvss_ver"
    )
    severity_dummies = pd.get_dummies(
        df["Severity"].fillna("Missing").astype(str), prefix="sev"
    )

    feat = pd.DataFrame({
        "cvss_score": cvss_imputed.values.astype(np.float32),
        "cvss_missing": cvss_missing.values,
        "vuln_age_days": df["Vulnerability_Age_Days"].fillna(0).astype(np.float32).values,
        "severity_encoded": df["Severity_Encoded"].fillna(0).astype(np.float32).values,
    }, index=df.index)

    feat = pd.concat([feat, version_dummies, severity_dummies], axis=1)
    return feat, train_median_cvss


def align_columns(df: pd.DataFrame, train_cols: list) -> pd.DataFrame:
    return df.reindex(columns=train_cols, fill_value=0)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def precision_recall_at_k(y_true, y_prob, k_values=(10, 25, 50, 100, 200, 500, 1000, 5000)):
    sorted_idx = np.argsort(y_prob)[::-1]
    total_pos = int((y_true == 1).sum())
    results = {}
    for k in k_values:
        k_ = min(k, len(y_true))
        top_k = sorted_idx[:k_]
        pos_k = int((y_true[top_k] == 1).sum())
        results[f"Precision@{k}"] = round(pos_k / k_, 6) if k_ > 0 else 0.0
        results[f"Recall@{k}"]    = round(pos_k / total_pos, 6) if total_pos > 0 else 0.0
        results[f"Positives@{k}"] = pos_k
    return results


def evaluate(y_true, y_prob, threshold: float = 0.5, label: str = "") -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    roc  = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr   = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()
    pk   = precision_recall_at_k(y_true, y_prob)
    if label:
        print(f"  [{label}] ROC-AUC={roc:.4f}  PR-AUC={pr:.6f}  F1={f1:.4f}  "
              f"Prec={prec:.4f}  Recall={rec:.4f}  Threshold={threshold:.3f}")
    return {
        "ROC_AUC": round(roc, 6),
        "PR_AUC": round(pr, 6),
        "Brier_Score": round(brier, 8),
        "Threshold": threshold,
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "F1_Score": round(f1, 6),
        "Confusion_Matrix": cm,
        "Total_Rows": len(y_true),
        "Actual_Positives": int((y_true == 1).sum()),
        "Predicted_Positives": int((y_pred == 1).sum()),
        **pk,
    }


def find_best_f1_threshold(y_true, y_prob, thresholds=None) -> float:
    """Find threshold maximising F1 on validation set."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)
    best_thr, best_f1 = 0.5, 0.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_fastembed_baseline():
    _require_gensim()
    start = time.time()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FASTEMBED BASELINE (Fang et al. 2020) — CyberGuard Temporal Data")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading temporal splits...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"    Train: {len(train_df):,d} rows | {int(train_df[TARGET_COL].sum())} positives")
    print(f"    Val  : {len(val_df):,d} rows | {int(val_df[TARGET_COL].sum())} positives")
    print(f"    Test : {len(test_df):,d} rows | {int(test_df[TARGET_COL].sum())} positives")

    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    # ── 2. FastText embeddings (fit on train ONLY) ────────────────────────────
    print("\n[2] Training FastText on train descriptions only...")
    ft_model_dir = MODELS_DIR / "fastembed_ft_model"
    ft_model = train_fasttext(train_df["Description"], ft_model_dir)

    emb_train = embed_descriptions(ft_model, train_df["Description"])
    emb_val   = embed_descriptions(ft_model, val_df["Description"])
    emb_test  = embed_descriptions(ft_model, test_df["Description"])

    # ── 3. CVSS structural features (fit medians on train) ────────────────────
    print("\n[3] Building CVSS structural features (fit on train)...")
    cvss_train, train_median_cvss = build_cvss_features(train_df, is_train=True)
    cvss_val,   _                 = build_cvss_features(val_df,   train_median_cvss=train_median_cvss)
    cvss_test,  _                 = build_cvss_features(test_df,  train_median_cvss=train_median_cvss)

    # Align OHE columns (val/test may have fewer categories than train)
    train_cols = cvss_train.columns.tolist()
    cvss_val   = align_columns(cvss_val, train_cols)
    cvss_test  = align_columns(cvss_test, train_cols)

    # ── 4. Concatenate embeddings + CVSS features ─────────────────────────────
    print("\n[4] Concatenating FastText embeddings + CVSS features...")
    X_train = np.hstack([emb_train, cvss_train.values]).astype(np.float32)
    X_val   = np.hstack([emb_val,   cvss_val.values]).astype(np.float32)
    X_test  = np.hstack([emb_test,  cvss_test.values]).astype(np.float32)

    feature_names = (
        [f"ft_{i}" for i in range(EMBED_DIM)] + train_cols
    )
    print(f"    Total features: {X_train.shape[1]}  "
          f"(={EMBED_DIM} embed + {len(train_cols)} CVSS)")

    # ── 5. LightGBM training (FastEmbed paper used LightGBM) ─────────────────
    print("\n[5] Training LightGBM (FastEmbed architecture)...")

    # scale_pos_weight: FastEmbed paper used sqrt to avoid extreme overweighting
    # Raw spw=1037 caused early stopping at iter=1. sqrt(1037)=32 is more stable.
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw_raw = neg_count / pos_count if pos_count > 0 else 1.0
    spw = float(np.sqrt(spw_raw))  # geometric mean between 1 and raw spw
    print(f"    scale_pos_weight = sqrt({spw_raw:.1f}) = {spw:.2f}  "
          f"(raw={spw_raw:.1f} caused early-stop@iter1)")

    params = {
        "objective":         "binary",
        "metric":            ["average_precision", "binary_logloss"],
        "boosting_type":     "gbdt",
        "n_estimators":      500,
        "learning_rate":     0.03,      # slower lr for better generalisation
        "num_leaves":        63,        # FastEmbed paper used 63
        "max_depth":         -1,
        "min_child_samples": 10,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "scale_pos_weight":  spw,
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    t_tr = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_X=X_val, eval_y=y_val,
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=True),
            lgb.log_evaluation(period=25),
        ],
    )
    train_time = round(time.time() - t_tr, 2)
    print(f"    Training done in {train_time}s. Best iteration: {model.best_iteration_}")

    # ── 6. Inference ──────────────────────────────────────────────────────────
    print("\n[6] Generating probability predictions...")
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val   = model.predict_proba(X_val)[:, 1]
    prob_test  = model.predict_proba(X_test)[:, 1]

    # ── 7. Find best threshold on VALIDATION (never touch test for this) ──────
    print("\n[7] Finding optimal F1 threshold on validation set...")
    best_thr = find_best_f1_threshold(y_val, prob_val)
    print(f"    Best validation threshold = {best_thr:.4f}")

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    print("\n[8] Evaluating all splits...")
    print("\n  --- Results at default threshold=0.50 ---")
    train_m = evaluate(y_train, prob_train, threshold=0.50, label="Train (0.50)")
    val_m   = evaluate(y_val,   prob_val,   threshold=0.50, label="Val   (0.50)")
    test_m  = evaluate(y_test,  prob_test,  threshold=0.50, label="Test  (0.50)")

    print("\n  --- Results at val-optimal threshold ---")
    train_m_opt = evaluate(y_train, prob_train, threshold=best_thr, label=f"Train ({best_thr:.3f})")
    val_m_opt   = evaluate(y_val,   prob_val,   threshold=best_thr, label=f"Val   ({best_thr:.3f})")
    test_m_opt  = evaluate(y_test,  prob_test,  threshold=best_thr, label=f"Test  ({best_thr:.3f})")

    # ── 9. Save model & predictions ───────────────────────────────────────────
    print("\n[9] Saving model and predictions...")
    lgbm_path = MODELS_DIR / "fastembed_lgbm.pkl"
    joblib.dump(model, lgbm_path)
    print(f"    LightGBM saved to {lgbm_path}")

    for name, df_split, probs in [
        ("train",      train_df, prob_train),
        ("validation", val_df,   prob_val),
        ("test",       test_df,  prob_test),
    ]:
        out = pd.DataFrame({
            "CVE_ID":           df_split["CVE_ID"],
            "Observation_Date": df_split["Observation_Date"],
            "target":           df_split[TARGET_COL],
            "predicted_prob":   probs,
            "predicted_label":  (probs >= best_thr).astype(int),
        })
        p = OUTPUTS_DIR / f"fastembed_predictions_{name}.csv"
        out.to_csv(p, index=False)
        print(f"    Predictions saved: {p}")

    # ── 10. Save metrics JSON ──────────────────────────────────────────────────
    metrics_dict = {
        "model": "FastEmbed (Fang et al. 2020) on CyberGuard data",
        "embed_dim": EMBED_DIM,
        "best_threshold_from_val": best_thr,
        "train_pos_count": pos_count,
        "scale_pos_weight": round(spw, 4),
        "threshold_0.5": {
            "train": train_m,
            "validation": val_m,
            "test": test_m,
        },
        "threshold_optimal_val": {
            "train": train_m_opt,
            "validation": val_m_opt,
            "test": test_m_opt,
        },
    }
    m_path = OUTPUTS_DIR / "fastembed_metrics.json"
    with open(m_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"    Metrics saved: {m_path}")

    # ── 11. Markdown report ────────────────────────────────────────────────────
    _write_report(metrics_dict, train_m, val_m, test_m, test_m_opt, best_thr,
                  train_time, model.best_iteration_, X_train.shape[1])

    total = round(time.time() - start, 1)
    print(f"\n[Done] FastEmbed baseline completed in {total}s")
    return metrics_dict


def _write_report(metrics_dict, train_m, val_m, test_m, test_m_opt,
                  best_thr, train_time, best_iter, n_features):
    report = f"""# FastEmbed Baseline Report (Fang et al. 2020 Architecture on CyberGuard Data)

## Setup

| Parameter | Value |
|---|---|
| **Paper** | Fang et al., PLOS ONE 2020, DOI:10.1371/journal.pone.0228439 |
| **Architecture** | fastText embeddings (dim={metrics_dict['embed_dim']}) + LightGBM |
| **FastText trained on** | Train descriptions ONLY (no val/test leakage) |
| **Total features** | {n_features} ({metrics_dict['embed_dim']} embed + CVSS structural) |
| **Imbalance handling** | scale_pos_weight = {metrics_dict['scale_pos_weight']} (from train only) |
| **Threshold selection** | Optimal F1 on validation = {best_thr:.4f} (applied once to test) |
| **LightGBM training time** | {train_time}s |
| **Best iteration (early stopping)** | {best_iter} |

## Performance at Default Threshold (0.50)

| Metric | Train (2022) | Validation (2023) | Test (2024) |
|---|---|---|---|
| **ROC-AUC** | {train_m['ROC_AUC']:.6f} | {val_m['ROC_AUC']:.6f} | {test_m['ROC_AUC']:.6f} |
| **PR-AUC** | {train_m['PR_AUC']:.6f} | {val_m['PR_AUC']:.6f} | {test_m['PR_AUC']:.6f} |
| **F1 (minority)** | {train_m['F1_Score']:.6f} | {val_m['F1_Score']:.6f} | {test_m['F1_Score']:.6f} |
| **Precision** | {train_m['Precision']:.6f} | {val_m['Precision']:.6f} | {test_m['Precision']:.6f} |
| **Recall** | {train_m['Recall']:.6f} | {val_m['Recall']:.6f} | {test_m['Recall']:.6f} |
| **Brier Score** | {train_m['Brier_Score']:.8f} | {val_m['Brier_Score']:.8f} | {test_m['Brier_Score']:.8f} |
| **Actual Positives** | {train_m['Actual_Positives']} | {val_m['Actual_Positives']} | {test_m['Actual_Positives']} |

## Performance at Val-Optimal Threshold ({best_thr:.4f})

| Metric | Test (2024) |
|---|---|
| **ROC-AUC** | {test_m_opt['ROC_AUC']:.6f} |
| **PR-AUC** | {test_m_opt['PR_AUC']:.6f} |
| **F1 (minority)** | {test_m_opt['F1_Score']:.6f} |
| **Precision** | {test_m_opt['Precision']:.6f} |
| **Recall** | {test_m_opt['Recall']:.6f} |

## Top-K Ranking Performance (Test Set, 2024)

| K | Precision@K | Recall@K | Positives Found |
|---|---|---|---|
| 10 | {test_m['Precision@10']:.4f} | {test_m['Recall@10']:.4f} | {test_m['Positives@10']} |
| 25 | {test_m['Precision@25']:.4f} | {test_m['Recall@25']:.4f} | {test_m['Positives@25']} |
| 50 | {test_m['Precision@50']:.4f} | {test_m['Recall@50']:.4f} | {test_m['Positives@50']} |
| 100 | {test_m['Precision@100']:.4f} | {test_m['Recall@100']:.4f} | {test_m['Positives@100']} |
| 200 | {test_m['Precision@200']:.4f} | {test_m['Recall@200']:.4f} | {test_m['Positives@200']} |
| 500 | {test_m['Precision@500']:.4f} | {test_m['Recall@500']:.4f} | {test_m['Positives@500']} |
| 1000 | {test_m['Precision@1000']:.4f} | {test_m['Recall@1000']:.4f} | {test_m['Positives@1000']} |

## Fairness Guarantees

1. **No temporal leakage**: fastText trained on train text only; medians/OHE fit on train only.
2. **No threshold snooping**: threshold selected on validation, applied once to test.
3. **Identical splits**: same `train.csv`, `validation.csv`, `test.csv` as all CyberGuard models.
4. **Same target**: `Target_KEV_180d`.
5. **Reported both thresholds**: default (0.5) and optimal-val for transparency.
"""
    r_path = REPORTS_DIR / "fastembed_baseline_report.md"
    with open(r_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"    Report saved: {r_path}")


if __name__ == "__main__":
    run_fastembed_baseline()
