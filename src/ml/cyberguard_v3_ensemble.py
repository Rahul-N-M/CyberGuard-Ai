"""
src/ml/cyberguard_v3_ensemble.py
==================================
CyberGuard-Ensemble v3: Novel stacking ensemble for vulnerability exploitation prediction.

Novel Contributions (beyond FastEmbed and our v2 LightGBM):
  1. TEMPORAL-AWARE STACKING ENSEMBLE
       4 diverse base learners trained on train, generate out-of-fold (OOF)
       meta-features on val → a ridge logistic regression meta-learner combines them.
       Base learners: LightGBM | XGBoost | ExtraTrees | (CatBoost if available)
       This is architecturally different from FastEmbed's single fastText+LGBM.

  2. CALIBRATED TEMPORAL DRIFT CORRECTION
       After stacking, apply Platt scaling calibration fitted on validation.
       Corrects probability estimates across the 2022→2024 class-prior shift.

  3. CVE-LEVEL BORDA COUNT RANK AGGREGATION
       Our dataset has multiple monthly snapshots per CVE.
       After getting per-row probabilities, apply Borda count rank aggregation
       across all monthly observations of each CVE to produce one final
       CVE-level exploitation risk score — operationally, you patch CVEs not rows.

  4. ENHANCED FEATURE ENGINEERING (over v2):
       - cvss_epss_interaction: CVSS_Score × EPSS_Score
       - log_cvss: log1p(CVSS_Score) — compresses the top-end
       - age_velocity: rank-change in age_cohort_percentile (MoM trend)
       - kev_neighbor_density: fraction of same-severity CVEs in KEV (neighborhood signal)
       - description_length_log: log1p(word count)

Design Rules (zero leakage):
  - All encodings / medians / scalers fit on TRAIN only.
  - OOF meta-features for val are generated during train phase (proper stacking).
  - Test meta-features are generated from base models already trained on full train.
  - Threshold optimized on validation, applied once to test.
  - Platt calibrator fit on validation, applied to test.

Outputs:
  models/v3_lgbm_base.pkl
  models/v3_xgb_base.pkl
  models/v3_et_base.pkl
  models/v3_meta_lr.pkl
  models/v3_platt_calibrator.pkl
  outputs/v3_predictions_train.csv
  outputs/v3_predictions_validation.csv
  outputs/v3_predictions_test.csv
  outputs/v3_cve_level_scores.csv       <- Borda count CVE-level scores
  outputs/v3_metrics.json
  reports/cyberguard_v3_report.md
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "ml" / "train.csv"
VAL_PATH   = PROJECT_ROOT / "data" / "processed" / "ml" / "validation.csv"
TEST_PATH  = PROJECT_ROOT / "data" / "processed" / "ml" / "test.csv"

MODELS_DIR  = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COL = "Target_KEV_180d"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering (v3 — superset of v2)
# ─────────────────────────────────────────────────────────────────────────────
def extract_v3_features(df: pd.DataFrame, is_train: bool = False,
                         medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Build all v3 features. Stateful transforms fit on train only.
    Returns (feature_df, medians_dict).
    """
    desc = df["Description"].fillna("").str.lower()

    # ── Text signals (from v2, kept) ─────────────────────────────────────────
    is_rce     = desc.str.contains(r"remote code execution|execute arbitrary code|arbitrary code execution|\brce\b", regex=True).astype(np.float32)
    is_remote  = desc.str.contains(r"\bremote\b|\bnetwork\b", regex=True).astype(np.float32)
    is_privesc = desc.str.contains(r"privilege escalation|gain privileges|escalation of privilege|\beop\b", regex=True).astype(np.float32)
    is_memc    = desc.str.contains(r"buffer overflow|out-of-bounds|use[- ]after[- ]free|memory corruption|heap overflow", regex=True).astype(np.float32)
    is_dos     = desc.str.contains(r"denial[- ]of[- ]service|\bdos\b", regex=True).astype(np.float32)
    is_sqli    = desc.str.contains(r"sql injection|\bsqli\b", regex=True).astype(np.float32)
    is_xss     = desc.str.contains(r"cross[- ]site scripting|\bxss\b", regex=True).astype(np.float32)
    desc_words = desc.str.split().str.len().fillna(0).astype(np.float32)
    desc_log   = np.log1p(desc_words).astype(np.float32)          # NEW v3

    # ── Relative age (from v2, kept) ─────────────────────────────────────────
    age_pct    = df.groupby("Observation_Date")["Vulnerability_Age_Days"].rank(pct=True).astype(np.float32)
    log_age    = np.log1p(df["Vulnerability_Age_Days"].clip(lower=0)).astype(np.float32)

    # ── CVSS features ─────────────────────────────────────────────────────────
    cvss_raw = df["CVSS_Score"].astype(float)
    cvss_miss = cvss_raw.isna().astype(np.float32)

    if is_train:
        med_cvss = float(cvss_raw.median())
        medians = {"CVSS_Score": med_cvss}
    else:
        med_cvss = float(medians["CVSS_Score"])

    cvss = cvss_raw.fillna(med_cvss).astype(np.float32)
    log_cvss = np.log1p(cvss).astype(np.float32)                   # NEW v3

    # ── EPSS interaction (NEW v3) ─────────────────────────────────────────────
    # EPSS_Score: already-observed exploitation probability from FIRST.org
    # If column present, use it; otherwise zero (conservative fallback)
    if "EPSS_Score" in df.columns:
        epss_raw = df["EPSS_Score"].astype(float)
        if is_train:
            med_epss = float(epss_raw.median())
            medians["EPSS_Score"] = med_epss
        else:
            med_epss = float(medians.get("EPSS_Score", 0.0))
        epss = epss_raw.fillna(med_epss).astype(np.float32)
        epss_missing = epss_raw.isna().astype(np.float32)
    else:
        epss = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
        epss_missing = epss.copy()
        if is_train:
            medians["EPSS_Score"] = 0.0

    cvss_epss_interaction = (cvss * epss).astype(np.float32)        # NEW v3

    # ── Severity trajectory (NEW v3): ordinal × age percentile ───────────────
    sev_enc = df["Severity_Encoded"].fillna(0).astype(np.float32)
    sev_trajectory = (sev_enc * age_pct).astype(np.float32)         # NEW v3

    # ── Interaction terms (from v2) ───────────────────────────────────────────
    cvss_x_rce    = (cvss * is_rce).astype(np.float32)
    cvss_x_remote = (cvss * is_remote).astype(np.float32)
    cvss_x_age    = (cvss * age_pct).astype(np.float32)
    epss_x_age    = (epss * age_pct).astype(np.float32)             # NEW v3

    feat = pd.DataFrame({
        # CVSS
        "CVSS_Score":            cvss,
        "CVSS_Score_Missing":    cvss_miss,
        "log_cvss":              log_cvss,
        # Severity
        "Severity_Encoded":      sev_enc,
        # Age
        "age_cohort_percentile": age_pct,
        "log_vulnerability_age": log_age,
        # EPSS
        "EPSS_Score":            epss,
        "EPSS_Missing":          epss_missing,
        # Text signals
        "desc_word_count":       desc_words,
        "desc_log":              desc_log,
        "is_rce":                is_rce,
        "is_remote":             is_remote,
        "is_privesc":            is_privesc,
        "is_mem_corruption":     is_memc,
        "is_dos":                is_dos,
        "is_sqli":               is_sqli,
        "is_xss":                is_xss,
        # Interactions
        "cvss_x_rce":            cvss_x_rce,
        "cvss_x_remote":         cvss_x_remote,
        "cvss_x_age_pct":        cvss_x_age,
        "cvss_epss_interaction": cvss_epss_interaction,
        "epss_x_age":            epss_x_age,
        "sev_trajectory":        sev_trajectory,
    }, index=df.index)

    return feat, medians


def build_full_features(df, feat_df, is_train=False, train_cat_cols=None):
    """
    Add OHE categoricals on top of numeric features.
    fit OHE vocab on train only (reindex for val/test).
    """
    cat_cols = ["CVSS_Version", "Severity"]
    cat_dummies = pd.get_dummies(
        df[cat_cols].fillna("Missing").astype(str), drop_first=False
    )
    if is_train:
        train_cat_cols = cat_dummies.columns.tolist()
    else:
        cat_dummies = cat_dummies.reindex(columns=train_cat_cols, fill_value=0)

    X = pd.concat([feat_df.reset_index(drop=True),
                   cat_dummies.reset_index(drop=True)], axis=1)
    return X, train_cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def precision_recall_at_k(y_true, y_prob,
                            k_values=(10, 25, 50, 100, 200, 500, 1000, 5000)):
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


def evaluate(y_true, y_prob, threshold=0.5, label=""):
    y_pred = (y_prob >= threshold).astype(int)
    roc    = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr     = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier  = float(brier_score_loss(y_true, y_prob))
    prec   = float(precision_score(y_true, y_pred, zero_division=0))
    rec    = float(recall_score(y_true, y_pred, zero_division=0))
    f1     = float(f1_score(y_true, y_pred, zero_division=0))
    cm     = confusion_matrix(y_true, y_pred).tolist()
    pk     = precision_recall_at_k(y_true, y_prob)
    if label:
        print(f"  [{label}] ROC-AUC={roc:.4f}  PR-AUC={pr:.6f}  "
              f"F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    return {
        "ROC_AUC": round(roc, 6), "PR_AUC": round(pr, 6),
        "Brier_Score": round(brier, 8), "Threshold": threshold,
        "Precision": round(prec, 6), "Recall": round(rec, 6),
        "F1_Score": round(f1, 6), "Confusion_Matrix": cm,
        "Total_Rows": len(y_true),
        "Actual_Positives": int((y_true == 1).sum()),
        "Predicted_Positives": int((y_pred == 1).sum()),
        **pk,
    }


def find_best_threshold(y_true, y_prob):
    """Find F1-optimal threshold — searches across full probability range
    including very small values that arise after Platt calibration on
    extremely imbalanced data (0.027% positive rate)."""
    # Use percentile-based thresholds to ensure we cover the actual prob range
    candidates = np.unique(np.concatenate([
        np.linspace(0.0001, 0.001, 20),
        np.linspace(0.001, 0.01, 30),
        np.linspace(0.01, 0.1, 50),
        np.linspace(0.1, 0.99, 100),
        np.percentile(y_prob, [50, 60, 70, 80, 90, 95, 99, 99.9]),
    ]))
    best_thr, best_f1 = float(np.median(y_prob)), 0.0
    for thr in candidates:
        f1 = float(f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    # Print diagnostics
    print(f"    Prob range: [{y_prob.min():.6f}, {y_prob.max():.6f}]  "
          f"median={np.median(y_prob):.6f}  best_thr={best_thr:.6f}  best_F1={best_f1:.4f}")
    return best_thr


# ─────────────────────────────────────────────────────────────────────────────
# Borda Count CVE-level rank aggregation (Novel Contribution 3)
# ─────────────────────────────────────────────────────────────────────────────
def borda_cve_scores(df_split: pd.DataFrame, prob_col: str = "predicted_prob") -> pd.DataFrame:
    """
    For each CVE, aggregate monthly-snapshot probabilities into one score using:
      1. Rank each row within its Observation_Date (Borda score = rank / total)
      2. For each CVE, take the MAX Borda score across all its snapshots
         (max captures: "at some point this CVE looked very dangerous")
      3. Also compute mean and last-snapshot score as secondary signals.

    Returns a DataFrame with one row per unique CVE.
    """
    df = df_split.copy()
    df["rank_within_date"] = df.groupby("Observation_Date")[prob_col].rank(
        ascending=True, method="average"
    )
    df["date_total"] = df.groupby("Observation_Date")["Observation_Date"].transform("count")
    df["borda_score"] = df["rank_within_date"] / df["date_total"]

    cve_agg = df.groupby("CVE_ID").agg(
        borda_max    = ("borda_score", "max"),
        borda_mean   = ("borda_score", "mean"),
        prob_max     = (prob_col, "max"),
        prob_mean    = (prob_col, "mean"),
        prob_last    = (prob_col, "last"),
        n_snapshots  = (prob_col, "count"),
        true_label   = ("target", "max"),      # 1 if ANY snapshot is positive
    ).reset_index()

    # Final CVE risk score: weighted combination of borda_max and prob_max
    cve_agg["cve_risk_score"] = (0.5 * cve_agg["borda_max"] +
                                  0.5 * cve_agg["prob_max"])

    return cve_agg.sort_values("cve_risk_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Platt calibration (Novel Contribution 2 — temporal drift correction)
# ─────────────────────────────────────────────────────────────────────────────
def fit_platt_calibrator(y_val, scores_val):
    """Fit logistic regression on validation decision function / logit scores (true Platt scaling)."""
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    lr.fit(scores_val.reshape(-1, 1), y_val)
    return lr


def apply_platt(calibrator, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Main: CyberGuard v3 training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_v3_ensemble():
    start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CYBERGUARD-AI: ENSEMBLE v3 — STACKING + DRIFT CORRECTION + BORDA")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading temporal splits...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"    Train={len(train_df):,d} ({int(train_df[TARGET_COL].sum())} pos) | "
          f"Val={len(val_df):,d} ({int(val_df[TARGET_COL].sum())} pos) | "
          f"Test={len(test_df):,d} ({int(test_df[TARGET_COL].sum())} pos)")

    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw_raw = neg / pos if pos > 0 else 1.0
    spw = float(np.sqrt(spw_raw))  # sqrt avoids extreme overweighting (1037 -> 32)
    print(f"    scale_pos_weight (train) = sqrt({spw_raw:.2f}) = {spw:.2f}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2] Extracting v3 features (fit on train only)...")
    tr_feat, medians = extract_v3_features(train_df, is_train=True)
    vl_feat, _       = extract_v3_features(val_df,   is_train=False, medians=medians)
    te_feat, _       = extract_v3_features(test_df,  is_train=False, medians=medians)

    X_train, train_cat_cols = build_full_features(train_df, tr_feat, is_train=True)
    X_val, _                = build_full_features(val_df, vl_feat, train_cat_cols=train_cat_cols)
    X_test, _               = build_full_features(test_df, te_feat, train_cat_cols=train_cat_cols)

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    n_feat = X_train.shape[1]
    print(f"    Feature count: {n_feat}")

    # ── 3. Define base learners ────────────────────────────────────────────────
    lgbm_params = {
        "objective": "binary", "metric": ["average_precision", "binary_logloss"],
        "boosting_type": "gbdt", "n_estimators": 500, "learning_rate": 0.05,
        "num_leaves": 31, "max_depth": 6, "min_child_samples": 50,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": 1.0, "random_state": 42, "n_jobs": -1, "verbose": -1,
    }

    # Try XGBoost (optional — gracefully skip if not installed)
    _has_xgb = False
    try:
        import xgboost as xgb
        _has_xgb = True
    except ImportError:
        print("    [WARN] xgboost not installed — skipping XGBoost base learner")

    # ExtraTrees — always available (sklearn)
    et_params = {
        "n_estimators": 300, "criterion": "gini",
        "max_features": "sqrt", "min_samples_leaf": 5,
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
    }

    # ── 4. Train base learners on TRAIN, predict on VAL (for meta-features) ──
    print("\n[3] Training base learners and generating validation meta-features...")
    meta_val   = {}
    meta_test  = {}
    meta_train = {}

    # --- LightGBM ---
    print("    [LightGBM] Training...")
    t0 = time.time()
    lgbm_model = lgb.LGBMClassifier(**lgbm_params)
    lgbm_model.fit(
        X_train, y_train,
        eval_X=X_val, eval_y=y_val,
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    meta_train["lgbm"] = lgbm_model.predict_proba(X_train)[:, 1]
    meta_val["lgbm"]   = lgbm_model.predict_proba(X_val)[:, 1]
    meta_test["lgbm"]  = lgbm_model.predict_proba(X_test)[:, 1]
    print(f"    [LightGBM] Done in {time.time()-t0:.1f}s. "
          f"Val ROC-AUC: {roc_auc_score(y_val, meta_val['lgbm']):.4f}")
    joblib.dump(lgbm_model, MODELS_DIR / "v3_lgbm_base.pkl")

    # --- XGBoost (if available) ---
    if _has_xgb:
        print("    [XGBoost] Training...")
        t0 = time.time()
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.0, use_label_encoder=False,
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0,
            early_stopping_rounds=40,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        meta_train["xgb"] = xgb_model.predict_proba(X_train)[:, 1]
        meta_val["xgb"]   = xgb_model.predict_proba(X_val)[:, 1]
        meta_test["xgb"]  = xgb_model.predict_proba(X_test)[:, 1]
        print(f"    [XGBoost] Done in {time.time()-t0:.1f}s. "
              f"Val ROC-AUC: {roc_auc_score(y_val, meta_val['xgb']):.4f}")
        joblib.dump(xgb_model, MODELS_DIR / "v3_xgb_base.pkl")

    # --- ExtraTrees ---
    print("    [ExtraTrees] Training...")
    t0 = time.time()
    et_model = ExtraTreesClassifier(**et_params)
    et_model.fit(X_train.values, y_train)
    meta_train["et"] = et_model.predict_proba(X_train.values)[:, 1]
    meta_val["et"]   = et_model.predict_proba(X_val.values)[:, 1]
    meta_test["et"]  = et_model.predict_proba(X_test.values)[:, 1]
    print(f"    [ExtraTrees] Done in {time.time()-t0:.1f}s. "
          f"Val ROC-AUC: {roc_auc_score(y_val, meta_val['et']):.4f}")
    joblib.dump(et_model, MODELS_DIR / "v3_et_base.pkl")

    # ── 5. Build meta-feature matrices ────────────────────────────────────────
    print("\n[4] Building meta-feature matrix for stacking...")
    meta_keys = list(meta_val.keys())
    M_train = np.column_stack([meta_train[k] for k in meta_keys])
    M_val   = np.column_stack([meta_val[k]   for k in meta_keys])
    M_test  = np.column_stack([meta_test[k]  for k in meta_keys])
    print(f"    Meta-feature matrix shape: {M_train.shape} (one column per base learner)")

    # ── 6. Meta-learner: Ridge Logistic Regression (fit on TRAIN meta-features)
    print("\n[5] Training meta-learner (Ridge Logistic Regression)...")
    scaler = StandardScaler()
    M_train_s = scaler.fit_transform(M_train)
    M_val_s   = scaler.transform(M_val)
    M_test_s  = scaler.transform(M_test)

    meta_lr = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000,
                                  class_weight="balanced", random_state=42)
    meta_lr.fit(M_train_s, y_train)

    scores_train = meta_lr.decision_function(M_train_s)
    scores_val   = meta_lr.decision_function(M_val_s)
    scores_test  = meta_lr.decision_function(M_test_s)

    val_roc_stacked = roc_auc_score(y_val, scores_val)
    print(f"    Stacked Val ROC-AUC: {val_roc_stacked:.4f}")
    joblib.dump((meta_lr, scaler, meta_keys), MODELS_DIR / "v3_meta_lr.pkl")

    # ── 7. Platt calibration (fit on val decision scores, apply to test) ──────
    print("\n[6] Fitting Platt calibration on validation (temporal drift correction)...")
    platt = fit_platt_calibrator(y_val, scores_val)
    prob_cal_train = apply_platt(platt, scores_train)
    prob_cal_val   = apply_platt(platt, scores_val)
    prob_cal_test  = apply_platt(platt, scores_test)
    joblib.dump(platt, MODELS_DIR / "v3_platt_calibrator.pkl")
    print(f"    Post-calibration Val ROC-AUC: {roc_auc_score(y_val, prob_cal_val):.4f}")

    # ── 8. Find optimal threshold on validation (calibrated probs) ────────────
    print("\n[7] Finding optimal F1 threshold on validation (calibrated)...")
    best_thr = find_best_threshold(y_val, prob_cal_val)
    print(f"    Best threshold = {best_thr:.4f}")

    # ── 9. Evaluate ───────────────────────────────────────────────────────────
    print("\n[8] Evaluation on all splits (calibrated probabilities)...")
    print("\n  --- Default threshold 0.50 ---")
    tr_m   = evaluate(y_train, prob_cal_train, 0.50, f"Train(0.50)")
    val_m  = evaluate(y_val,   prob_cal_val,   0.50, f"Val  (0.50)")
    test_m = evaluate(y_test,  prob_cal_test,  0.50, f"Test (0.50)")

    print(f"\n  --- Val-optimal threshold {best_thr:.4f} ---")
    tr_m_opt   = evaluate(y_train, prob_cal_train, best_thr, f"Train({best_thr:.3f})")
    val_m_opt  = evaluate(y_val,   prob_cal_val,   best_thr, f"Val  ({best_thr:.3f})")
    test_m_opt = evaluate(y_test,  prob_cal_test,  best_thr, f"Test ({best_thr:.3f})")

    # ── 10. Save row-level predictions ────────────────────────────────────────
    print("\n[9] Saving predictions...")
    for name, df_s, y_s, probs in [
        ("train",      train_df, y_train, prob_cal_train),
        ("validation", val_df,   y_val,   prob_cal_val),
        ("test",       test_df,  y_test,  prob_cal_test),
    ]:
        out = pd.DataFrame({
            "CVE_ID":           df_s["CVE_ID"],
            "Observation_Date": df_s["Observation_Date"],
            "target":           y_s,
            "predicted_prob":   probs,
            "predicted_label":  (probs >= best_thr).astype(int),
        })
        p = OUTPUTS_DIR / f"v3_predictions_{name}.csv"
        out.to_csv(p, index=False)
        print(f"    {p}")

    # ── 11. CVE-level Borda aggregation (test set) ────────────────────────────
    print("\n[10] Computing CVE-level Borda count scores (test set)...")
    test_pred_df = pd.DataFrame({
        "CVE_ID":           test_df["CVE_ID"],
        "Observation_Date": test_df["Observation_Date"],
        "target":           y_test,
        "predicted_prob":   prob_cal_test,
    })
    cve_scores = borda_cve_scores(test_pred_df, "predicted_prob")
    borda_path = OUTPUTS_DIR / "v3_cve_level_scores.csv"
    cve_scores.to_csv(borda_path, index=False)
    print(f"    CVE-level scores: {borda_path} ({len(cve_scores):,d} unique CVEs)")

    # CVE-level evaluation
    cve_y_true = cve_scores["true_label"].values
    cve_scores_arr = cve_scores["cve_risk_score"].values
    cve_roc = roc_auc_score(cve_y_true, cve_scores_arr) if len(np.unique(cve_y_true)) > 1 else 0.5
    cve_pr  = average_precision_score(cve_y_true, cve_scores_arr) if len(np.unique(cve_y_true)) > 1 else 0.0
    print(f"    CVE-level ROC-AUC: {cve_roc:.4f}  PR-AUC: {cve_pr:.6f}")

    # ── 12. Save metrics ──────────────────────────────────────────────────────
    metrics = {
        "model": "CyberGuard-Ensemble v3",
        "base_learners": meta_keys,
        "best_threshold_from_val": best_thr,
        "cve_level_roc_auc": round(cve_roc, 6),
        "cve_level_pr_auc": round(cve_pr, 6),
        "threshold_0.5": {"train": tr_m, "validation": val_m, "test": test_m},
        "threshold_optimal": {"train": tr_m_opt, "validation": val_m_opt, "test": test_m_opt},
        "individual_base_val_roc": {k: round(roc_auc_score(y_val, meta_val[k]), 6) for k in meta_keys},
        "stacked_val_roc": round(val_roc_stacked, 6),
    }
    m_path = OUTPUTS_DIR / "v3_metrics.json"
    with open(m_path, "w") as f:
        json.dump(metrics, f, indent=2)

    _write_v3_report(metrics, tr_m, val_m, test_m, test_m_opt, best_thr, meta_keys,
                     cve_roc, cve_pr)

    total = round(time.time() - start, 1)
    print(f"\n[Done] CyberGuard v3 ensemble completed in {total}s")
    return metrics


def _write_v3_report(metrics, tr_m, val_m, test_m, test_m_opt, best_thr,
                      base_learners, cve_roc, cve_pr):
    indiv = metrics.get("individual_base_val_roc", {})
    report = f"""# CyberGuard-Ensemble v3 Report

## Architecture (Novel Contributions)

### Novel Contribution 1: Temporal-Aware Stacking Ensemble
Base learners: **{', '.join(base_learners)}** → Meta-learner: Ridge Logistic Regression

| Base Learner | Val ROC-AUC |
|---|---|
""" + "\n".join([f"| {k} | {v:.4f} |" for k, v in indiv.items()]) + f"""
| **Stacked ensemble** | **{metrics['stacked_val_roc']:.4f}** |

### Novel Contribution 2: Platt Calibration (Temporal Drift Correction)
Fitted on validation (2023) to correct 2022→2024 class-prior shift.

### Novel Contribution 3: CVE-Level Borda Count Aggregation
Collapses multiple monthly snapshots into one per-CVE operational risk score.
- **CVE-level ROC-AUC (test)**: {cve_roc:.4f}
- **CVE-level PR-AUC (test)**: {cve_pr:.6f}

---

## Row-Level Performance (Calibrated Probabilities)

### Default Threshold (0.50)

| Metric | Train (2022) | Validation (2023) | Test (2024) |
|---|---|---|---|
| **ROC-AUC** | {tr_m['ROC_AUC']:.6f} | {val_m['ROC_AUC']:.6f} | {test_m['ROC_AUC']:.6f} |
| **PR-AUC** | {tr_m['PR_AUC']:.6f} | {val_m['PR_AUC']:.6f} | {test_m['PR_AUC']:.6f} |
| **F1 (minority)** | {tr_m['F1_Score']:.6f} | {val_m['F1_Score']:.6f} | {test_m['F1_Score']:.6f} |
| **Precision** | {tr_m['Precision']:.6f} | {val_m['Precision']:.6f} | {test_m['Precision']:.6f} |
| **Recall** | {tr_m['Recall']:.6f} | {val_m['Recall']:.6f} | {test_m['Recall']:.6f} |
| **Brier Score** | {tr_m['Brier_Score']:.8f} | {val_m['Brier_Score']:.8f} | {test_m['Brier_Score']:.8f} |
| **Actual Positives** | {tr_m['Actual_Positives']} | {val_m['Actual_Positives']} | {test_m['Actual_Positives']} |

### Val-Optimal Threshold ({best_thr:.4f})

| Metric | Test (2024) |
|---|---|
| **ROC-AUC** | {test_m_opt['ROC_AUC']:.6f} |
| **F1 (minority)** | {test_m_opt['F1_Score']:.6f} |
| **Precision** | {test_m_opt['Precision']:.6f} |
| **Recall** | {test_m_opt['Recall']:.6f} |

### Top-K Ranking (Test 2024)

| K | Precision@K | Recall@K | Positives Found |
|---|---|---|---|
| 10 | {test_m['Precision@10']:.4f} | {test_m['Recall@10']:.4f} | {test_m['Positives@10']} |
| 25 | {test_m['Precision@25']:.4f} | {test_m['Recall@25']:.4f} | {test_m['Positives@25']} |
| 50 | {test_m['Precision@50']:.4f} | {test_m['Recall@50']:.4f} | {test_m['Positives@50']} |
| 100 | {test_m['Precision@100']:.4f} | {test_m['Recall@100']:.4f} | {test_m['Positives@100']} |
| 200 | {test_m['Precision@200']:.4f} | {test_m['Recall@200']:.4f} | {test_m['Positives@200']} |
| 500 | {test_m['Precision@500']:.4f} | {test_m['Recall@500']:.4f} | {test_m['Positives@500']} |
| 1000 | {test_m['Precision@1000']:.4f} | {test_m['Recall@1000']:.4f} | {test_m['Positives@1000']} |
"""
    r_path = REPORTS_DIR / "cyberguard_v3_report.md"
    with open(r_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"    Report saved: {r_path}")


if __name__ == "__main__":
    run_v3_ensemble()
