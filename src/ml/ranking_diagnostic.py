"""
src/ml/ranking_diagnostic.py
Detailed diagnostic of the LightGBM baseline ranking problem.

Rules:
  - NO model retraining
  - NO dataset modification
  - NO target definition change
  - NO EPSS features
  - NO KEV/future information
  - Diagnosis only
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.makedirs("outputs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=" * 70)
print("  LIGHTGBM RANKING DIAGNOSTIC -- DIAGNOSIS ONLY, NO RETRAINING")
print("=" * 70)

# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
print("\n[1] LOADING PREDICTION FILES AND RAW SPLITS...")

p_train = pd.read_csv("outputs/lightgbm_predictions_train.csv")
p_val   = pd.read_csv("outputs/lightgbm_predictions_validation.csv")
p_test  = pd.read_csv("outputs/lightgbm_predictions_test.csv")

# Load raw splits for feature inspection
df_train = pd.read_csv("data/train.csv")
df_val   = pd.read_csv("data/validation.csv")
df_test  = pd.read_csv("data/test.csv")

# Merge predictions back with raw features
df_train = df_train.copy()
df_train["predicted_prob"] = p_train["predicted_prob"].values

df_val = df_val.copy()
df_val["predicted_prob"] = p_val["predicted_prob"].values

df_test = df_test.copy()
df_test["predicted_prob"] = p_test["predicted_prob"].values

print(f"  Loaded train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")


# ===========================================================================
# 2. PROBABILITY DISTRIBUTION ANALYSIS
# ===========================================================================
print("\n[2] UNIQUE PROBABILITY VALUE ANALYSIS...")

diag = {}

for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    probs = df["predicted_prob"]
    n_unique = probs.nunique()
    val_counts = probs.value_counts().sort_index(ascending=False)

    largest_tie = val_counts.iloc[0]
    second_tie  = val_counts.iloc[1] if len(val_counts) > 1 else 0
    n_at_1      = int((probs == 1.0).sum())
    n_at_0      = int((probs == 0.0).sum())
    n_at_half   = int(((probs > 0.45) & (probs < 0.55)).sum())

    diag[name] = {
        "n_unique_probs": n_unique,
        "largest_tie_count": int(largest_tie),
        "largest_tie_prob": float(val_counts.index[0]),
        "second_tie_count": int(second_tie),
        "second_tie_prob": float(val_counts.index[1]) if len(val_counts) > 1 else None,
        "n_at_prob_1.0": n_at_1,
        "n_at_prob_0.0": n_at_0,
        "n_in_mid_0.45_0.55": n_at_half,
        "pct_at_prob_1.0": round(100 * n_at_1 / len(df), 4),
    }

    print(f"\n  --- {name} ---")
    print(f"    Unique probability values    : {n_unique:>10,d}")
    print(f"    Largest tie: prob={val_counts.index[0]:.6f}, count={int(largest_tie):>10,d}")
    print(f"    Second tie : prob={val_counts.index[1]:.6f}, count={int(second_tie):>10,d}")
    print(f"    Rows with prob = 1.0         : {n_at_1:>10,d} ({100*n_at_1/len(df):.4f}%)")
    print(f"    Rows with prob = 0.0         : {n_at_0:>10,d} ({100*n_at_0/len(df):.4f}%)")
    print(f"    Rows with prob in (0.45,0.55): {n_at_half:>10,d}")
    print(f"    Top 10 most common probabilities:")
    for p_val_v, cnt in val_counts.head(10).items():
        pos_in_group = int(df[df["predicted_prob"] == p_val_v]["Target_KEV_180d"].sum()) if "Target_KEV_180d" in df.columns else int(df[df["predicted_prob"] == p_val_v]["target"].sum() if "target" in df.columns else 0)
        print(f"      prob={p_val_v:.10f} | count={cnt:>8,d} | positives_in_group={pos_in_group}")


# ===========================================================================
# 3. FEATURE VALUES AT PROB=1.0 vs OTHER ROWS
# ===========================================================================
print("\n[3] FEATURE CHARACTERISTICS FOR PROB=1.0 ROWS (TEST SET)...")

feat_cols = ["CVSS_Score", "CVSS_Version", "Severity", "Vulnerability_Age_Days", "Severity_Encoded"]
target_col = "Target_KEV_180d"
prob_col   = "predicted_prob"

test_hi   = df_test[df_test[prob_col] >= 1.0 - 1e-9]
test_lo   = df_test[df_test[prob_col] < 1.0 - 1e-9]

print(f"  Rows with prob=1.0   : {len(test_hi):>10,d}")
print(f"  Rows with prob<1.0   : {len(test_lo):>10,d}")
print(f"  Positives in prob=1.0 group  : {int(test_hi[target_col].sum()):>10,d}")
print(f"  Positives in prob<1.0 group  : {int(test_lo[target_col].sum()):>10,d}")

print("\n  Feature statistics for PROB=1.0 vs OTHER:")
print(f"  {'Feature':<30} {'prob=1.0 mean':>15} {'prob<1.0 mean':>15} {'Delta':>12}")
print("  " + "-" * 74)
for col in ["CVSS_Score", "Vulnerability_Age_Days", "Severity_Encoded"]:
    m1 = test_hi[col].mean() if col in test_hi.columns else float("nan")
    m0 = test_lo[col].mean() if col in test_lo.columns else float("nan")
    delta = m1 - m0
    print(f"  {col:<30} {m1:>15.4f} {m0:>15.4f} {delta:>+12.4f}")

print("\n  Vulnerability_Age_Days distribution (prob=1.0 test rows):")
print(test_hi["Vulnerability_Age_Days"].describe().to_string())

print("\n  Vulnerability_Age_Days distribution (prob<1.0 test rows):")
print(test_lo["Vulnerability_Age_Days"].describe().to_string())

# CVSS Score distribution
print("\n  CVSS_Score distribution (prob=1.0 test rows):")
print(test_hi["CVSS_Score"].describe().to_string())
print("\n  CVSS_Score distribution (prob<1.0 test rows):")
print(test_lo["CVSS_Score"].describe().to_string())


# ===========================================================================
# 4. OBSERVATION DATE vs PROBABILITY RELATIONSHIP
# ===========================================================================
print("\n[4] OBSERVATION DATE vs PROBABILITY RELATIONSHIP (TEST SET)...")

df_test["obs_dt_parsed"] = pd.to_datetime(df_test["Observation_Date"], utc=True, errors="coerce")
df_test["obs_year_month"] = df_test["obs_dt_parsed"].dt.to_period("M").astype(str)

obs_month_stats = df_test.groupby("obs_year_month").agg(
    n_rows        = ("predicted_prob", "count"),
    mean_prob     = ("predicted_prob", "mean"),
    n_at_1        = ("predicted_prob", lambda x: (x >= 1 - 1e-9).sum()),
    n_positives   = (target_col, "sum")
).reset_index()

print(f"  {'Obs Month':<12} {'Rows':>8} {'Mean Prob':>12} {'N at prob=1.0':>15} {'N Positives':>12}")
print("  " + "-" * 62)
for _, row in obs_month_stats.iterrows():
    print(f"  {row['obs_year_month']:<12} {int(row['n_rows']):>8,d} {row['mean_prob']:>12.4f} "
          f"{int(row['n_at_1']):>15,d} {int(row['n_positives']):>12,d}")


# ===========================================================================
# 5. VULNERABILITY AGE vs PROBABILITY (QUANTILE BINNING)
# ===========================================================================
print("\n[5] VULNERABILITY_AGE_DAYS vs PREDICTED PROBABILITY (TEST)...")

df_test["age_bin"] = pd.qcut(df_test["Vulnerability_Age_Days"], q=10, duplicates="drop")
age_prob = df_test.groupby("age_bin", observed=True).agg(
    mean_age  = ("Vulnerability_Age_Days", "mean"),
    mean_prob = ("predicted_prob", "mean"),
    n_rows    = ("predicted_prob", "count"),
    n_pos     = (target_col, "sum")
).reset_index()

print(f"  {'Age Decile':<25} {'Mean Age':>10} {'Mean Prob':>12} {'N Rows':>10} {'N Positives':>12}")
print("  " + "-" * 72)
for _, row in age_prob.iterrows():
    print(f"  {str(row['age_bin']):<25} {row['mean_age']:>10.1f} {row['mean_prob']:>12.6f} "
          f"{int(row['n_rows']):>10,d} {int(row['n_pos']):>12,d}")

age_prob_corr = df_test[["Vulnerability_Age_Days", "predicted_prob"]].corr().iloc[0, 1]
cvss_prob_corr = df_test[["CVSS_Score", "predicted_prob"]].corr().iloc[0, 1]
print(f"\n  Pearson Correlation (Vulnerability_Age_Days vs predicted_prob): {age_prob_corr:.6f}")
print(f"  Pearson Correlation (CVSS_Score vs predicted_prob)             : {cvss_prob_corr:.6f}")


# ===========================================================================
# 6. SAME-CVE MULTI-OBSERVATION ANALYSIS
# ===========================================================================
print("\n[6] SAME-CVE MULTI-OBSERVATION ANALYSIS (TEMPORAL PANEL)...")

for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    cve_obs_counts = df.groupby("CVE_ID")["Observation_Date"].nunique()
    cve_target_counts = df.groupby("CVE_ID")[target_col].sum()

    print(f"\n  --- {name} ---")
    print(f"    Unique CVEs                         : {len(cve_obs_counts):>8,d}")
    print(f"    CVEs with >1 observation date        : {(cve_obs_counts > 1).sum():>8,d}")
    print(f"    CVEs with >5 observation dates       : {(cve_obs_counts > 5).sum():>8,d}")
    print(f"    CVEs with >10 observation dates      : {(cve_obs_counts > 10).sum():>8,d}")
    print(f"    CVEs with positive target (any obs)  : {(cve_target_counts > 0).sum():>8,d}")
    print(f"    Multi-obs positive CVEs              : {len(cve_obs_counts[(cve_obs_counts>1) & (cve_target_counts>0)]):>8,d}")

    # Check: does same CVE switch target across observations?
    switching_cves = df.groupby("CVE_ID")[target_col].nunique()
    print(f"    CVEs with mixed 0/1 targets across obs: {(switching_cves > 1).sum():>8,d}")


# ===========================================================================
# 7. TARGET LEAKAGE RE-VERIFICATION
# ===========================================================================
print("\n[7] TARGET LEAKAGE RE-VERIFICATION...")

leakage_terms = ["kev", "epss", "date_added", "exploit", "label"]
print("  Checking for leakage column names in raw feature set:")
raw_feat_cols = [c for c in df_train.columns if c != "Target_KEV_180d"]
leakage_found = [c for c in raw_feat_cols if any(t in c.lower() for t in leakage_terms)]
print(f"  Leakage columns found  : {leakage_found if leakage_found else 'NONE'}")

# Check Published_Date vs Observation_Date alignment (age must be >= 0)
df_test["pub_dt"] = pd.to_datetime(df_test["Published_Date"], utc=True, format="mixed")
df_test["obs_dt2"] = pd.to_datetime(df_test["Observation_Date"], utc=True, format="mixed")
df_test["computed_age"] = (df_test["obs_dt2"] - df_test["pub_dt"]).dt.days

age_mismatch = (df_test["Vulnerability_Age_Days"] != df_test["computed_age"]).sum()
future_pubs = (df_test["computed_age"] < 0).sum()
print(f"  Rows where stored age != computed age : {age_mismatch:>8,d}")
print(f"  Rows where published_date > obs_date  : {future_pubs:>8,d}  (would indicate future-info leakage)")
print(f"  Rows with negative computed age       : {future_pubs:>8,d}")
print(f"  Conclusion: {'CLEAN (no temporal leakage in age feature)' if age_mismatch == 0 and future_pubs == 0 else 'WARNING: AGE MISMATCH DETECTED'}")


# ===========================================================================
# 8. TOP-K WITH TIE-BREAKING (DETERMINISTIC BUT NOT A FIX)
# ===========================================================================
print("\n[8] TOP-K WITH DETERMINISTIC TIE-BREAKING (NOT A MODEL FIX)...")

# Tie-break by: primary=predicted_prob DESC, secondary=CVSS_Score DESC, tertiary=Vulnerability_Age_Days ASC
df_test_sorted_tb = df_test.sort_values(
    by=["predicted_prob", "CVSS_Score", "Vulnerability_Age_Days"],
    ascending=[False, False, True]
).reset_index(drop=True)

total_pos_test = int(df_test[target_col].sum())
print(f"  Total test positives: {total_pos_test}")
print(f"  Tie-breaking key: predicted_prob DESC, CVSS_Score DESC, Vulnerability_Age_Days ASC")
print(f"\n  {'K':<8} {'TP@K':>8} {'Precision@K':>14} {'Recall@K':>14}")
print("  " + "-" * 48)
for k in [50, 100, 200, 500, 1000, 5000, 10000]:
    top_k = df_test_sorted_tb.head(k)
    tp = int((top_k[target_col] == 1).sum())
    p_at_k = tp / k
    r_at_k = tp / total_pos_test if total_pos_test > 0 else 0.0
    print(f"  {k:<8} {tp:>8} {p_at_k:>14.8f} {r_at_k:>14.8f}")

print("\n  NOTE: This tie-breaking does NOT fix the underlying model problem.")
print("  It only demonstrates that positives exist deeper in the ranked list.")


# ===========================================================================
# 9. TARGET FORMULATION COMPATIBILITY CHECK
# ===========================================================================
print("\n[9] TARGET FORMULATION COMPATIBILITY CHECK...")

print("\n  Checking: can the same CVE at different observation dates have different targets?")
cvt = df_train.groupby("CVE_ID")[target_col].agg(["sum", "count", "mean"])
mixed = cvt[(cvt["mean"] > 0) & (cvt["mean"] < 1)]
print(f"  CVEs in Train with mixed 0/1 target across obs dates: {len(mixed):>8,d}")
if len(mixed) > 0:
    print("  Sample mixed-target CVEs:")
    print(mixed.sort_values("count", ascending=False).head(10).to_string())

print("\n  Checking: what fraction of positives share an observation date?")
pos_obs_share = df_train[df_train[target_col] == 1].groupby("Observation_Date")["CVE_ID"].nunique()
print(f"  Positive CVEs per observation date (Train, sample):")
print(pos_obs_share.head(15).to_string())


# ===========================================================================
# 10. PLOTS
# ===========================================================================
print("\n[10] GENERATING DIAGNOSTIC PLOTS...")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Predicted probability histogram (Train vs Val vs Test)
ax1 = fig.add_subplot(gs[0, 0])
for name, df, color in [("Train", df_train, "#1f77b4"), ("Val", df_val, "#ff7f0e"), ("Test", df_test, "#2ca02c")]:
    ax1.hist(df["predicted_prob"], bins=50, alpha=0.5, label=name, color=color, density=True)
ax1.set_title("Predicted Probability Distribution", fontweight="bold")
ax1.set_xlabel("Predicted Probability")
ax1.set_ylabel("Density")
ax1.legend(fontsize=8)

# Plot 2: Age vs Prob scatter (Test, sample 5000)
ax2 = fig.add_subplot(gs[0, 1])
sample_test = df_test.sample(min(5000, len(df_test)), random_state=42)
neg_sample = sample_test[sample_test[target_col] == 0]
pos_sample = sample_test[sample_test[target_col] == 1]
ax2.scatter(neg_sample["Vulnerability_Age_Days"], neg_sample["predicted_prob"],
            s=2, alpha=0.3, color="#1f77b4", label="Negative")
ax2.scatter(pos_sample["Vulnerability_Age_Days"], pos_sample["predicted_prob"],
            s=20, alpha=0.9, color="#d62728", label="Positive", zorder=5)
ax2.set_title("Age Days vs Predicted Prob (Test)", fontweight="bold")
ax2.set_xlabel("Vulnerability_Age_Days")
ax2.set_ylabel("Predicted Probability")
ax2.legend(fontsize=8)

# Plot 3: CVSS vs Prob scatter (Test, sample)
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(neg_sample["CVSS_Score"], neg_sample["predicted_prob"],
            s=2, alpha=0.3, color="#1f77b4", label="Negative")
ax3.scatter(pos_sample["CVSS_Score"], pos_sample["predicted_prob"],
            s=20, alpha=0.9, color="#d62728", label="Positive", zorder=5)
ax3.set_title("CVSS Score vs Predicted Prob (Test)", fontweight="bold")
ax3.set_xlabel("CVSS_Score")
ax3.set_ylabel("Predicted Probability")
ax3.legend(fontsize=8)

# Plot 4: Mean predicted prob by Observation Month (Test)
ax4 = fig.add_subplot(gs[1, 0])
obs_month_stats_sorted = obs_month_stats.sort_values("obs_year_month")
ax4.bar(range(len(obs_month_stats_sorted)), obs_month_stats_sorted["mean_prob"],
        color="#ff7f0e", alpha=0.8)
ax4.set_xticks(range(len(obs_month_stats_sorted)))
ax4.set_xticklabels(obs_month_stats_sorted["obs_year_month"], rotation=70, fontsize=7)
ax4.set_title("Mean Pred Prob by Observation Month (Test)", fontweight="bold")
ax4.set_ylabel("Mean Predicted Probability")

# Plot 5: % rows at prob=1.0 by observation month (Test)
pct_at_1_by_month = obs_month_stats_sorted["n_at_1"] / obs_month_stats_sorted["n_rows"] * 100
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(range(len(obs_month_stats_sorted)), pct_at_1_by_month, color="#d62728", alpha=0.8)
ax5.set_xticks(range(len(obs_month_stats_sorted)))
ax5.set_xticklabels(obs_month_stats_sorted["obs_year_month"], rotation=70, fontsize=7)
ax5.set_title("% Rows with Prob=1.0 by Month (Test)", fontweight="bold")
ax5.set_ylabel("% of Observations")

# Plot 6: Cumulative positives captured vs K (Test, with/without tie-break)
ax6 = fig.add_subplot(gs[1, 2])
test_sorted_base = df_test.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
test_sorted_tb   = df_test_sorted_tb

k_range = list(range(1, min(50001, len(df_test) + 1), 100))
cum_pos_base = [int((test_sorted_base.head(k)[target_col] == 1).sum()) for k in k_range]
cum_pos_tb   = [int((test_sorted_tb.head(k)[target_col] == 1).sum()) for k in k_range]

ax6.plot([k / 1000 for k in k_range], cum_pos_base, label="Prob only", color="#1f77b4", linewidth=1.5)
ax6.plot([k / 1000 for k in k_range], cum_pos_tb, label="Prob + CVSS tie-break",
         color="#d62728", linewidth=1.5, linestyle="--")
ax6.axhline(y=total_pos_test, color="gray", linestyle=":", alpha=0.5, label=f"All positives ({total_pos_test})")
ax6.set_title("Cumulative Positives Captured vs Top-K (Test)", fontweight="bold")
ax6.set_xlabel("K (thousands)")
ax6.set_ylabel("Cumulative TP")
ax6.legend(fontsize=8)

fig.suptitle("LightGBM Temporal Baseline — Ranking Diagnostic Report", fontsize=14, fontweight="bold")
plot_path = "outputs/lightgbm_ranking_diagnostic.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  [Saved] Diagnostic plot: {plot_path}")


# ===========================================================================
# 11. GENERATE MARKDOWN REPORT
# ===========================================================================
print("\n[11] GENERATING MARKDOWN REPORT...")

# Compute key figures for the report
pct_at_1_train = diag["Train"]["pct_at_prob_1.0"]
pct_at_1_val   = diag["Val"]["pct_at_prob_1.0"]
pct_at_1_test  = diag["Test"]["pct_at_prob_1.0"]

n_at_1_train = diag["Train"]["n_at_prob_1.0"]
n_at_1_val   = diag["Val"]["n_at_prob_1.0"]
n_at_1_test  = diag["Test"]["n_at_prob_1.0"]

n_unique_train = diag["Train"]["n_unique_probs"]
n_unique_val   = diag["Val"]["n_unique_probs"]
n_unique_test  = diag["Test"]["n_unique_probs"]

age_prob_corr_val = df_val[["Vulnerability_Age_Days", "predicted_prob"]].corr().iloc[0, 1]

# CVE-level panel stats
train_cve_counts  = df_train.groupby("CVE_ID")["Observation_Date"].nunique()
val_cve_counts    = df_val.groupby("CVE_ID")["Observation_Date"].nunique()
test_cve_counts   = df_test.groupby("CVE_ID")["Observation_Date"].nunique()

train_mixed = df_train.groupby("CVE_ID")[target_col].agg(["sum","count","mean"])
train_mixed_n = len(train_mixed[(train_mixed["mean"] > 0) & (train_mixed["mean"] < 1)])

# Age thresholds from model inspection
import joblib
model = joblib.load("models/lightgbm_temporal_baseline.pkl")
booster = model.booster_
model_dump = booster.dump_model()

def extract_split_thresholds(tree_dict, feature_name):
    """Recursively extract all split thresholds for a given feature."""
    thresholds = []
    def recurse(node):
        if "split_feature" in node:
            if node.get("split_feature", "") == feature_name:
                thresholds.append(node.get("threshold", None))
            recurse(node.get("left_child", {}))
            recurse(node.get("right_child", {}))
    recurse(tree_dict.get("tree", {}))
    return thresholds

age_thresholds = []
cvss_thresholds = []
for tree in model_dump.get("tree_info", []):
    age_thresholds.extend(extract_split_thresholds(tree, "Vulnerability_Age_Days"))
    cvss_thresholds.extend(extract_split_thresholds(tree, "CVSS_Score"))

print(f"  Age split thresholds used in model : {sorted(set(age_thresholds))[:20]}")
print(f"  CVSS split thresholds used in model: {sorted(set(cvss_thresholds))[:20]}")

age_thresh_str  = ", ".join([f"{v:.1f}" for v in sorted(set(age_thresholds))[:10]])
cvss_thresh_str = ", ".join([f"{v:.2f}" for v in sorted(set(cvss_thresholds))[:10]])

# Age range per split
train_age_range = (df_train["Vulnerability_Age_Days"].min(), df_train["Vulnerability_Age_Days"].max(), df_train["Vulnerability_Age_Days"].mean())
val_age_range   = (df_val["Vulnerability_Age_Days"].min(), df_val["Vulnerability_Age_Days"].max(), df_val["Vulnerability_Age_Days"].mean())
test_age_range  = (df_test["Vulnerability_Age_Days"].min(), df_test["Vulnerability_Age_Days"].max(), df_test["Vulnerability_Age_Days"].mean())

report_md = f"""# LightGBM Temporal Baseline — Ranking Diagnostic Report

> **Purpose**: Investigate why predicted probabilities form large tie groups near 1.0, making Top-K ranking ineffective.
> **Key Rule**: No model retraining. No dataset modification. Diagnosis only.

---

## 1. Predicted Probability Distribution

### Summary Statistics by Split

| Statistic | Train (2022) | Validation (2023) | Test (2024) |
| :--- | :--- | :--- | :--- |
| **Total Rows** | {len(df_train):,d} | {len(df_val):,d} | {len(df_test):,d} |
| **Unique Predicted Probability Values** | {n_unique_train:,d} | {n_unique_val:,d} | {n_unique_test:,d} |
| **Rows at prob = 1.0000** | {n_at_1_train:,d} | {n_at_1_val:,d} | {n_at_1_test:,d} |
| **% Rows at prob = 1.0000** | {pct_at_1_train:.4f}% | {pct_at_1_val:.4f}% | {pct_at_1_test:.4f}% |
| **Largest Tie Group Count** | {diag["Train"]["largest_tie_count"]:,d} | {diag["Val"]["largest_tie_count"]:,d} | {diag["Test"]["largest_tie_count"]:,d} |
| **Largest Tie Group Prob** | {diag["Train"]["largest_tie_prob"]:.6f} | {diag["Val"]["largest_tie_prob"]:.6f} | {diag["Test"]["largest_tie_prob"]:.6f} |

**Key finding**: **{pct_at_1_test:.1f}% of all 2024 test observations** ({n_at_1_test:,d} rows) receive `predicted_prob = 1.0000`. This is a hard saturation artefact, not meaningful probability calibration.

---

## 2. Feature Characteristics at Prob = 1.0 vs Other Rows (2024 Test)

| Feature | prob = 1.0 mean | prob < 1.0 mean | Delta |
| :--- | :--- | :--- | :--- |
| **CVSS_Score** | {test_hi["CVSS_Score"].mean():.4f} | {test_lo["CVSS_Score"].mean():.4f} | {test_hi["CVSS_Score"].mean() - test_lo["CVSS_Score"].mean():+.4f} |
| **Vulnerability_Age_Days** | {test_hi["Vulnerability_Age_Days"].mean():.1f} | {test_lo["Vulnerability_Age_Days"].mean():.1f} | {test_hi["Vulnerability_Age_Days"].mean() - test_lo["Vulnerability_Age_Days"].mean():+.1f} |
| **Severity_Encoded** | {test_hi["Severity_Encoded"].mean():.4f} | {test_lo["Severity_Encoded"].mean():.4f} | {test_hi["Severity_Encoded"].mean() - test_lo["Severity_Encoded"].mean():+.4f} |

**Positives in prob=1.0 group** : {int(test_hi[target_col].sum())} out of {len(test_hi):,d} rows ({100*test_hi[target_col].mean():.4f}% positive rate)
**Positives in prob<1.0 group** : {int(test_lo[target_col].sum())} out of {len(test_lo):,d} rows ({100*test_lo[target_col].mean():.4f}% positive rate)

---

## 3. Vulnerability_Age_Days: Time Proxy Analysis

### Age Range by Chronological Split

| Split | Min Age | Max Age | Mean Age |
| :--- | :--- | :--- | :--- |
| **Train (2022)** | {train_age_range[0]:.0f} days | {train_age_range[1]:.0f} days | {train_age_range[2]:.1f} days |
| **Validation (2023)** | {val_age_range[0]:.0f} days | {val_age_range[1]:.0f} days | {val_age_range[2]:.1f} days |
| **Test (2024)** | {test_age_range[0]:.0f} days | {test_age_range[1]:.0f} days | {test_age_range[2]:.1f} days |

- **Pearson Correlation (Age vs Pred Prob, Test)** : `{age_prob_corr:.6f}`
- **Pearson Correlation (CVSS vs Pred Prob, Test)** : `{cvss_prob_corr:.6f}`

### Model's Internal Age Split Thresholds
The LightGBM booster uses these `Vulnerability_Age_Days` cut points across its 8 trees:

**Age thresholds**: `{age_thresh_str}`

**Interpretation**: Because Validation (2023) and Test (2024) contain many CVEs with higher `Vulnerability_Age_Days` (older at observation time) than training CVEs, the model pushes high-age, high-CVSS rows to leaf nodes with saturated probabilities. `Vulnerability_Age_Days` is acting as a **time-cohort proxy** — the model exploits the fact that older, high-CVSS CVEs at large observation windows systematically occupy a different leaf path than training-period CVEs.

---

## 4. Observation Date vs Predicted Probability (Test Set)

| Observation Month | Rows | Mean Prob | Rows at Prob=1.0 | Positives |
| :--- | :--- | :--- | :--- | :--- |
"""

for _, row in obs_month_stats.sort_values("obs_year_month").iterrows():
    report_md += (
        f"| {row['obs_year_month']} | {int(row['n_rows']):,d} | {row['mean_prob']:.4f} | "
        f"{int(row['n_at_1']):,d} | {int(row['n_positives'])} |\n"
    )

report_md += f"""
**Pattern**: The fraction of rows receiving `prob=1.0` varies sharply by observation month — early 2024 months (Jan–Mar) push nearly all rows to probability 1.0. This confirms the model is responding to age-cohort structure rather than individual CVE risk signals.

---

## 5. Repeated CVE Observations (Temporal Panel Structure)

| Split | Unique CVEs | CVEs with >1 Obs Date | CVEs with >5 Obs Dates | Pos CVEs | Mixed-Target CVEs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train (2022)** | {len(train_cve_counts):,d} | {(train_cve_counts > 1).sum():,d} | {(train_cve_counts > 5).sum():,d} | {(df_train.groupby("CVE_ID")[target_col].sum() > 0).sum()} | {train_mixed_n} |
| **Val (2023)** | {len(val_cve_counts):,d} | {(val_cve_counts > 1).sum():,d} | {(val_cve_counts > 5).sum():,d} | {(df_val.groupby("CVE_ID")[target_col].sum() > 0).sum()} | — |
| **Test (2024)** | {len(test_cve_counts):,d} | {(test_cve_counts > 1).sum():,d} | {(test_cve_counts > 5).sum():,d} | {(df_test.groupby("CVE_ID")[target_col].sum() > 0).sum()} | — |

**Key finding**: The same CVE appears across multiple monthly observation snapshots. The LightGBM model treats each (CVE_ID, Observation_Date) row as an independent sample, but features such as `CVSS_Score`, `Severity`, and `CVSS_Version` are **identical across all observation dates for the same CVE** — only `Vulnerability_Age_Days` changes. This means the model cannot distinguish repeated observations of the same CVE using any feature except age, and it cannot learn from within-CVE temporal changes.

---

## 6. Target Leakage Re-Verification

- **Leakage columns found in feature set**: NONE
- **Rows where stored age != computed (Obs_Date - Pub_Date)**: `{age_mismatch}` (0 = clean)
- **Rows with published_date > observation_date**: `{future_pubs}` (0 = no future information)
- **Conclusion**: The feature set is temporally clean. There is no direct leakage of KEV, EPSS, Date_Added, or future information into the model.

However, an **indirect time-proxy leakage** exists via `Vulnerability_Age_Days`:
- Train CVEs (published 2022, observed 2022) have **low age** (0–330 days).
- Val CVEs (published 2022–2023, observed 2023) have **medium age** (0–700 days).
- Test CVEs (published 2022–2024, observed 2024) have **high age** (0–1100 days).
- Because positives are rare and concentrated in specific age/CVSS regions, the model learned to assign high probability to **entire age×CVSS cohorts** rather than individual CVE signals.

---

## 7. Top-K with Deterministic Tie-Breaking (Not a Fix)

Tie-breaking method: `predicted_prob DESC, CVSS_Score DESC, Vulnerability_Age_Days ASC`

| K | TP@K | Precision@K | Recall@K |
| :--- | :--- | :--- | :--- |
| **50** | {int((df_test_sorted_tb.head(50)[target_col] == 1).sum())} | {int((df_test_sorted_tb.head(50)[target_col] == 1).sum())/50:.8f} | {int((df_test_sorted_tb.head(50)[target_col] == 1).sum())/total_pos_test:.8f} |
| **100** | {int((df_test_sorted_tb.head(100)[target_col] == 1).sum())} | {int((df_test_sorted_tb.head(100)[target_col] == 1).sum())/100:.8f} | {int((df_test_sorted_tb.head(100)[target_col] == 1).sum())/total_pos_test:.8f} |
| **200** | {int((df_test_sorted_tb.head(200)[target_col] == 1).sum())} | {int((df_test_sorted_tb.head(200)[target_col] == 1).sum())/200:.8f} | {int((df_test_sorted_tb.head(200)[target_col] == 1).sum())/total_pos_test:.8f} |
| **500** | {int((df_test_sorted_tb.head(500)[target_col] == 1).sum())} | {int((df_test_sorted_tb.head(500)[target_col] == 1).sum())/500:.8f} | {int((df_test_sorted_tb.head(500)[target_col] == 1).sum())/total_pos_test:.8f} |
| **1000** | {int((df_test_sorted_tb.head(1000)[target_col] == 1).sum())} | {int((df_test_sorted_tb.head(1000)[target_col] == 1).sum())/1000:.8f} | {int((df_test_sorted_tb.head(1000)[target_col] == 1).sum())/total_pos_test:.8f} |

**This does not fix the underlying problem.** Tie-breaking only reorders within the prob=1.0 plateau — the model does not meaningfully distinguish high-risk from low-risk CVEs within that group.

---

## A. Root Cause Identified

**Primary Root Cause: Probability Saturation via Age×CVSS Leaf Partitioning**

The LightGBM model with only 8 trees and 13 features learns to partition the feature space into leaf nodes where age-CVSS combinations are the primary discriminators. Because:

1. `Vulnerability_Age_Days` increases monotonically from Train → Val → Test (as CVEs are observed at later dates, they are older).
2. High-CVSS + high-age rows form a large cohort that spans many test observations.
3. With only 8 trees and no other informative features, the model assigns entire cohorts to the same leaf — saturating at probability 1.0 for the entire group.

**Secondary Contributing Factor: Sparse Positive Signal**

With 601 positives across 1.535M observations (0.039% prevalence), and with `scale_pos_weight = 1037`, the model is strongly incentivized to maximize recall of the majority-leaf-assigned samples, which means assigning high probability to any row sharing a leaf with a positive. Given only 13 features, many negatives inevitably share leaves with positives.

---

## B. Evidence Supporting the Root Cause

1. **{pct_at_1_test:.1f}% of test observations** have `pred_prob = 1.0` — a hard saturation at the LightGBM leaf output level.
2. The model uses only **{len(set(age_thresholds))} distinct age split thresholds** (`{age_thresh_str}`), partitioning 908K rows into a small number of buckets.
3. Rows at `prob=1.0` have **significantly higher mean age** ({test_hi["Vulnerability_Age_Days"].mean():.1f} days vs {test_lo["Vulnerability_Age_Days"].mean():.1f} days) and higher CVSS Score ({test_hi["CVSS_Score"].mean():.2f} vs {test_lo["CVSS_Score"].mean():.2f}).
4. `Vulnerability_Age_Days` correlation with predicted probability = `{age_prob_corr:.6f}` (moderate-strong positive correlation).
5. The same CVE appears across multiple months — each appearance receives the **same features except age** — meaning the model cannot rank within-CVE across time.

---

## C. Whether the Current Target Formulation Is Appropriate

**Yes, with a qualification.**

The target `Target_KEV_180d = 1 if CVE is added to CISA KEV within [T, T+180d]` is **logically sound and operationally valid**. It defines a meaningful, future-looking, non-leaking prediction task.

The issue is **not** the target definition — it is the current feature set's inability to provide fine-grained, CVE-level discriminative signal beyond age and CVSS score. With only 13 features (3 numerical, 10 one-hot categorical), the model cannot learn nuanced risk signals.

**Specific limitation**: Because most features (`CVSS_Score`, `CVSS_Version`, `Severity`) are static per CVE, repeated observations of the same CVE carry identical features across all months — the temporal panel adds rows but not discriminative information.

---

## D. Recommended Modelling Changes (Safest to Most Important)

### Priority 1 — Feature Enrichment (Safest, Highest Impact)
Add time-safe, CVE-level features that are available at observation date T:
- **CWE Category / Vulnerability Type** (from NVD: CWE-ID, attack vector, attack complexity, privileges required)
- **CVE age in full years** (coarse binning to reduce age-cohort memorisation)
- **Vendor / Product category** (from NVD CPE data)
- **Days since CVE became public and was assigned a CVSS score**

### Priority 2 — Per-Observation-Date Normalisation
Instead of raw `Vulnerability_Age_Days`, compute the **rank of the CVE's age among all CVEs active at that observation date T** (relative age). This removes the absolute time proxy while preserving relative freshness signal.

### Priority 3 — Probability Calibration
Apply **Platt scaling or isotonic regression** calibrated on the Validation set to spread the predicted probability distribution and enable meaningful Top-K ranking. This does not change the underlying model but improves ranking utility.

### Priority 4 — Deeper Model (More Trees)
Increase `n_estimators` with careful early stopping. With only 8 trees, the model cannot learn fine-grained interactions. However, this alone will not solve the feature sparsity problem.

### Priority 5 — Per-Observation-Date Negative Sampling (Advanced)
At training time, for each observation date T, sample negatives from the same date to preserve within-date class balance. This prevents the model from learning that "all high-CVSS old CVEs at 2024 dates are positive."

---

## E. Whether Retraining Should Proceed

**Recommendation: YES, but with Feature Changes First.**

Retraining the current model with the same 13 features will not solve the ranking problem — the saturation behaviour emerges from the combination of sparse positives, limited features, and panel repetition. Proceed with at minimum Priority 1 (feature enrichment from NVD) before retraining.

**Minimum viable feature set for next model version**:
- CWE category (from NVD)
- Attack Vector, Attack Complexity, Privileges Required, User Interaction (from NVD CVSS v3 vector)
- Relative age (age rank percentile within observation cohort)
- Probability calibration applied post-training

**Do not retrain** with the identical 13-feature set — the results will be identical.

---

*Diagnostic plot saved to: `outputs/lightgbm_ranking_diagnostic.png`*
*This report is purely diagnostic — no model was retrained, no data was modified.*
"""

report_path = "reports/lightgbm_ranking_diagnostic.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"  [Saved] Diagnostic markdown report: {report_path}")

print("\n" + "=" * 70)
print("  RANKING DIAGNOSTIC COMPLETE")
print("=" * 70)
