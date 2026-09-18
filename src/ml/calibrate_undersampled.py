# -*- coding: utf-8 -*-
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier

from src.ml.alperin_baseline import load_dataset, compute_metrics, build_preprocessor

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'outputs'
REPORT_DIR = ROOT / 'reports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
TOPK_LIST = [100, 500]

val_df = load_dataset('validation')
test_df = load_dataset('test')

TARGET_COL = 'Target_KEV_180d'
FEATURE_COLS = [
    'CVE_ID',
    'Observation_Date',
    'CVSS_Score',
    'CVSS_Version',
    'Severity',
    'Vulnerability_Age_Days',
    'Description',
]

X_train = load_dataset('train')[FEATURE_COLS].copy()
y_train = load_dataset('train')[TARGET_COL].values

X_val = val_df[FEATURE_COLS].copy()
y_val = val_df[TARGET_COL].values

X_test = test_df[FEATURE_COLS].copy()
y_test = test_df[TARGET_COL].values

print('[Preprocessing] Fitting ColumnTransformer on X_train...')
preprocess = build_preprocessor()
preprocess.fit(X_train)
X_train_proc = preprocess.transform(X_train)
X_val_proc = preprocess.transform(X_val)
X_test_proc = preprocess.transform(X_test)

pos_indices = np.where(y_train == 1)[0]
neg_indices = np.where(y_train == 0)[0]

models_to_eval = {
    'u10': 10,
    'u50': 50,
}

raw_val_probs = {}
raw_test_probs = {}

for exp_name, ratio in models_to_eval.items():
    print(f'[Model Extraction] Training Random Forest for {exp_name} (ratio={ratio})...')
    num_neg = len(pos_indices) * ratio
    rng = np.random.RandomState(42)
    selected_neg = rng.choice(neg_indices, size=num_neg, replace=False)
    sub_idx = np.sort(np.concatenate([pos_indices, selected_neg]))
    
    rf = RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        random_state=42,
        class_weight=None,
        n_jobs=-1,
    )
    rf.fit(X_train_proc[sub_idx], y_train[sub_idx])
    raw_val_probs[exp_name] = rf.predict_proba(X_val_proc)[:, 1]
    raw_test_probs[exp_name] = rf.predict_proba(X_test_proc)[:, 1]

alp_val_df = pd.read_csv(OUTPUT_DIR / 'alperin_predictions_validation.csv') if (OUTPUT_DIR / 'alperin_predictions_validation.csv').exists() else None
alp_test_df = pd.read_csv(OUTPUT_DIR / 'alperin_predictions_test.csv')

raw_val_probs['Alperin'] = alp_val_df['predicted_prob'].values if alp_val_df is not None else None
raw_test_probs['Alperin'] = alp_test_df['predicted_prob'].values

calibrated_probs = {}

for m in ['u10', 'u50']:
    calibrated_probs[(m, 'None')] = {
        'val': raw_val_probs[m],
        'test': raw_test_probs[m],
    }
    
    print(f'[Calibration] Fitting Platt scaling on Validation for {m}...')
    platt = LogisticRegression(C=1e5, solver='lbfgs', random_state=42)
    val_p_reshaped = raw_val_probs[m].reshape(-1, 1)
    test_p_reshaped = raw_test_probs[m].reshape(-1, 1)
    platt.fit(val_p_reshaped, y_val)
    
    calibrated_probs[(m, 'Platt')] = {
        'val': platt.predict_proba(val_p_reshaped)[:, 1],
        'test': platt.predict_proba(test_p_reshaped)[:, 1],
    }
    
    print(f'[Calibration] Fitting Isotonic Regression on Validation for {m}...')
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_val_probs[m], y_val)
    
    calibrated_probs[(m, 'Isotonic')] = {
        'val': iso.transform(raw_val_probs[m]),
        'test': iso.transform(raw_test_probs[m]),
    }

calibrated_probs[('Alperin', 'None')] = {
    'val': raw_val_probs['Alperin'],
    'test': raw_test_probs['Alperin'],
}

metrics_rows = []
threshold_rows = []
test_pred_df = pd.DataFrame({
    'CVE_ID': test_df['CVE_ID'].values,
    'Observation_Date': test_df['Observation_Date'].values,
    'target': y_test,
})

def deterministic_sort(df, prob_col='predicted_prob'):
    return df.sort_values(
        by=[prob_col, 'CVE_ID', 'Observation_Date'],
        ascending=[False, True, True],
        kind='mergesort',
    )

for (m, method), prob_dict in calibrated_probs.items():
    p_val = prob_dict['val']
    p_test = prob_dict['test']
    
    exp_label = f'{m}_{method}'
    test_pred_df[f'predicted_prob_{exp_label}'] = p_test
    test_pred_df[f'predicted_label_{exp_label}'] = (p_test > 0.5).astype(int)
    
    m_test = compute_metrics(y_test, p_test, (p_test > 0.5).astype(int))
    
    df_test_temp = pd.DataFrame({
        'CVE_ID': test_df['CVE_ID'].values,
        'Observation_Date': test_df['Observation_Date'].values,
        'target': y_test,
        'predicted_prob': p_test,
    })
    df_sorted = deterministic_sort(df_test_temp)
    max_tie = int(df_sorted['predicted_prob'].value_counts().max())
    
    p100 = float(df_sorted.head(100)['target'].sum() / 100.0)
    r100 = float(df_sorted.head(100)['target'].sum() / y_test.sum())
    p500 = float(df_sorted.head(500)['target'].sum() / 500.0)
    r500 = float(df_sorted.head(500)['target'].sum() / y_test.sum())
    
    metrics_rows.append({
        'model': m,
        'calibration': method,
        'split': 'test',
        'ROC_AUC': m_test['ROC_AUC'],
        'PR_AUC': m_test['PR_AUC'],
        'Brier': m_test['Brier'],
        'ECE': m_test['ECE'],
        'Precision': m_test['Precision'],
        'Recall': m_test['Recall'],
        'F1': m_test['F1'],
        'TP': m_test['TP'],
        'FP': m_test['FP'],
        'TN': m_test['TN'],
        'FN': m_test['FN'],
        'Precision@100': p100,
        'Recall@100': r100,
        'Precision@500': p500,
        'Recall@500': r500,
        'max_tie_count': max_tie,
    })
    
    if p_val is not None:
        for thr in THRESHOLDS:
            pred_l = (p_val > thr).astype(int)
            tp = int(((pred_l == 1) & (y_val == 1)).sum())
            fp = int(((pred_l == 1) & (y_val == 0)).sum())
            tn = int(((pred_l == 0) & (y_val == 0)).sum())
            fn = int(((pred_l == 0) & (y_val == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            threshold_rows.append({
                'model': m,
                'calibration': method,
                'threshold': thr,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
            })

metrics_df = pd.DataFrame(metrics_rows)
metrics_path = OUTPUT_DIR / 'undersampled_calibration_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)

thresholds_df = pd.DataFrame(threshold_rows)
thresholds_path = OUTPUT_DIR / 'undersampled_calibration_thresholds.csv'
thresholds_df.to_csv(thresholds_path, index=False)

test_pred_path = OUTPUT_DIR / 'undersampled_calibration_predictions_test.csv'
test_pred_df.to_csv(test_pred_path, index=False)

plt.figure(figsize=(9, 7))
for m in ['u10', 'u50']:
    for method in ['None', 'Platt', 'Isotonic']:
        p_test = calibrated_probs[(m, method)]['test']
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, p_test, n_bins=10, strategy='quantile')
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{m} ({method})')

plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Reliability / Calibration Curves (2024 Test Set)')
plt.legend()
plt.tight_layout()
curve_path = OUTPUT_DIR / 'calibration_u10_u50.png'
plt.savefig(curve_path, dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sub_df = metrics_df[metrics_df['model'].isin(['u10', 'u50'])].copy()
sub_df['label'] = sub_df['model'] + ' (' + sub_df['calibration'] + ')'

axes[0].barh(sub_df['label'], sub_df['ECE'], color='skyblue')
axes[0].set_xlabel('Expected Calibration Error (ECE)')
axes[0].set_title('Test ECE Before and After Calibration')

axes[1].barh(sub_df['label'], sub_df['Brier'], color='salmon')
axes[1].set_xlabel('Brier Score')
axes[1].set_title('Test Brier Score Before and After Calibration')

plt.tight_layout()
bar_path = OUTPUT_DIR / 'calibration_comparison.png'
plt.savefig(bar_path, dpi=300)
plt.close()

report_path = REPORT_DIR / 'undersampled_calibration_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('# Undersampled Model Post-Hoc Calibration Report\n\n')
    f.write('This report presents calibration results for u10 and u50.\n\n')
    f.write('## Generated Outputs\n')
    f.write(f'- Calibration Metrics: {metrics_path.name}\n')
    f.write(f'- Validation Thresholds: {thresholds_path.name}\n')
    f.write(f'- Test Predictions: {test_pred_path.name}\n')

print(f'\n[Saved] calibration metrics -> {metrics_path}')
print(f'[Saved] calibration thresholds -> {thresholds_path}')
print(f'[Saved] test predictions -> {test_pred_path}')
print(f'[Saved] reliability curves plot -> {curve_path}')
print(f'[Saved] calibration comparison plot -> {bar_path}')
print(f'[Saved] calibration report -> {report_path}')
