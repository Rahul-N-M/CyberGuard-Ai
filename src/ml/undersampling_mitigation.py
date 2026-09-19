import json
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
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

from src.ml.alperin_baseline import (
    load_dataset,
    build_preprocessor,
    compute_metrics,
    save_feature_importance,
    plot_roc_pr_curves,
    evaluate_topk,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'outputs'
REPORT_DIR = ROOT / 'reports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    'original': None,
    'u100': 100,
    'u50': 50,
    'u25': 25,
    'u10': 10,
}

THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
TOPK_LIST = [100, 500]

train_df = load_dataset('train')
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

X_train_full = train_df[FEATURE_COLS].copy()
y_train_full = train_df[TARGET_COL].values

X_val = val_df[FEATURE_COLS].copy()
y_val = val_df[TARGET_COL].values

X_test = test_df[FEATURE_COLS].copy()
y_test = test_df[TARGET_COL].values

print('[Preprocessing] Fitting ColumnTransformer on X_train_full...')
preprocess = build_preprocessor()
preprocess.fit(X_train_full)
X_train_full_proc = preprocess.transform(X_train_full)
X_val_proc = preprocess.transform(X_val)
X_test_proc = preprocess.transform(X_test)
print('[Preprocessing] Done.')

cat_features = preprocess.named_transformers_['cat'].get_feature_names_out()
num_features = preprocess.named_transformers_['num'].get_feature_names_out()
lsa_features = [f'lsa_{i}' for i in range(preprocess.named_transformers_['txt'].named_steps['svd'].n_components)]
all_features = list(cat_features) + list(num_features) + lsa_features

pos_indices = np.where(y_train_full == 1)[0]
neg_indices = np.where(y_train_full == 0)[0]

metrics_rows = []
threshold_rows = []
feature_imp_rows = []
test_pred_frames = []
roc_curves = {}
pr_curves = {}

def deterministic_sort(df, prob_col='predicted_prob'):
    return df.sort_values(
        by=[prob_col, 'CVE_ID', 'Observation_Date'],
        ascending=[False, True, True],
        kind='mergesort',
    )

for exp_name, ratio in EXPERIMENTS.items():
    print(f'\n[Training] Running Undersampling Experiment: {exp_name} (ratio={ratio})...')
    
    if ratio is None:
        selected_train_indices = np.arange(len(y_train_full))
    else:
        num_neg_to_select = len(pos_indices) * ratio
        rng = np.random.RandomState(42)
        selected_neg_indices = rng.choice(neg_indices, size=num_neg_to_select, replace=False)
        selected_train_indices = np.sort(np.concatenate([pos_indices, selected_neg_indices]))
    
    X_train_sub = X_train_full_proc[selected_train_indices]
    y_train_sub = y_train_full[selected_train_indices]
    
    num_pos = int((y_train_sub == 1).sum())
    num_neg = int((y_train_sub == 0).sum())
    print(f'  Training distribution: {num_pos} positives, {num_neg} negatives (Total: {len(y_train_sub)})')

    rf = RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        random_state=42,
        class_weight=None,
        n_jobs=-1,
    )
    rf.fit(X_train_sub, y_train_sub)

    feat_imp_path = OUTPUT_DIR / f'undersampling_feature_importance_{exp_name}.csv'
    importances = rf.feature_importances_
    df_imp = pd.DataFrame({'feature': all_features, 'importance': importances})
    df_imp.sort_values('importance', ascending=False, inplace=True)
    df_imp.to_csv(feat_imp_path, index=False)
    if exp_name == 'original':
        df_imp.to_csv(OUTPUT_DIR / 'undersampling_feature_importance.csv', index=False)
    feature_imp_rows.append({'experiment': exp_name, 'path': str(feat_imp_path)})

    probs_train = rf.predict_proba(X_train_sub)[:, 1]
    labels_train = (probs_train > 0.5).astype(int)
    pred_train = pd.DataFrame({
        'CVE_ID': train_df.iloc[selected_train_indices]['CVE_ID'].values,
        'Observation_Date': train_df.iloc[selected_train_indices]['Observation_Date'].values,
        'target': y_train_sub,
        'predicted_prob': probs_train,
        'predicted_label': labels_train,
    })

    probs_val = rf.predict_proba(X_val_proc)[:, 1]
    labels_val = (probs_val > 0.5).astype(int)
    pred_val = pd.DataFrame({
        'CVE_ID': val_df['CVE_ID'].values,
        'Observation_Date': val_df['Observation_Date'].values,
        'target': y_val,
        'predicted_prob': probs_val,
        'predicted_label': labels_val,
    })

    probs_test = rf.predict_proba(X_test_proc)[:, 1]
    labels_test = (probs_test > 0.5).astype(int)
    pred_test = pd.DataFrame({
        'CVE_ID': test_df['CVE_ID'].values,
        'Observation_Date': test_df['Observation_Date'].values,
        'target': y_test,
        'predicted_prob': probs_test,
        'predicted_label': labels_test,
    })

    test_pred_frames.append(pd.DataFrame({
        'CVE_ID': test_df['CVE_ID'].values,
        'Observation_Date': test_df['Observation_Date'].values,
        'target': y_test,
        f'predicted_prob_{exp_name}': probs_test,
        f'predicted_label_{exp_name}': labels_test,
    }))

    for split_name, df in [('train', pred_train), ('validation', pred_val), ('test', pred_test)]:
        metrics = compute_metrics(df['target'].values, df['predicted_prob'].values, df['predicted_label'].values)
        metrics_row = {
            'experiment': exp_name,
            'split': split_name,
            'train_positives': num_pos,
            'train_negatives': num_neg,
            **metrics,
        }
        metrics_rows.append(metrics_row)
        
        if split_name == 'validation':
            fpr, tpr, _ = roc_curve(df['target'].values, df['predicted_prob'].values)
            precision, recall, _ = precision_recall_curve(df['target'].values, df['predicted_prob'].values)
            roc_curves[exp_name] = (fpr, tpr)
            pr_curves[exp_name] = (recall, precision)

    for thr in THRESHOLDS:
        pred_labels = (pred_val['predicted_prob'] > thr).astype(int)
        tp = int(((pred_labels == 1) & (pred_val['target'] == 1)).sum())
        fp = int(((pred_labels == 1) & (pred_val['target'] == 0)).sum())
        tn = int(((pred_labels == 0) & (pred_val['target'] == 0)).sum())
        fn = int(((pred_labels == 0) & (pred_val['target'] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        threshold_rows.append({
            'experiment': exp_name,
            'threshold': thr,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
        })

    for split_name, df in [('validation', pred_val), ('test', pred_test)]:
        df_sorted = deterministic_sort(df, prob_col='predicted_prob')
        tie_counts = df_sorted['predicted_prob'].value_counts()
        max_tie = tie_counts.max()
        for K in TOPK_LIST:
            top_k = df_sorted.head(K)
            pos_k = top_k[top_k['target'] == 1].shape[0]
            prec_k = pos_k / K
            rec_k = pos_k / df_sorted['target'].sum()
            metrics_rows.append({
                'experiment': exp_name,
                'split': f'{split_name}_topk_{K}',
                'Precision@K': prec_k,
                'Recall@K': rec_k,
                'max_tie_count': max_tie,
            })

metrics_df = pd.DataFrame(metrics_rows)
metrics_path = OUTPUT_DIR / 'undersampling_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)

thresholds_df = pd.DataFrame(threshold_rows)
thresholds_path = OUTPUT_DIR / 'undersampling_thresholds.csv'
thresholds_df.to_csv(thresholds_path, index=False)

merged_test = reduce(lambda left, right: pd.merge(left, right, on=['CVE_ID', 'Observation_Date', 'target'], how='inner'), test_pred_frames)
test_pred_path = OUTPUT_DIR / 'undersampling_test_predictions.csv'
merged_test.to_csv(test_pred_path, index=False)

plt.figure(figsize=(8, 6))
for exp, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=exp)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves (validation) – Undersampling experiments')
plt.legend()
roc_path = OUTPUT_DIR / 'undersampling_comparison.png'
plt.savefig(roc_path, dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
for exp, (rec, prec) in pr_curves.items():
    plt.plot(rec, prec, label=exp)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curves (validation) – Undersampling experiments')
plt.legend()
pr_path = OUTPUT_DIR / 'undersampling_pr_curves.png'
plt.savefig(pr_path, dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
for exp in EXPERIMENTS.keys():
    df_thr = thresholds_df[thresholds_df['experiment'] == exp]
    plt.plot(df_thr['threshold'], df_thr['f1'], marker='o', label=exp)
plt.xlabel('Threshold')
plt.ylabel('F1')
plt.title('F1 vs threshold (validation) – Undersampling experiments')
plt.legend()
thr_path = OUTPUT_DIR / 'undersampling_threshold_curves.png'
plt.savefig(thr_path, dpi=300)
plt.close()

report_path = REPORT_DIR / 'undersampling_bias_mitigation_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('# Undersampling Bias Mitigation Report\n\n')
    f.write('This report presents the results of training-only majority-class undersampling on the Alperin baseline.\n\n')
    f.write('## Generated Outputs\n')
    f.write(f'- Metrics CSV: {metrics_path.name}\n')
    f.write(f'- Thresholds CSV: {thresholds_path.name}\n')
    f.write(f'- Test Predictions CSV: {test_pred_path.name}\n')

print(f'\n[Saved] undersampling metrics -> {metrics_path}')
print(f'[Saved] undersampling thresholds -> {thresholds_path}')
print(f'[Saved] undersampling comparison plot -> {roc_path}')
print(f'[Saved] undersampling PR curves -> {pr_path}')
print(f'[Saved] undersampling threshold curves -> {thr_path}')
print(f'[Saved] undersampling test predictions -> {test_pred_path}')
print(f'[Saved] undersampling report -> {report_path}')
