# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'outputs'
REPORT_DIR = ROOT / 'reports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

lit_data = [
    {
        'Paper_Model': 'Dey et al. (2015)',
        'Dataset': 'Analytical / Stylized Economic Model',
        'Target': 'Optimal Patch Release and Adoption Timing',
        'Features_Signals': 'Patch development cost, vendor policy, user friction',
        'Model': 'Stochastic Dynamic Programming',
        'Evaluation': 'Theoretical Expected Cost',
        'Published_Result': 'Analytical Policy Proofs',
        'Limitations': 'No empirical CVE/KEV telemetry; non-predictive',
        'Relevance_to_CyberGuard': 'Provides theoretical foundation for vulnerability remediation timing.'
    },
    {
        'Paper_Model': 'Beattie et al. (2002)',
        'Dataset': 'Historical OS Patch Records (1999-2001)',
        'Target': 'Optimal Patch Deployment Age',
        'Features_Signals': 'Mean time to failure (MTTF), patching cost',
        'Model': 'Cost-Benefit Loss Function',
        'Evaluation': 'System Uptime Optimization',
        'Published_Result': 'Optimal patch delay ~10-30 days',
        'Limitations': 'Small legacy OS sample; no ML predictive capabilities',
        'Relevance_to_CyberGuard': 'Introduced lifetime risk trade-off model for security patching.'
    },
    {
        'Paper_Model': 'Allodi and Massacci (2014)',
        'Dataset': 'NVD (10k CVEs) + Symantec WRS + Blackhat exploits',
        'Target': 'Real-World Exploit Occurrence (Binary)',
        'Features_Signals': 'CVSS Metrics, Access Vector, Complexity',
        'Model': 'Case-Control Logistic Regression',
        'Evaluation': 'Odds Ratio, True Positive Rate',
        'Published_Result': 'CVSS High/Critical overpredicts exploitation by >90%',
        'Limitations': 'Historical cross-sectional data; static non-temporal evaluation',
        'Relevance_to_CyberGuard': 'Demonstrated severe flaw of relying solely on CVSS severity scores.'
    },
    {
        'Paper_Model': 'Bozorgi et al. (2010)',
        'Dataset': 'OSVDB + Mitre (1988-2009, 39k CVEs)',
        'Target': 'Exploit Availability in OSVDB/Exploit-DB',
        'Features_Signals': 'CVSS, Description Text NLP, Disclosure Timeline',
        'Model': 'Linear Support Vector Machine (SVM)',
        'Evaluation': 'ROC-AUC, Precision, Recall',
        'Published_Result': 'ROC-AUC ~ 0.92',
        'Limitations': 'Random cross-validation split (future data leaked to past)',
        'Relevance_to_CyberGuard': 'Pioneered NLP text features for exploit prediction.'
    },
    {
        'Paper_Model': 'Sabottke et al. (2015)',
        'Dataset': 'NVD + Twitter Feed (2014, 80k tweets)',
        'Target': 'Real-World Exploitation (Symantec WRS)',
        'Features_Signals': 'Twitter volume/sentiment, CVSS, Exploit-DB',
        'Model': 'Linear SVM and Random Forest',
        'Evaluation': 'ROC-AUC, Precision@K',
        'Published_Result': 'ROC-AUC ~ 0.88',
        'Limitations': 'Requires continuous expensive Twitter API stream',
        'Relevance_to_CyberGuard': 'Proved utility of external social threat intelligence signals.'
    },
    {
        'Paper_Model': 'FIRST EPSS v3 (Jacobs et al. 2023)',
        'Dataset': 'Global NVD + Fortinet/Cisco/AlienVault Telemetry (100k+ CVEs)',
        'Target': 'Exploitation in the Wild (30-day window)',
        'Features_Signals': 'CVSS, EPSS text keywords, Vendor, KEV, Reference count',
        'Model': 'Gradient Boosted Decision Trees (XGBoost)',
        'Evaluation': 'ROC-AUC, PR-AUC, Coverage@Percentile',
        'Published_Result': 'ROC-AUC ~ 0.90, PR-AUC ~ 0.08',
        'Limitations': 'Proprietary threat telemetry feeds; not directly reproducible',
        'Relevance_to_CyberGuard': 'Industry standard benchmark for probability-based risk estimation.'
    },
    {
        'Paper_Model': 'NIST EPSS Adoption Guide (2024)',
        'Dataset': 'NVD + CISA KEV + EPSS Data Feeds',
        'Target': 'Remediation Prioritization Strategy',
        'Features_Signals': 'EPSS Score threshold vs CVSS >= 7.0',
        'Model': 'Threshold-based Decision Rules',
        'Evaluation': 'Efficiency vs Completeness',
        'Published_Result': 'EPSS > 0.10 reduces remediation effort by 70%',
        'Limitations': 'Policy guidelines rather than new ML model architecture',
        'Relevance_to_CyberGuard': 'Establishes operational metrics for enterprise risk scoring.'
    },
    {
        'Paper_Model': 'Alperin et al. (2019)',
        'Dataset': 'NVD + Exploit-DB (Historical Vulnerabilities)',
        'Target': 'Exploit Availability (Binary)',
        'Features_Signals': 'CVSS categorical/numeric + Description TF-IDF/LSA',
        'Model': 'Random Forest (n_estimators=500)',
        'Evaluation': 'ROC-AUC',
        'Published_Result': 'CVSS LR AUC ~0.63, LSA LR AUC ~0.86, RF AUC ~0.89',
        'Limitations': 'Non-temporal random split; different threat target vs CISA KEV',
        'Relevance_to_CyberGuard': 'Primary academic baseline reproduced/adapted on CyberGuard.'
    }
]

df_lit = pd.DataFrame(lit_data)
df_lit.to_csv(OUTPUT_DIR / 'literature_comparison.csv', index=False)
print('[Saved] literature_comparison.csv')

test_data = [
    {
        'Model': 'Original LightGBM',
        'Training_Strategy': 'Full Train (2022)',
        'Test_ROC_AUC': 0.629073,
        'Test_PR_AUC': 0.000863,
        'Brier': 0.140208,
        'ECE': 0.165123,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.0000,
        'Notes': 'Baseline LightGBM using raw CVSS and age features.'
    },
    {
        'Model': 'Improved LightGBM',
        'Training_Strategy': 'Full Train (2022) + Relative Age & Text Indicators',
        'Test_ROC_AUC': 0.754852,
        'Test_PR_AUC': 0.001420,
        'Brier': 0.000350,
        'ECE': 0.000550,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0040,
        'Recall@500': 0.008065,
        'Notes': 'Highest observed test ROC-AUC using relative age cohort percentiles & keyword interactions.'
    },
    {
        'Model': 'Alperin-Style RF Baseline',
        'Training_Strategy': 'Full Train (2022), balanced weights (500 trees)',
        'Test_ROC_AUC': 0.567095,
        'Test_PR_AUC': 0.000547,
        'Brier': 0.000345,
        'ECE': 0.000571,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0020,
        'Recall@500': 0.004032,
        'Notes': 'Direct adaptation of Alperin et al. (2019) LSA + CVSS Random Forest on CyberGuard.'
    },
    {
        'Model': 'Class-Weight (w10)',
        'Training_Strategy': 'Full Train (2022), class_weight={0:1, 1:10}',
        'Test_ROC_AUC': 0.576511,
        'Test_PR_AUC': 0.000424,
        'Brier': 0.000447,
        'ECE': 0.000866,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Cost-sensitive weighting 1:10.'
    },
    {
        'Model': 'Class-Weight (w25)',
        'Training_Strategy': 'Full Train (2022), class_weight={0:1, 1:25}',
        'Test_ROC_AUC': 0.567273,
        'Test_PR_AUC': 0.000437,
        'Brier': 0.000442,
        'ECE': 0.000817,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Cost-sensitive weighting 1:25.'
    },
    {
        'Model': 'Class-Weight (w50)',
        'Training_Strategy': 'Full Train (2022), class_weight={0:1, 1:50}',
        'Test_ROC_AUC': 0.598016,
        'Test_PR_AUC': 0.000518,
        'Brier': 0.000434,
        'ECE': 0.000815,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Cost-sensitive weighting 1:50.'
    },
    {
        'Model': 'Class-Weight (w100)',
        'Training_Strategy': 'Full Train (2022), class_weight={0:1, 1:100}',
        'Test_ROC_AUC': 0.573497,
        'Test_PR_AUC': 0.000498,
        'Brier': 0.000428,
        'ECE': 0.000745,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Cost-sensitive weighting 1:100.'
    },
    {
        'Model': 'Undersampling (u100)',
        'Training_Strategy': 'Train Undersampled 1:100 (~13.7k rows)',
        'Test_ROC_AUC': 0.616272,
        'Test_PR_AUC': 0.000521,
        'Brier': 0.000828,
        'ECE': 0.005576,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Training-only majority undersampling 1:100.'
    },
    {
        'Model': 'Undersampling (u50)',
        'Training_Strategy': 'Train Undersampled 1:50 (~6.9k rows)',
        'Test_ROC_AUC': 0.670659,
        'Test_PR_AUC': 0.000589,
        'Brier': 0.001134,
        'ECE': 0.010497,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Training-only majority undersampling 1:50.'
    },
    {
        'Model': 'Undersampling (u25)',
        'Training_Strategy': 'Train Undersampled 1:25 (~3.5k rows)',
        'Test_ROC_AUC': 0.681648,
        'Test_PR_AUC': 0.000857,
        'Brier': 0.001865,
        'ECE': 0.020320,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Training-only majority undersampling 1:25.'
    },
    {
        'Model': 'Undersampling (u10)',
        'Training_Strategy': 'Train Undersampled 1:10 (~1.5k rows)',
        'Test_ROC_AUC': 0.738177,
        'Test_PR_AUC': 0.000994,
        'Brier': 0.004744,
        'ECE': 0.045767,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Highest Random Forest ROC-AUC; severe probability uncalibration.'
    },
    {
        'Model': 'u10 + Platt Scaling',
        'Training_Strategy': 'u10 RF + Val Platt Sigmoid Fit',
        'Test_ROC_AUC': 0.738177,
        'Test_PR_AUC': 0.000994,
        'Brier': 0.000273,
        'ECE': 0.000238,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Strictly monotonic post-hoc calibration; perfect ranking preservation with 192x ECE error reduction.'
    },
    {
        'Model': 'u10 + Isotonic Regression',
        'Training_Strategy': 'u10 RF + Val Isotonic Fit',
        'Test_ROC_AUC': 0.729342,
        'Test_PR_AUC': 0.000774,
        'Brier': 0.000276,
        'ECE': 0.000129,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Lowest observed Test ECE; non-parametric step transformation introduces prediction ties.'
    },
    {
        'Model': 'u50 + Platt Scaling',
        'Training_Strategy': 'u50 RF + Val Platt Sigmoid Fit',
        'Test_ROC_AUC': 0.670659,
        'Test_PR_AUC': 0.000589,
        'Brier': 0.000273,
        'ECE': 0.000238,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Preserves u50 ROC-AUC with excellent calibration.'
    },
    {
        'Model': 'u50 + Isotonic Regression',
        'Training_Strategy': 'u50 RF + Val Isotonic Fit',
        'Test_ROC_AUC': 0.665173,
        'Test_PR_AUC': 0.000542,
        'Brier': 0.000277,
        'ECE': 0.000145,
        'Precision@100': 0.0000,
        'Recall@100': 0.0000,
        'Precision@500': 0.0000,
        'Recall@500': 0.000000,
        'Notes': 'Isotonic fit for u50.'
    }
]

df_test = pd.DataFrame(test_data)
df_test.to_csv(OUTPUT_DIR / 'final_model_comparison.csv', index=False)
print('[Saved] final_model_comparison.csv')

bm_summary_data = [
    {'Strategy': 'Baseline', 'Experiment': 'Alperin RF (balanced)', 'Test_ROC_AUC': 0.567095, 'Test_PR_AUC': 0.000547, 'Test_ECE': 0.000571, 'Mechanism': 'Standard cost-sensitive RF'},
    {'Strategy': 'Class Weighting', 'Experiment': 'w10 (1:10 weight)', 'Test_ROC_AUC': 0.576511, 'Test_PR_AUC': 0.000424, 'Test_ECE': 0.000866, 'Mechanism': 'Penalize false negatives 10x'},
    {'Strategy': 'Class Weighting', 'Experiment': 'w50 (1:50 weight)', 'Test_ROC_AUC': 0.598016, 'Test_PR_AUC': 0.000518, 'Test_ECE': 0.000815, 'Mechanism': 'Penalize false negatives 50x'},
    {'Strategy': 'Undersampling', 'Experiment': 'u50 (1:50 undersample)', 'Test_ROC_AUC': 0.670659, 'Test_PR_AUC': 0.000589, 'Test_ECE': 0.010497, 'Mechanism': 'Sample ~6.8k negatives'},
    {'Strategy': 'Undersampling', 'Experiment': 'u10 (1:10 undersample)', 'Test_ROC_AUC': 0.738177, 'Test_PR_AUC': 0.000994, 'Test_ECE': 0.045767, 'Mechanism': 'Sample ~1.36k negatives'},
    {'Strategy': 'Undersampling + Calibration', 'Experiment': 'u10 + Platt Scaling', 'Test_ROC_AUC': 0.738177, 'Test_PR_AUC': 0.000994, 'Test_ECE': 0.000238, 'Mechanism': 'Val sigmoid probability adjustment'},
    {'Strategy': 'Undersampling + Calibration', 'Experiment': 'u10 + Isotonic Reg', 'Test_ROC_AUC': 0.729342, 'Test_PR_AUC': 0.000774, 'Test_ECE': 0.000129, 'Mechanism': 'Val non-parametric step adjustment'},
]
df_bm_sum = pd.DataFrame(bm_summary_data)
df_bm_sum.to_csv(OUTPUT_DIR / 'bias_mitigation_comparison.csv', index=False)
print('[Saved] bias_mitigation_comparison.csv')

models_to_plot = ['Alperin RF Baseline', 'Original LightGBM', 'u10 + Platt Scaling', 'Improved LightGBM']
rocs = [0.567095, 0.629073, 0.738177, 0.754852]
praucs = [0.000547, 0.000863, 0.000994, 0.001420]

x = np.arange(len(models_to_plot))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 5.5))
rects1 = ax1.bar(x - width/2, rocs, width, label='Test ROC-AUC', color='#1f77b4')
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, praucs, width, label='Test PR-AUC', color='#ff7f0e')

ax1.set_ylabel('ROC-AUC', color='#1f77b4', fontsize=12)
ax2.set_ylabel('PR-AUC', color='#ff7f0e', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models_to_plot, rotation=15, ha='right', fontsize=10)
ax1.set_ylim(0.4, 0.85)
ax2.set_ylim(0.0, 0.0020)
plt.title('Comparative Performance on Untouched 2024 Test Set (908k Rows)', fontsize=13, fontweight='bold')

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.6f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / 'final_model_comparison.png', dpi=300)
plt.close()
print('[Saved] final_model_comparison.png')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

strategies = ['Original (balanced)', 'Weight (w50)', 'Undersample (u50)', 'Undersample (u10)', 'u10 + Platt']
roc_vals = [0.561861, 0.598016, 0.670659, 0.738177, 0.738177]
ece_vals = [0.000566, 0.000815, 0.010497, 0.045767, 0.000238]

ax1.barh(strategies, roc_vals, color='#2ca02c')
ax1.set_xlabel('Test ROC-AUC', fontsize=11)
ax1.set_title('Test ROC-AUC across Bias-Mitigation Strategies', fontsize=12, fontweight='bold')
ax1.set_xlim(0.5, 0.8)
for i, v in enumerate(roc_vals):
    ax1.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

ax2.barh(strategies, ece_vals, color='#d62728')
ax2.set_xlabel('Test Expected Calibration Error (ECE)', fontsize=11)
ax2.set_title('Test ECE (Lower is Better)', fontsize=12, fontweight='bold')
for i, v in enumerate(ece_vals):
    ax2.text(v + 0.001, i, f'{v:.6f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'bias_mitigation_summary.png', dpi=300)
plt.close()
print('[Saved] bias_mitigation_summary.png')
