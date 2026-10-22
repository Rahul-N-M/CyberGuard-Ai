import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import empty_state, page_header
from src.dashboard.data import load_model_metrics


def _metric_rows(name: str, metrics: dict) -> list[dict]:
    return [
        {"Evaluation": name, "Metric": key.replace("_", " "), "Value": value}
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key.lower() in {
            "roc_auc", "pr_auc", "f1", "f1_score", "precision", "recall",
            "brier", "brier_score", "ece",
        }
    ]


def _section_metrics(label: str, data: dict) -> None:
    if not isinstance(data, dict):
        return
    rows = _metric_rows(label, data)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _comparison_table(models: dict[str, dict]) -> None:
    metrics = ("ROC_AUC", "PR_AUC", "F1", "F1_Score", "Precision", "Recall")
    rows = []
    for model, values in models.items():
        row = {"Model": model}
        for metric in metrics:
            value = values.get(metric)
            if value is not None:
                row[metric.replace("_Score", "").replace("_", "-")] = value
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


def render() -> None:
    page_header("Model performance", "Saved evaluation outputs only. Metrics from different datasets and protocols are intentionally separated.")
    outputs = load_model_metrics()
    if not outputs:
        empty_state("No saved evaluation metric files were found.", "Run the existing evaluation pipeline and save its outputs before viewing performance.")
        return

    v3 = outputs.get("v3_metrics.json")
    st.subheader("Temporal evaluation · 2022–2024")
    if not v3:
        empty_state("The saved v3 temporal evaluation metrics are not available.")
    else:
        st.caption(f"{v3.get('model', 'CyberGuard v3')} · saved test metrics")
        threshold_group = v3.get("threshold_0.5", {})
        test = threshold_group.get("test", {}) if isinstance(threshold_group, dict) else {}
        _section_metrics("2024 temporal test · threshold 0.5", test)
        _section_metrics("CVE-level 2024 temporal evaluation", {
            "ROC_AUC": v3.get("cve_level_roc_auc"),
            "PR_AUC": v3.get("cve_level_pr_auc"),
        })
        optimal = v3.get("threshold_optimal", {})
        if isinstance(optimal, dict) and isinstance(optimal.get("test"), dict):
            with st.expander(f"Validation-selected threshold ({v3.get('best_threshold_from_val', 'saved')})"):
                _section_metrics("2024 temporal test · validation-selected threshold", optimal["test"])

        if isinstance(test, dict):
            diagnostics = {
                key: value for key, value in test.items()
                if any(word in key.lower() for word in ("calibr", "brier", "ece"))
                and isinstance(value, (int, float))
            }
            if diagnostics:
                st.subheader("Calibration diagnostics")
                _section_metrics("2024 temporal test", diagnostics)
            else:
                empty_state("No saved calibration diagnostics are available for the temporal test split.")
        learners = v3.get("individual_base_val_roc", {})
        if learners:
            st.subheader("Base-learner validation ROC-AUC")
            chart = px.bar(
                x=list(learners), y=list(learners.values()), template="plotly_dark",
                labels={"x": "Learner", "y": "Validation ROC-AUC"},
            )
            chart.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(chart, use_container_width=True)
        empty_state(
            "Prediction and ranking distributions are unavailable because no v3 row/CVE prediction CSVs are present.",
            "Feature-importance output is also unavailable unless separately saved by the existing pipeline.",
        )

    st.divider()
    st.subheader("Reference benchmark · 2009–2015")
    benchmark = outputs.get("benchmark_2009_2015_comparison.json", {})
    if benchmark:
        st.caption(f"{benchmark.get('dataset', '')} · {benchmark.get('evaluation_strategy', '')}")
        _comparison_table({
            "FastEmbed empirical": benchmark.get("FastEmbed_Empirical_Run", {}),
            "CyberGuard v3 empirical": benchmark.get("CyberGuard_v3_Empirical_Run", {}),
        })
    else:
        empty_state("No saved 2009–2015 benchmark comparison is present.")

    st.divider()
    st.subheader("Reference benchmark · 2013–2018")
    reference = outputs.get("cyberguard_reference_metrics.json", {})
    reference_1318 = reference.get("dataset_2013_2018_reference", {})
    if reference_1318:
        st.caption(reference_1318.get("dataset", "Fang et al. 2013–2018 NVD reference dataset"))
        _comparison_table({
            "FastEmbed 5-fold": {"ROC_AUC": reference_1318.get("fastembed_replicated_5fold_roc_auc")},
            "CyberGuard v3 5-fold": {"ROC_AUC": reference_1318.get("cyberguard_v3_5fold_roc_auc")},
        })
    else:
        empty_state("No saved 2013–2018 CyberGuard reference metrics are present.")

    st.divider()
    st.subheader("Temporal evaluation · 2022–2024")
    fastembed = outputs.get("fastembed_metrics.json", {})
    fastembed_test = fastembed.get("threshold_0.5", {}).get("test", {})
    cyberguard_test = v3.get("threshold_0.5", {}).get("test", {}) if v3 else {}
    if fastembed_test or cyberguard_test:
        _comparison_table({
            "FastEmbed temporal test": fastembed_test,
            "CyberGuard v3 temporal test": cyberguard_test,
        })
        if v3:
            st.caption(
                f"CyberGuard CVE-level Borda ranking: ROC-AUC {v3.get('cve_level_roc_auc', 'n/a')}, "
                f"PR-AUC {v3.get('cve_level_pr_auc', 'n/a')}. This is a ranking aggregate, not a calibrated probability."
            )
    else:
        empty_state("Saved temporal FastEmbed/CyberGuard comparison metrics are unavailable.")

    st.warning(
        "The reference benchmarks use older, curated/reference datasets; the 2022–2024 results are temporal-stream evaluations "
        "with a different class prevalence and evaluation setup. Do not compare their headline scores as if they were one experiment."
    )
