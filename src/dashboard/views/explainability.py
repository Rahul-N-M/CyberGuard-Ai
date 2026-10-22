import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, format_probability, page_header
from src.dashboard.data import load_calibrated_predictions
from src.dashboard.risk import optimize_budget


BUDGETS = {5: "5h · Emergency", 10: "10h · DevOps", 20: "20h · Sprint", 40: "40h · Engineering week"}


def render() -> None:
    page_header("Explainability & security insights", "Threat evidence, business context, and exact optimizer rationale for each recommended action.")
    budget = st.select_slider("Explain recommendations for budget", options=list(BUDGETS), value=10, format_func=lambda value: BUDGETS[value], key="explain_budget")
    result = optimize_budget(float(budget))
    selected = result["selected"]
    if selected.empty:
        empty_state("No remediation action was selected for this budget.")
        return
    predictions = load_calibrated_predictions()
    selected = selected.merge(predictions, on="cve_id", how="left")
    selected["option_label"] = selected.apply(
        lambda row: f"{row['cve_id']} · {row.get('asset_name', row.get('asset_id', 'asset'))}",
        axis=1,
    )
    idx = st.selectbox("Recommended fix", selected.index.tolist(), format_func=lambda i: selected.loc[i, "option_label"])
    row = selected.loc[idx]

    threat_col, business_col, solver_col = st.columns(3)
    with threat_col:
        st.markdown("#### 1 · Threat intelligence")
        probability = row.get("model_probability")
        kev_value = pd.to_numeric(pd.Series([row.get("kev")]), errors="coerce").iloc[0]
        if pd.isna(probability):
            st.write("**Calibrated model probability:** unavailable")
        else:
            st.write(f"**Calibrated model probability:** {format_probability(probability)}")
        st.write(f"**EPSS:** {format_probability(row.get('epss_score'))}")
        kev_status = "Listed" if kev_value == 1 else ("Not listed" if kev_value == 0 else "Unavailable")
        st.write(f"**CISA KEV:** {kev_status}")
        st.write(f"**CVSS:** {row.get('cvss_score', 'Unavailable')} · {row.get('severity', 'Unknown severity')}")
    with business_col:
        st.markdown("#### 2 · Business context")
        st.write(f"**Asset:** {row.get('asset_name', row.get('asset_id', 'Unavailable'))}")
        st.write(f"**Department:** {row.get('department_name', 'Unavailable')}")
        st.write(f"**Business impact:** {row.get('business_impact_score', 'Unavailable')} / 10")
        st.write(f"**Exposure:** {'Internet-facing (1.5×)' if bool(row.get('internet_exposure')) else 'Internal (1.0×)'}")
    with solver_col:
        st.markdown("#### 3 · Optimization rationale")
        st.write(f"**Budget:** {budget}h · **Effort:** {row.get('remediation_hours')}h")
        st.write(f"**Modeled risk addressed:** {row.get('risk_score', 0):.3f}")
        st.write(f"**Solver:** `{result['solver']}`")
        st.write(
            f"Selected as part of the maximum modeled-risk allocation that fits the hard {budget}h limit; "
            "alternatives were evaluated jointly rather than by CVSS rank alone."
        )
    st.info(
        "Predicted exploitation likelihood is not a guarantee that exploitation will occur. "
        "EPSS and calibrated model probabilities are signals; the optimizer uses the repository's documented "
        "EPSS + KEV business-risk formula, not the model probability."
    )
