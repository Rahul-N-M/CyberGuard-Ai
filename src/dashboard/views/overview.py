import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import empty_state, metric_row, page_header
from src.dashboard.data import load_calibrated_predictions, latest_data_refresh, load_threat_intelligence
from src.dashboard.risk import actionable_risk_actions, load_risk_actions, optimize_budget


def render() -> None:
    page_header("Executive overview", "Enterprise exposure, exploitation signals, and budget-aware remediation.")
    actions = load_risk_actions()
    actionable = actionable_risk_actions()
    threat = load_threat_intelligence()
    threat = threat.merge(load_calibrated_predictions(), on="cve_id", how="left")
    budget_labels = {5: "5h · Emergency", 10: "10h · DevOps", 20: "20h · Sprint", 40: "40h · Engineering week"}
    selected = st.selectbox("Remediation budget", list(budget_labels), format_func=lambda hours: budget_labels[hours], key="overview_budget")
    result = optimize_budget(float(selected))

    cve_count = threat["cve_id"].nunique()
    kev_ids = threat.loc[pd.to_numeric(threat["kev"], errors="coerce").eq(1), "cve_id"].nunique()
    high_ids = threat.loc[threat["severity"].astype(str).str.upper().isin(["HIGH", "CRITICAL"]), "cve_id"].nunique()
    total_risk = float(actions["risk_score"].sum())
    refresh = latest_data_refresh()
    refresh_text = refresh.strftime("%Y-%m-%d %H:%M UTC") if refresh is not None else "Unknown"
    metric_row([
        ("Unique CVEs", f"{cve_count:,}", "Local NVD export"),
        ("CISA KEV", f"{kev_ids:,}", "Known-exploited CVEs in local data"),
        ("High / critical", f"{high_ids:,}", "NVD CVSS severity"),
        ("Actionable instances", f"{len(actionable):,}", "Open or in-progress asset/CVE pairs"),
        ("Modeled enterprise risk", f"{total_risk:,.1f}", "Composite score; not financial loss"),
        ("Modeled risk addressed", f"{result['exact_risk']:,.1f}", f"{selected}h budget · exact optimizer"),
    ])
    st.caption(f"Latest available source file modification: {refresh_text}. Data is local; no live feeds are queried.")

    left, right = st.columns(2)
    with left:
        st.subheader("EPSS score distribution")
        epss = pd.to_numeric(threat["epss_score"], errors="coerce").dropna()
        if epss.empty:
            empty_state("EPSS scores are unavailable in the local exports.")
        else:
            chart = px.histogram(x=epss, nbins=35, labels={"x": "EPSS score", "y": "CVEs"}, template="plotly_dark")
            chart.update_layout(margin=dict(l=0, r=0, t=15, b=0), showlegend=False)
            st.plotly_chart(chart, use_container_width=True)
        if threat["model_probability"].notna().sum() == 0:
            empty_state(
                "Calibrated CyberGuard model probabilities are not present.",
                "Generate outputs/v3_predictions_test.csv by running the existing v3 pipeline to populate model-probability views.",
            )
    with right:
        st.subheader("Modeled risk by department and severity")
        if "department_name" not in actions.columns:
            empty_state("Department mapping is not available in the enterprise export.")
        else:
            plot_data = actions.copy()
            plot_data["severity"] = plot_data["severity"].fillna("Unknown")
            by_dept = plot_data.groupby(["department_name", "severity"], dropna=False)["risk_score"].sum().reset_index()
            chart = px.bar(
                by_dept, x="department_name", y="risk_score", color="severity",
                category_orders={"severity": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "Unknown"]},
                template="plotly_dark", labels={"department_name": "Department", "risk_score": "Modeled risk"},
            )
            chart.update_layout(margin=dict(l=0, r=0, t=15, b=0), xaxis_tickangle=-25, legend_title_text="Severity")
            st.plotly_chart(chart, use_container_width=True)
    st.caption(
        "Risk is the repository's modeled composite (EPSS + documented KEV bonus, business impact, exposure, and CVSS). "
        "It is not a prediction of breach probability or financial loss."
    )
