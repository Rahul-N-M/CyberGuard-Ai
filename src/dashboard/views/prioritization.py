import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import empty_state, format_probability, metric_row, page_header
from src.dashboard.data import load_calibrated_predictions
from src.dashboard.risk import actionable_risk_actions


def render() -> None:
    page_header(
        "Vulnerability prioritization matrix",
        "Actionable enterprise flaws ranked by composite business risk: P(Exploit) × Business Impact × Exposure × (CVSS / 10).",
    )
    raw_data = actionable_risk_actions().copy()
    predictions = load_calibrated_predictions()
    data = raw_data.merge(predictions, on="cve_id", how="left")

    # 1. Backlog Summary KPIs
    total_flaws = len(data)
    kev_count = int(pd.to_numeric(data["kev"], errors="coerce").eq(1).sum())
    high_impact_count = int(pd.to_numeric(data["business_impact_score"], errors="coerce").ge(8.0).sum())
    total_risk = float(pd.to_numeric(data["risk_score"], errors="coerce").sum())

    metric_row([
        ("Actionable Flaws", f"{total_flaws:,}", "Across 30 ShopEasy assets"),
        ("Active In Wild (KEV)", f"{kev_count:,}", "Confirmed weaponized CVEs"),
        ("Critical Assets (Impact ≥8)", f"{high_impact_count:,}", "Tier-1 enterprise infrastructure"),
        ("Total Risk Surface", f"{total_risk:,.1f}", "Cumulative modeled business risk"),
    ])

    # 2. Threat vs. Technical Severity Plot (Authentic Empirical Scatter)
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    st.subheader("📊 Threat Likelihood vs. CVSS Technical Severity")
    st.caption("Illustrates why CVSS alone fails: scores of 9.0–10.0 span both active zero-days and benign unexploited flaws.")

    plot_sample = pd.concat([
        data.nlargest(300, "risk_score"),
        data.sample(min(300, len(data)), random_state=42),
    ]).drop_duplicates(subset=["cve_id", "asset_id"]).copy()

    plot_sample["KEV Status"] = pd.to_numeric(plot_sample["kev"], errors="coerce").map({1: "Weaponized (In KEV)", 0: "Not in KEV"}).fillna("Not in KEV")
    plot_sample["EPSS Probability"] = pd.to_numeric(plot_sample["epss_score"], errors="coerce").fillna(0.01)
    plot_sample["CVSS Score"] = pd.to_numeric(plot_sample["cvss_score"], errors="coerce").fillna(5.0)

    fig_threat = px.scatter(
        plot_sample,
        x="EPSS Probability",
        y="CVSS Score",
        color="severity",
        symbol="KEV Status",
        hover_name="cve_id",
        hover_data={
            "asset_name": True,
            "department_name": True,
            "business_impact_score": True,
            "risk_score": ":.3f",
            "remediation_hours": ":.1f",
            "EPSS Probability": ":.4f",
            "CVSS Score": ":.1f",
        },
        template="plotly_dark",
        labels={"EPSS Probability": "Exploit Likelihood (EPSS Score)", "CVSS Score": "Technical Severity (CVSS)"},
    )
    fig_threat.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_threat, use_container_width=True)

    # 3. Filter Controls
    st.subheader("📋 Flaw Catalog & Filter Controls")
    search = st.text_input("Search CVE, asset, or department", placeholder="e.g. CVE-2023-... or Payments", key="prio_search")

    with st.expander("Advanced Attribute Filters", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            data["severity"] = data["severity"].fillna("Unknown").astype(str)
            severity_values = sorted(data["severity"].unique())
            chosen_severities = st.multiselect("Severity filter", severity_values, default=severity_values, key="prio_sev")
            
            depts = sorted(data["department_name"].dropna().unique()) if "department_name" in data.columns else []
            chosen_depts = st.multiselect("Department filter", depts, default=depts, key="prio_dept")
        with col_f2:
            kev_opt = st.selectbox("CISA KEV Status", ["All", "KEV Only", "Non-KEV"], key="prio_kev")
            min_score, max_score = st.slider("CVSS score range", 0.0, 10.0, (0.0, 10.0), 0.1, key="priority_cvss")

    filtered = data.copy()
    filtered = filtered[filtered["severity"].astype(str).isin(chosen_severities)]
    if chosen_depts and "department_name" in filtered.columns:
        filtered = filtered[filtered["department_name"].isin(chosen_depts)]
    if kev_opt == "KEV Only":
        filtered = filtered[pd.to_numeric(filtered["kev"], errors="coerce").eq(1)]
    elif kev_opt == "Non-KEV":
        filtered = filtered[pd.to_numeric(filtered["kev"], errors="coerce").fillna(0).eq(0)]

    cvss_values = pd.to_numeric(filtered["cvss_score"], errors="coerce")
    cvss_mask = cvss_values.between(min_score, max_score)
    if min_score == 0 and max_score == 10:
        cvss_mask |= cvss_values.isna()
    filtered = filtered[cvss_mask]

    if search.strip():
        needle = search.strip().casefold()
        columns = [col for col in ("cve_id", "asset_name", "department_name") if col in filtered.columns]
        match = pd.Series(False, index=filtered.index)
        for column in columns:
            match |= filtered[column].fillna("").astype(str).str.casefold().str.contains(needle, regex=False)
        filtered = filtered[match]

    filtered = filtered.sort_values("risk_score", ascending=False)

    if filtered.empty:
        empty_state("No actionable asset/CVE instances match these filters.")
        return

    view = filtered.copy()
    view["Model P(exploit)"] = view.get("model_probability", pd.Series(index=view.index, dtype=float)).map(format_probability)
    view["Optimizer exploit proxy"] = view["optimizer_exploit_proxy"].map(format_probability)
    view["CVSS"] = view["cvss_score"]
    view["Business impact"] = view["business_impact_score"]
    view["Exposure multiplier"] = view["internet_exposure"].map({1: "1.5× internet", 0: "1.0× internal", True: "1.5× internet", False: "1.0× internal"}).fillna("Unknown")
    view["Business risk"] = view["risk_score"].round(4)
    view["Recommended action"] = view["remediation_type"].fillna("Review remediation plan")
    columns = [
        "cve_id", "Model P(exploit)", "Optimizer exploit proxy", "CVSS", "asset_name",
        "department_name", "Business impact", "Exposure multiplier", "remediation_hours",
        "Business risk", "Recommended action",
    ]
    headers = [
        "CVE", "Calibrated model P", "Risk-formula exploit proxy", "CVSS", "Asset",
        "Department", "Impact", "Exposure", "Effort (h)", "Modeled risk", "Action",
    ]
    st.caption(f"{len(view):,} actionable instances · sorted by modeled business risk. Formula: P(Exploit) × Biz_Impact × Exposure × (CVSS/10).")
    st.dataframe(view[columns].rename(columns=dict(zip(columns, headers))), use_container_width=True, hide_index=True)

    csv_data = view[columns].to_csv(index=False)
    st.download_button(
        label="📥 Export prioritized vulnerabilities (CSV)",
        data=csv_data,
        file_name="cyberguard_prioritized_vulnerabilities.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # 4. Individual Flaw Detail Inspector
    options = view.index.tolist()
    selected_idx = st.selectbox(
        "Inspect a vulnerability detail",
        options,
        format_func=lambda idx: f"{view.loc[idx, 'cve_id']} · {view.loc[idx, 'asset_name']} · risk {view.loc[idx, 'risk_score']:.3f}",
    )
    selected = view.loc[selected_idx]
    st.subheader(f"🔍 {selected['cve_id']} · {selected.get('asset_name', 'Asset details')}")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.write(f"**Asset:** {selected.get('asset_name', 'Unavailable')}")
        st.write(f"**Department:** {selected.get('department_name', 'Unavailable')}")
        st.write(f"**Criticality:** {selected.get('criticality', 'Unavailable')}")
    with col_d2:
        st.write(f"**CVSS:** {selected.get('cvss_score', 'N/A')} ({selected.get('severity', 'Unknown')})")
        st.write(f"**EPSS Score:** {format_probability(selected.get('epss_score'))}")
        st.write(f"**CISA KEV Listed:** {'Yes (Confirmed weaponization)' if selected.get('kev') == 1 else 'No'}")
    with col_d3:
        st.write(f"**Business Impact:** {selected.get('business_impact_score')}/10")
        st.write(f"**Remediation Effort:** {selected.get('remediation_hours')} hours")
        st.write(f"**Action Required:** {selected.get('remediation_type', 'Patch')}")

    st.info(
        f"**Business Risk Formulation:** Exploit Proxy ({format_probability(selected['optimizer_exploit_proxy'])}) "
        f"× Business Impact ({selected['business_impact_score']}) "
        f"× Exposure ({'1.5' if selected['internet_exposure'] else '1.0'}) "
        f"× (CVSS {selected.get('cvss_score', 10.0)} / 10) = **{selected['risk_score']:.4f}**"
    )



