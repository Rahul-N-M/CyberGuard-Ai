import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import empty_state, page_header
from src.dashboard.data import load_assets
from src.dashboard.risk import load_risk_actions


def render() -> None:
    page_header("Enterprise asset inventory", "ShopEasy is a simulated enterprise archetype from the repository, not a real customer.")
    assets = load_assets()
    risks = load_risk_actions()
    if assets.empty:
        empty_state("The enterprise asset inventory is not available.")
        return
    department_filter = st.multiselect("Department", sorted(assets["department_name"].dropna().unique()))
    exposure_filter = st.selectbox("Network exposure", ["All", "Internet-facing", "Internal"])
    filtered = assets.copy()
    if department_filter:
        filtered = filtered[filtered["department_name"].isin(department_filter)]
    if exposure_filter != "All":
        exposed = exposure_filter == "Internet-facing"
        filtered = filtered[pd.to_numeric(filtered["internet_exposure"], errors="coerce").eq(int(exposed))]

    by_asset = risks.groupby("asset_id", as_index=False).agg(
        modeled_risk=("risk_score", "sum"),
        mapped_instances=("cve_id", "size"),
        kev_instances=("kev", lambda values: pd.to_numeric(values, errors="coerce").eq(1).sum()),
    )
    filtered = filtered.merge(by_asset, on="asset_id", how="left")
    st.caption(f"{len(assets)} configured assets across {assets['department_name'].nunique()} departments.")
    display_columns = [
        col for col in (
            "asset_id", "asset_name", "department_name", "asset_type", "criticality",
            "internet_exposure", "business_impact_score", "environment", "installed_software",
            "mapped_instances", "kev_instances", "modeled_risk",
        ) if col in filtered.columns
    ]
    st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)

    st.subheader("Department modeled-risk breakdown")
    department_risk = risks.groupby("department_name", dropna=False).agg(
        modeled_risk=("risk_score", "sum"),
        mapped_instances=("cve_id", "size"),
        internet_exposed_instances=("internet_exposure", lambda values: pd.to_numeric(values, errors="coerce").eq(1).sum()),
    ).reset_index()
    if department_risk.empty:
        empty_state("No department risk mapping is available.")
    else:
        chart = px.bar(
            department_risk.sort_values("modeled_risk", ascending=False),
            x="department_name", y="modeled_risk", color="internet_exposed_instances",
            template="plotly_dark", labels={
                "department_name": "Department", "modeled_risk": "Modeled risk",
                "internet_exposed_instances": "Internet-exposed instances",
            },
        )
        chart.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_tickangle=-25)
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("🔍 Asset Vulnerability Drill-Down")
    selected_asset_id = st.selectbox(
        "Select an asset to view its vulnerability surface:",
        filtered["asset_id"].tolist(),
        format_func=lambda aid: f"{aid} · {filtered.loc[filtered['asset_id'] == aid, 'asset_name'].values[0]} ({filtered.loc[filtered['asset_id'] == aid, 'department_name'].values[0]})",
    )
    if selected_asset_id:
        asset_vulns = risks[risks["asset_id"] == selected_asset_id].copy()
        if not asset_vulns.empty:
            st.caption(f"{len(asset_vulns)} vulnerabilities mapped to {selected_asset_id}. Showing top prioritized flaws:")
            drill_cols = [col for col in ("cve_id", "severity", "cvss_score", "epss_score", "kev", "remediation_hours", "risk_score", "status") if col in asset_vulns.columns]
            st.dataframe(asset_vulns[drill_cols].sort_values("risk_score", ascending=False).head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No active vulnerabilities mapped to this asset.")

