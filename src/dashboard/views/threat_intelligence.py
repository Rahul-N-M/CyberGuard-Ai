import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import empty_state, format_probability, page_header
from src.dashboard.data import load_calibrated_predictions, load_threat_intelligence


def render() -> None:
    page_header("Threat intelligence", "Local NVD, FIRST EPSS, and CISA KEV data joined by CVE identifier.")
    data = load_threat_intelligence()
    predictions = load_calibrated_predictions()
    data = data.merge(predictions, on="cve_id", how="left")

    with st.expander("Filters", expanded=True):
        data["severity"] = data["severity"].fillna("Unknown").astype(str)
        severity_values = sorted(data["severity"].unique())
        chosen_severities = st.multiselect("Severity", severity_values, default=severity_values)
        kev_filter = st.selectbox("CISA KEV status", ["All", "KEV only", "Not in KEV"])
        min_cvss, max_cvss = st.slider("CVSS range", 0.0, 10.0, (0.0, 10.0), 0.1)
        min_prob, max_prob = st.slider("EPSS score range", 0.0, 1.0, (0.0, 1.0), 0.01)
        date_values = data["published_date"].dropna()
        if not date_values.empty:
            date_range = st.date_input(
                "Published date", value=(date_values.min().date(), date_values.max().date()),
                min_value=date_values.min().date(), max_value=date_values.max().date(),
            )
        else:
            date_range = None

    filtered = data[data["severity"].astype(str).isin(chosen_severities)].copy()
    cvss = pd.to_numeric(filtered["cvss_score"], errors="coerce")
    epss = pd.to_numeric(filtered["epss_score"], errors="coerce")
    cvss_mask = cvss.between(min_cvss, max_cvss)
    epss_mask = epss.between(min_prob, max_prob)
    if min_cvss == 0 and max_cvss == 10:
        cvss_mask |= cvss.isna()
    if min_prob == 0 and max_prob == 1:
        epss_mask |= epss.isna()
    filtered = filtered[cvss_mask & epss_mask]
    kev = pd.to_numeric(filtered["kev"], errors="coerce")
    if kev_filter == "KEV only":
        filtered = filtered[kev.eq(1)]
    elif kev_filter == "Not in KEV":
        filtered = filtered[kev.eq(0)]
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        date_mask = filtered["published_date"].between(start, end, inclusive="left")
        all_dates = data["published_date"].dropna()
        if not all_dates.empty and start.date() <= all_dates.min().date() and (end - pd.Timedelta(days=1)).date() >= all_dates.max().date():
            date_mask |= filtered["published_date"].isna()
        filtered = filtered[date_mask]

    st.caption(f"{len(filtered):,} CVEs match the selected filters.")
    if "model_probability" not in filtered.columns or filtered["model_probability"].notna().sum() == 0:
        empty_state(
            "Calibrated model probabilities are not available for this dataset.",
            "EPSS is displayed as threat intelligence, not as a CyberGuard model prediction. Run the saved v3 pipeline to generate calibrated predictions.",
        )

    table = filtered.copy()
    table["KEV status"] = pd.to_numeric(table["kev"], errors="coerce").map({1: "Yes", 0: "No"}).fillna("Unavailable")
    table["CVSS"] = pd.to_numeric(table["cvss_score"], errors="coerce")
    table["EPSS"] = pd.to_numeric(table["epss_score"], errors="coerce")
    table["Calibrated model probability"] = table.get("model_probability", pd.Series(index=table.index, dtype=float)).map(format_probability)
    display_columns = [
        ("cve_id", "CVE"), ("CVSS", "CVSS"), ("severity", "Severity"),
        ("Calibrated model probability", "Calibrated model P"), ("EPSS", "EPSS"),
        ("KEV status", "CISA KEV"), ("published_date", "Published"),
    ]
    if filtered.empty:
        empty_state("No vulnerability records match the current filters.")
    else:
        st.dataframe(
            table[[source for source, _ in display_columns]].rename(columns=dict(display_columns)),
            use_container_width=True, hide_index=True,
        )
        chart_data = table.dropna(subset=["EPSS", "CVSS"])
        if not chart_data.empty:
            chart = px.scatter(
                chart_data, x="EPSS", y="CVSS", color="severity", hover_name="cve_id",
                symbol="KEV status", template="plotly_dark", labels={"EPSS": "EPSS score", "CVSS": "CVSS score"},
            )
            chart.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(chart, use_container_width=True)
