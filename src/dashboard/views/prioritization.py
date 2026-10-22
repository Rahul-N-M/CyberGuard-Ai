import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, format_probability, page_header
from src.dashboard.data import load_calibrated_predictions
from src.dashboard.risk import actionable_risk_actions


def render() -> None:
    page_header("Vulnerability prioritization", "Actionable asset/CVE instances ranked by the existing business-risk model.")
    data = actionable_risk_actions().copy()
    predictions = load_calibrated_predictions()
    data = data.merge(predictions, on="cve_id", how="left")

    search = st.text_input("Search CVE, asset, or department", placeholder="e.g. CVE-2023-... or Payments")
    data["severity"] = data["severity"].fillna("Unknown").astype(str)
    severity_values = sorted(data["severity"].unique())
    chosen_severities = st.multiselect("Severity filter", severity_values, default=severity_values)
    min_score, max_score = st.slider("CVSS score", 0.0, 10.0, (0.0, 10.0), 0.1, key="priority_cvss")
    data = data[data["severity"].astype(str).isin(chosen_severities)]
    cvss_values = pd.to_numeric(data["cvss_score"], errors="coerce")
    cvss_mask = cvss_values.between(min_score, max_score)
    if min_score == 0 and max_score == 10:
        cvss_mask |= cvss_values.isna()
    data = data[cvss_mask]
    if search.strip():
        needle = search.strip().casefold()
        columns = [col for col in ("cve_id", "asset_name", "department_name") if col in data.columns]
        match = pd.Series(False, index=data.index)
        for column in columns:
            match |= data[column].fillna("").astype(str).str.casefold().str.contains(needle, regex=False)
        data = data[match]
    data = data.sort_values("risk_score", ascending=False)

    if data.empty:
        empty_state("No actionable asset/CVE instances match these filters.")
        return
    view = data.copy()
    view["Model P(exploit)"] = view.get("model_probability", pd.Series(index=view.index, dtype=float)).map(format_probability)
    view["Optimizer exploit proxy"] = view["optimizer_exploit_proxy"].map(format_probability)
    view["CVSS"] = view["cvss_score"]
    view["Business impact"] = view["business_impact_score"]
    view["Exposure multiplier"] = view["internet_exposure"].map({1: "1.5× internet", 0: "1.0× internal", True: "1.5× internet", False: "1.0× internal"}).fillna("Unknown")
    view["Business risk"] = view["risk_score"]
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
    st.caption(f"{len(view):,} actionable instances · sorted by modeled business risk. Proxy and model probability are distinct.")
    st.dataframe(view[columns].rename(columns=dict(zip(columns, headers))), use_container_width=True, hide_index=True)
    options = view.index.tolist()
    selected_idx = st.selectbox(
        "Inspect a vulnerability",
        options,
        format_func=lambda idx: f"{view.loc[idx, 'cve_id']} · {view.loc[idx, 'asset_name']} · risk {view.loc[idx, 'risk_score']:.3f}",
    )
    selected = view.loc[selected_idx]
    st.subheader(f"{selected['cve_id']} · {selected.get('asset_name', 'Asset details')}")
    st.write(
        f"Department: **{selected.get('department_name', 'Unavailable')}** · "
        f"CVSS: **{selected.get('cvss_score', float('nan'))}** · "
        f"EPSS: **{format_probability(selected.get('epss_score'))}** · "
        f"KEV: **{'Yes' if selected.get('kev') == 1 else 'No / unavailable'}**"
    )
    st.write(
        f"Modeled risk = exploit proxy **{format_probability(selected['optimizer_exploit_proxy'])}** × "
        f"business impact **{selected['business_impact_score']}** × exposure × CVSS/10. "
        f"Remediation estimate: **{selected['remediation_hours']} hours**."
    )
    if pd.isna(selected.get("model_probability")):
        st.caption("A calibrated model probability is not available for this CVE in the saved prediction outputs.")
    else:
        st.caption(f"Saved calibrated test prediction: {format_probability(selected['model_probability'])}.")
