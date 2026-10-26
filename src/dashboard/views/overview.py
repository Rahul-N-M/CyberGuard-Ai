import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import empty_state, format_probability, metric_row, page_header
from src.dashboard.data import (
    load_calibrated_predictions,
    load_enterprise_actions,
    load_threat_intelligence,
    latest_data_refresh,
)
from src.dashboard.risk import actionable_risk_actions, load_risk_actions, optimize_budget


BUDGETS = {
    5: "5h · Emergency Response",
    10: "10h · DevOps Mini-Sprint",
    20: "20h · Standard Sprint Allocation",
    40: "40h · Full Engineering Week",
}


def render() -> None:
    page_header(
        "Executive overview & enterprise simulator",
        "Closed-loop cybersecurity risk intelligence: from exploit prediction to business-aware 0–1 knapsack remediation.",
    )
    actions = load_risk_actions()

    # 1. Problem Statement Callout Banner
    st.markdown(
        """
        <div style="background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; padding: 1rem 1.25rem; border-radius: 6px; margin-bottom: 1.5rem;">
            <div style="font-weight: 700; color: #38bdf8; margin-bottom: 0.3rem;">THE INDUSTRY CHALLENGE: WHY STANDARD CVSS PRIORITIZATION FAILS</div>
            <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.45;">
                • <b>Alert Fatigue:</b> CVSS flags <b>20% to 30%</b> of all vulnerabilities as "High" or "Critical", yet empirical research shows <b>less than 1%</b> are ever exploited in the wild (CISA KEV).<br>
                • <b>Zero Business Context:</b> CVSS scores flaws in isolation—treating a bug on an internal development machine identically to the same bug on a public payment gateway.<br>
                • <b>Ignores Engineering Budget:</b> Security teams have finite capacity (5–40 engineer-hours/sprint). Sorting top-down wastes hours on low-impact internal assets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Interactive Enterprise Sprint Simulation Controls
    st.subheader("⚡ Live Enterprise Remediation Simulation")
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        budget = st.select_slider(
            "Select Available Engineering Sprint Budget:",
            options=list(BUDGETS),
            value=10,
            format_func=lambda b: BUDGETS[b],
            key="overview_sim_budget",
        )
    with col_b2:
        st.caption("Remediation Scenario")
        st.write(f"Evaluating **18,666 actionable vulnerabilities** across 30 enterprise assets under a hard **{budget}-hour** constraint.")

    with st.spinner("Executing exact 0–1 knapsack optimization..."):
        result = optimize_budget(float(budget))

    diff_risk = result["exact_risk"] - result["baseline_risk"]

    metric_row([
        ("CyberGuard Exact Risk", f"{result['exact_risk']:.3f}", f"Solved via {result['solver']}"),
        ("Naive CVSS Greedy Risk", f"{result['baseline_risk']:.3f}", "Industry status-quo baseline"),
        ("Net Risk Reduction Advantage", f"+{diff_risk:.3f}", f"CyberGuard {result['exact_risk']:.2f} vs CVSS {result['baseline_risk']:.2f}"),
        ("Sprint Hours Used", f"{result['used_hours']:.1f}h / {budget}h", f"{result['remaining_hours']:.1f}h capacity remaining"),
    ])

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


    # 3. Side-by-Side Visualizations
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Algorithm Comparison: Risk Reduced")
        comp_df = pd.DataFrame([
            {
                "Strategy": "CyberGuard Exact 0-1 Knapsack",
                "Modeled Risk Reduced": result["exact_risk"],
                "Hours Spent": result["used_hours"],
                "Type": "Exact MILP Optimization",
            },
            {
                "Strategy": "Risk/Hour Greedy Heuristic",
                "Modeled Risk Reduced": result.get("density_risk", result["exact_risk"]),
                "Hours Spent": result.get("density_hours", result["used_hours"]),
                "Type": "Density Approximation",
            },
            {
                "Strategy": "CVSS Naive Greedy",
                "Modeled Risk Reduced": result["baseline_risk"],
                "Hours Spent": result["baseline_hours"],
                "Type": "Industry Status Quo",
            },
        ])
        chart_comp = px.bar(
            comp_df,
            x="Strategy",
            y="Modeled Risk Reduced",
            color="Strategy",
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8", "#818cf8", "#f43f5e"],
            text_auto=".2f",
        )
        chart_comp.update_layout(margin=dict(l=0, r=0, t=15, b=0), showlegend=False)
        st.plotly_chart(chart_comp, use_container_width=True)

    with col_chart2:
        st.subheader("Department Risk Reduction Allocation")
        selected = result["selected"]
        if not selected.empty and "department_name" in selected.columns:
            dept_summary = selected.groupby("department_name").agg(
                fixes=("cve_id", "count"),
                hours=("remediation_hours", "sum"),
                risk_reduced=("risk_score", "sum"),
            ).reset_index()
            chart_dept = px.bar(
                dept_summary,
                x="department_name",
                y="risk_reduced",
                color="hours",
                labels={"department_name": "Department", "risk_reduced": "Risk reduced", "hours": "Hours spent"},
                template="plotly_dark",
                color_continuous_scale="Tealgrn",
                text_auto=".2f",
            )
            chart_dept.update_layout(margin=dict(l=0, r=0, t=15, b=0), xaxis_tickangle=-25)
            st.plotly_chart(chart_dept, use_container_width=True)
        else:
            empty_state("No department breakdown available for this budget.")

    # 4. Actionable Sprint Patch Plan Table
    st.subheader(f"📋 Actionable Sprint Patch Plan · {len(selected)} Fixes Selected")
    if selected.empty:
        empty_state("No actions fit within this budget allocation.")
        return

    display = selected.copy()
    display["Risk reduction"] = display["risk_score"].round(4)
    display["Effort"] = display["remediation_hours"].apply(lambda h: f"{h:.1f}h")
    display["CVSS"] = display["cvss_score"].round(1)

    table_cols = [
        col for col in (
            "cve_id", "asset_name", "department_name", "severity", "CVSS",
            "Effort", "Risk reduction", "status",
        ) if col in display.columns
    ]
    st.dataframe(display[table_cols], use_container_width=True, hide_index=True)

    csv_data = display[table_cols].to_csv(index=False)
    st.download_button(
        label="📥 Export sprint remediation plan (CSV)",
        data=csv_data,
        file_name=f"cyberguard_sprint_plan_{budget}h.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # 5. Interactive 3-Layer Auditable Justification
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.subheader("🔍 Auditable 3-Layer Decision Justification")
    st.caption("Inspect why any specific recommendation was selected by the optimizer:")

    options = selected.index.tolist()
    selected_idx = st.selectbox(
        "Choose a fix from the current sprint plan:",
        options,
        format_func=lambda idx: f"{selected.loc[idx, 'cve_id']} · {selected.loc[idx, 'asset_name']} ({selected.loc[idx, 'department_name']}) · Risk {selected.loc[idx, 'risk_score']:.3f}",
        key="overview_cve_select",
    )
    row = selected.loc[selected_idx]

    preds = load_calibrated_predictions()
    prob_val = None
    if not preds.empty and "cve_id" in preds.columns:
        match_p = preds.loc[preds["cve_id"] == row["cve_id"], "model_probability"]
        if not match_p.empty:
            prob_val = match_p.iloc[0]

    c_threat, c_biz, c_solver = st.columns(3)
    with c_threat:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;">1 · THREAT INTELLIGENCE</div>
                <div style="font-size: 0.82rem; line-height: 1.6;">
                    • <b>CVE:</b> {row['cve_id']}<br>
                    • <b>EPSS Score:</b> {format_probability(row.get('epss_score'))}<br>
                    • <b>CISA KEV Weaponized:</b> {'Yes (Active In Wild)' if row.get('kev') == 1 else 'No'}<br>
                    • <b>Calibrated Model Likelihood:</b> {format_probability(prob_val) if prob_val is not None else 'Available in test stream'}<br>
                    • <b>CVSS Severity:</b> {row.get('cvss_score', 'N/A')} ({row.get('severity', 'Unknown')})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_biz:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;">2 · BUSINESS CONTEXT</div>
                <div style="font-size: 0.82rem; line-height: 1.6;">
                    • <b>Asset:</b> {row.get('asset_name', 'N/A')}<br>
                    • <b>Department:</b> {row.get('department_name', 'N/A')}<br>
                    • <b>Criticality:</b> {row.get('criticality', 'N/A')}<br>
                    • <b>Business Impact Score:</b> {row.get('business_impact_score', 'N/A')} / 10<br>
                    • <b>Exposure:</b> {'Internet-Facing (1.5×)' if row.get('internet_exposure') else 'Internal Network (1.0×)'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_solver:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight: 700; color: #38bdf8; margin-bottom: 0.5rem;">3 · OPTIMIZATION RATIONALE</div>
                <div style="font-size: 0.82rem; line-height: 1.6;">
                    • <b>Remediation Cost:</b> {row.get('remediation_hours', 0):.1f} engineer-hours<br>
                    • <b>Risk Addressed:</b> {row.get('risk_score', 0):.4f}<br>
                    • <b>Risk Density:</b> {(row.get('risk_score', 0) / max(0.1, row.get('remediation_hours', 1))):.4f} risk/hour<br>
                    • <b>Why Chosen:</b> Selected by exact 0-1 Knapsack branch-and-bound for maximal risk-reduction density under the hard {budget}h constraint.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 6. Enterprise Reference Footnote
    with st.expander("Enterprise Dataset & Baseline Topology Reference"):
        refresh = latest_data_refresh()
        refresh_text = refresh.strftime("%Y-%m-%d %H:%M UTC") if refresh is not None else "Unknown"
        st.caption(
            f"Evaluated across the ShopEasy enterprise archetype (30 assets across 7 business units). "
            f"Total enterprise modeled risk surface: {actions['risk_score'].sum():,.2f}. "
            f"Latest local artifact update: {refresh_text}. Operates with 0 live API latency."
        )
