import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import empty_state, metric_row, page_header
from src.dashboard.risk import optimize_budget


BUDGETS = {5: "5 hours · Emergency", 10: "10 hours · DevOps", 20: "20 hours · Sprint", 40: "40 hours · Engineering week"}


def render() -> None:
    page_header("Remediation optimizer", "Exact 0–1 knapsack selection maximizing enterprise risk reduction under hard engineer-hour limits.")
    budget = st.select_slider("Sprint budget", options=list(BUDGETS), value=10, format_func=lambda value: BUDGETS[value])
    with st.spinner("Solving the remediation allocation against the current enterprise mapping…"):
        result = optimize_budget(float(budget))

    diff_risk = result["exact_risk"] - result["baseline_risk"]
    metric_row([
        ("Modeled risk addressed", f"{result['exact_risk']:,.3f}", "Exact solver objective"),
        ("Hours used", f"{result['used_hours']:.2f}h", f"Budget: {budget}h"),
        ("Hours remaining", f"{result['remaining_hours']:.2f}h", "Constraint enforced on selected actions"),
        ("Absolute Risk Gain vs CVSS", f"+{diff_risk:,.3f}", f"CyberGuard {result['exact_risk']:.2f} vs CVSS {result['baseline_risk']:.2f}"),
    ])

    st.caption(f"Solver used: `{result['solver']}`. 'Risk addressed' is the modeled composite score, not measured real-world loss.")

    selected = result["selected"]
    if selected.empty:
        empty_state("No remediation action fits this budget.", "The optimizer returned a valid empty allocation; no action was forced into the plan.")
        return

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Selected fixes by department")
        if "department_name" in selected.columns:
            dept_summary = selected.groupby("department_name").agg(
                fixes=("cve_id", "count"),
                hours=("remediation_hours", "sum"),
                risk_reduced=("risk_score", "sum"),
            ).reset_index()
            chart = px.bar(
                dept_summary,
                x="department_name",
                y="risk_reduced",
                color="hours",
                labels={"department_name": "Department", "risk_reduced": "Risk reduced", "hours": "Hours"},
                template="plotly_dark",
                color_continuous_scale="Viridis",
            )
            chart.update_layout(margin=dict(l=0, r=0, t=15, b=0), xaxis_tickangle=-25)
            st.plotly_chart(chart, use_container_width=True)
    with col_chart2:
        st.subheader("CyberGuard vs. CVSS greedy")
        comp_df = pd.DataFrame([
            {"Strategy": "CyberGuard Knapsack (Exact)", "Modeled Risk Reduced": result["exact_risk"], "Hours Spent": result["used_hours"]},
            {"Strategy": "CVSS Naive Greedy", "Modeled Risk Reduced": result["baseline_risk"], "Hours Spent": result["baseline_hours"]},
        ])
        comp_chart = px.bar(
            comp_df,
            x="Strategy",
            y="Modeled Risk Reduced",
            color="Strategy",
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8", "#f43f5e"],
        )
        comp_chart.update_layout(margin=dict(l=0, r=0, t=15, b=0), showlegend=False)
        st.plotly_chart(comp_chart, use_container_width=True)

    display = selected.copy()
    display["Risk reduction (modeled)"] = display["risk_score"]
    display["Why selected"] = display.apply(
        lambda row: (
            f"{row['risk_score']:.3f} modeled risk for {row['remediation_hours']:.2f}h "
            f"({row['risk_per_hour']:.3f} risk/h); fits the exact budget allocation."
        ),
        axis=1,
    )
    columns = [
        col for col in (
            "cve_id", "asset_name", "department_name", "severity", "cvss_score",
            "remediation_hours", "Risk reduction (modeled)", "Why selected",
        ) if col in display.columns
    ]
    headers = [
        col.replace("_", " ").title() if col != "Why selected" else col
        for col in columns
    ]

    st.subheader(f"Recommended fixes · {len(selected)} actions selected")
    st.dataframe(display[columns].rename(columns=dict(zip(columns, headers))), use_container_width=True, hide_index=True)

    csv_data = display[columns].to_csv(index=False)
    st.download_button(
        label="📥 Export sprint remediation plan (CSV)",
        data=csv_data,
        file_name=f"cyberguard_remediation_plan_{budget}h.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Multi-Budget Knapsack Benchmark Reference (ShopEasy Topology)"):
        st.markdown(
            """
            | Sprint Budget | Solver Engine | Items Fixed | Hours Used | Optimal Risk Reduced | CVSS Greedy Risk | Net Risk Gain (ΔRisk) |
            | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
            | **5 Hours** | Google OR-Tools / HiGHS | **3** | 4.50 h | **10.51** | 7.97 | **+2.54 risk units** |
            | **10 Hours** | Google OR-Tools / HiGHS | **6** | 9.50 h | **21.28** | 12.38 | **+8.90 risk units** |
            | **20 Hours** | Google OR-Tools / HiGHS | **13** | 19.50 h | **42.27** | 24.81 | **+17.46 risk units** |
            | **40 Hours** | Google OR-Tools / HiGHS | **26** | 39.50 h | **84.34** | 56.40 | **+27.94 risk units** |
            """
        )

