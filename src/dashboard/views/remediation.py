import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, metric_row, page_header
from src.dashboard.risk import optimize_budget


BUDGETS = {5: "5 hours · Emergency", 10: "10 hours · DevOps", 20: "20 hours · Sprint", 40: "40 hours · Engineering week"}


def render() -> None:
    page_header("Remediation optimizer", "Exact 0–1 knapsack selection using the project's risk objective and remediation-hour constraints.")
    budget = st.select_slider("Sprint budget", options=list(BUDGETS), value=10, format_func=lambda value: BUDGETS[value])
    with st.spinner("Solving the remediation allocation against the current enterprise mapping…"):
        result = optimize_budget(float(budget))

    improvement = "N/A" if result["improvement_pct"] is None else f"{result['improvement_pct']:+.1f}%"
    metric_row([
        ("Modeled risk addressed", f"{result['exact_risk']:,.3f}", "Exact solver objective"),
        ("Hours used", f"{result['used_hours']:.2f}h", f"Budget: {budget}h"),
        ("Hours remaining", f"{result['remaining_hours']:.2f}h", "Constraint enforced on selected actions"),
        ("vs. naive CVSS greedy", improvement, f"Context only · baseline risk {result['baseline_risk']:,.3f}"),
    ])
    st.caption(f"Solver used: `{result['solver']}`. 'Risk addressed' is the modeled composite score, not measured real-world loss.")

    selected = result["selected"]
    if selected.empty:
        empty_state("No remediation action fits this budget.", "The optimizer returned a valid empty allocation; no action was forced into the plan.")
        return
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
    st.subheader(f"Recommended fixes · {len(selected)} actions")
    st.dataframe(display[columns], use_container_width=True, hide_index=True)
    st.info(
        "The exact optimizer maximizes the sum of modeled risk scores under the hard hour budget. "
        "The percentage is a contextual difference, not an optimality gap: the CVSS-only baseline ignores EPSS, "
        "KEV, business impact, and exposure, so its selection objective differs."
    )
