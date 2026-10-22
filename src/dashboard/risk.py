"""Adapters around the repository's existing business-risk and optimizer logic."""

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.data import load_enterprise_actions
from src.optimization.knapsack_solver import (
    KEV_BONUS,
    compute_risk_scores,
    get_actionable_dataframe,
    solve_greedy_cvss,
    solve_knapsack,
)


@st.cache_data(show_spinner=False)
def load_risk_actions() -> pd.DataFrame:
    scored = compute_risk_scores(load_enterprise_actions())
    scored["optimizer_exploit_proxy"] = np.clip(
        pd.to_numeric(scored["epss_score"], errors="coerce").fillna(0)
        + np.where(pd.to_numeric(scored["kev"], errors="coerce").fillna(0).eq(1), KEV_BONUS, 0),
        0.001,
        1.0,
    )
    return scored


@st.cache_data(show_spinner=False)
def actionable_risk_actions() -> pd.DataFrame:
    return get_actionable_dataframe(load_risk_actions())


@st.cache_data(show_spinner=False)
def optimize_budget(budget_hours: float) -> dict:
    """Run the actual exact solver and the repository's naive CVSS baseline."""
    actions = actionable_risk_actions().reset_index(drop=True)
    values = actions["risk_score"].to_numpy(dtype=float)
    weights = actions["remediation_hours"].to_numpy(dtype=float)
    feasible_positions = np.flatnonzero(weights <= budget_hours)
    if len(feasible_positions):
        # Actions that individually exceed capacity cannot be part of any feasible knapsack solution.
        local_selected, exact_risk, used_hours, solver_name = solve_knapsack(
            values[feasible_positions], weights[feasible_positions], budget_hours
        )
        selected_positions = feasible_positions[local_selected]
    else:
        selected_positions = np.array([], dtype=int)
        exact_risk, used_hours, solver_name = 0.0, 0.0, "no_feasible_actions"
    baseline_idx, baseline_risk, baseline_hours = solve_greedy_cvss(actions, budget_hours)
    chosen = actions.iloc[selected_positions].copy()
    chosen["risk_per_hour"] = chosen["risk_score"] / chosen["remediation_hours"]
    return {
        "selected": chosen.sort_values("risk_score", ascending=False),
        "exact_risk": float(exact_risk),
        "used_hours": float(used_hours),
        "remaining_hours": max(0.0, float(budget_hours) - float(used_hours)),
        "solver": solver_name,
        "baseline_risk": float(baseline_risk),
        "baseline_hours": float(baseline_hours),
        "improvement_pct": (
            (float(exact_risk) - float(baseline_risk)) / float(baseline_risk) * 100
            if baseline_risk > 0 else None
        ),
    }
