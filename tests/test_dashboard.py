import numpy as np
import pandas as pd

from src.dashboard.data import load_assets, load_threat_intelligence
from src.optimization.knapsack_solver import compute_risk_scores, solve_knapsack


def test_dashboard_loads_existing_enterprise_and_threat_data():
    assets = load_assets()
    threats = load_threat_intelligence()

    assert len(assets) == 30
    assert assets["department_name"].nunique() == 7
    assert {"cve_id", "cvss_score", "epss_score", "kev", "published_date"}.issubset(threats.columns)
    assert threats["cve_id"].is_unique


def test_dashboard_risk_adapter_uses_existing_business_risk_formula():
    rows = pd.DataFrame({
        "cvss_score": [10.0, 5.0],
        "epss_score": [0.2, 0.2],
        "kev": [0, 1],
        "business_impact_score": [10, 10],
        "internet_exposure": [1, 0],
        "remediation_hours": [1.0, 1.0],
        "status": ["Open", "Open"],
    })

    result = compute_risk_scores(rows)

    np.testing.assert_allclose(result["risk_score"], [3.0, 3.0])


def test_exact_solver_never_exceeds_budget():
    selected, value, hours, _ = solve_knapsack(
        np.array([4.0, 5.0, 7.0]),
        np.array([3.0, 4.0, 5.0]),
        7.0,
    )

    assert set(selected) == {0, 1}
    assert value == 9.0
    assert hours <= 7.0
