"""
src/optimization/knapsack_solver.py
0-1 Knapsack Vulnerability Remediation Optimizer for Enterprise CyberGuard-AI.

Each optimization item is an ASSET-CVE REMEDIATION ACTION (one row of the
enterprise dataset), not a unique CVE -- the same CVE can appear multiple
times if it affects multiple assets, and each occurrence is an independently
schedulable remediation action with its own risk_score and remediation_hours.

Optimizes remediation actions across enterprise assets under engineer time
budgets:
  - 5 hours (emergency rapid-response sprint)
  - 10 hours (daily DevOps remediation quota)
  - 20 hours (half-time weekly sprint)
  - 40 hours (full engineer-week allocation)

Mathematical Formulation:
  Maximize:   Sum(risk_score_i * x_i)
  Subject to: Sum(remediation_hours_i * x_i) <= Budget
              x_i in {0, 1}

Composite risk score (kept unchanged from the original implementation):
  risk_score = exploit_prob * business_impact_score * exposure_mult * (cvss/10)
  exploit_prob  = clip(epss_score + (KEV_BONUS if kev==1 else 0), 0.001, 1.0)
  exposure_mult = INTERNET_EXPOSURE_MULTIPLIER if internet_exposure else INTERNAL_EXPOSURE_MULTIPLIER

IMPORTANT -- these are documented MODELING ASSUMPTIONS, not empirically
calibrated constants:
  - KEV_BONUS (0.40): an additive bump to exploit probability when a CVE is
    in CISA's Known Exploited Vulnerabilities catalog. Chosen as a
    directionally-reasonable adjustment, not fit to observed exploitation
    data.
  - INTERNET_EXPOSURE_MULTIPLIER (1.5x): a multiplicative penalty for
    internet-facing assets. Also a modeling choice, not empirically derived.
Both should be treated as tunable knobs for sensitivity analysis, not as
validated risk-modeling parameters. This module does not change or attempt
to justify them further.

"Risk reduced" / "risk addressed" in this module always means the MODELED
composite risk score removed by remediating the selected actions -- NOT a
measured or predicted real-world reduction in breach probability or
financial loss. All outputs are phrased as "modeled risk exposure addressed"
for this reason.

Solvers Supported (both EXACT, not heuristic):
  1. Google OR-Tools KnapsackSolver (branch-and-bound, native int64 solver)
  2. SciPy MILP HiGHS branch-and-cut solver (operates on continuous doubles
     directly -- used as primary/fallback and as a cross-check)

Baselines (for comparison only, not used by the exact optimizer):
  1. Naive CVSS-only greedy baseline    -- sorts by cvss_score alone
  2. Risk-per-hour greedy baseline      -- sorts by risk_score / remediation_hours,
                                            i.e. the SAME objective as the exact
                                            solver, just filled greedily instead
                                            of optimally
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Modeling assumptions (see module docstring)
# ---------------------------------------------------------------------------
KEV_BONUS = 0.40
INTERNET_EXPOSURE_MULTIPLIER = 1.5
INTERNAL_EXPOSURE_MULTIPLIER = 1.0

# Fixed-point scale used only for the OR-Tools integer solver. Continuous
# risk_score / remediation_hours are multiplied by this factor and rounded
# to integers, since KnapsackSolver requires int64 inputs.
FIXEDPOINT_SCALE = 10_000

ACTIONABLE_STATUSES = {"open", "in_progress", "in progress"}


class KnapsackInputError(ValueError):
    """Raised when knapsack optimizer inputs fail validation."""


# ---------------------------------------------------------------------------
# Step 1: Composite risk score computation (formula unchanged) + validation
# ---------------------------------------------------------------------------

def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the composite actionable risk score for every asset-CVE
    remediation action row and returns an augmented COPY of df with a new
    'risk_score' column.

    Performs input validation before computing:
      - required columns present
      - 'kev' is a clean binary 0/1 indicator (no NaN, no other values)
      - median-imputes missing cvss_score / epss_score (existing behavior),
        but raises if a column is entirely missing (imputation impossible)
    """
    required_cols = [
        "cvss_score", "epss_score", "kev", "business_impact_score",
        "internet_exposure", "remediation_hours", "status",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KnapsackInputError(f"Dataset missing required columns: {missing_cols}")

    df = df.copy()

    # --- kev must be a clean binary indicator ---
    kev_raw = df["kev"]
    if kev_raw.isna().any():
        n_bad = int(kev_raw.isna().sum())
        raise KnapsackInputError(
            f"'kev' column has {n_bad} missing values; expected binary 0/1 for every row."
        )
    kev_numeric = pd.to_numeric(kev_raw, errors="coerce")
    valid_binary = kev_numeric.isin([0, 1])
    if kev_numeric.isna().any() or not valid_binary.all():
        bad_vals = sorted(pd.unique(kev_raw[~valid_binary]).tolist())
        raise KnapsackInputError(
            f"'kev' column contains non-binary values: {bad_vals}. Expected only 0/1."
        )
    df["kev"] = kev_numeric.astype(int)

    # --- median imputation for missing cvss/epss (existing behavior) ---
    for col in ["cvss_score", "epss_score"]:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                raise KnapsackInputError(f"Cannot impute '{col}': entire column is NaN.")
            df[col] = df[col].fillna(median_val)
        if not np.isfinite(df[col].astype(float)).all():
            raise KnapsackInputError(f"'{col}' contains non-finite (inf) values.")

    if not np.isfinite(df["business_impact_score"].astype(float)).all() or df["business_impact_score"].isna().any():
        raise KnapsackInputError("'business_impact_score' contains NaN/infinite values.")

    exposure_mult = np.where(
        df["internet_exposure"].astype(bool),
        INTERNET_EXPOSURE_MULTIPLIER,
        INTERNAL_EXPOSURE_MULTIPLIER,
    )
    exploit_prob = np.clip(
        df["epss_score"] + np.where(df["kev"] == 1, KEV_BONUS, 0.0),
        0.001, 1.0,
    )

    df["risk_score"] = (
        exploit_prob
        * df["business_impact_score"]
        * exposure_mult
        * (df["cvss_score"] / 10.0)
    ).astype(float)

    return df


def get_actionable_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to actionable (open / in_progress) remediation actions and
    validates the fields the knapsack optimizer depends on directly.
    Requires df['risk_score'] to already exist (call compute_risk_scores() first).

    Raises KnapsackInputError (rather than silently dropping/coercing rows) on:
      - missing/NaN/infinite risk_score
      - negative risk_score
      - missing/NaN/infinite remediation_hours
      - non-positive remediation_hours
      - duplicate actionable asset_id + cve_id remediation actions
    """
    if "risk_score" not in df.columns:
        raise KnapsackInputError("call compute_risk_scores() before get_actionable_dataframe()")

    actionable_mask = df["status"].astype(str).str.lower().isin(ACTIONABLE_STATUSES)
    actionable_df = df[actionable_mask].copy().reset_index(drop=True)

    if len(actionable_df) == 0:
        raise KnapsackInputError("No actionable (open/in_progress) remediation actions found.")

    # --- duplicate asset-CVE remediation actions ---
    if {"asset_id", "cve_id"}.issubset(actionable_df.columns):
        dup_mask = actionable_df.duplicated(subset=["asset_id", "cve_id"], keep=False)
        if dup_mask.any():
            n_dupes = int(dup_mask.sum())
            example = (
                actionable_df.loc[dup_mask, ["asset_id", "cve_id"]]
                .drop_duplicates()
                .head(5)
            )
            raise KnapsackInputError(
                f"Found {n_dupes} duplicate actionable asset-CVE remediation action rows "
                f"(same asset_id + cve_id appearing more than once among actionable rows). "
                f"Each row is treated as one independent optimizable action, so duplicates "
                f"would double-count the same action. Examples:\n{example.to_string(index=False)}\n"
                f"Deduplicate upstream (e.g. keep the latest scan per asset-CVE pair) before "
                f"calling the optimizer -- this function will not silently collapse them for you."
            )

    # --- risk_score validation ---
    risk = actionable_df["risk_score"].astype(float)
    if risk.isna().any() or not np.isfinite(risk).all():
        n_bad = int((~np.isfinite(risk)).sum())
        raise KnapsackInputError(f"{n_bad} actionable rows have missing/NaN/infinite risk_score.")
    if (risk < 0).any():
        raise KnapsackInputError("Negative risk_score values found; risk_score must be >= 0.")

    # --- remediation_hours validation ---
    hours = actionable_df["remediation_hours"].astype(float)
    if hours.isna().any() or not np.isfinite(hours).all():
        n_bad = int((~np.isfinite(hours)).sum())
        raise KnapsackInputError(f"{n_bad} actionable rows have missing/NaN/infinite remediation_hours.")
    if (hours <= 0).any():
        n_bad = int((hours <= 0).sum())
        raise KnapsackInputError(
            f"{n_bad} actionable rows have non-positive remediation_hours "
            f"(every remediation action must take > 0 engineer hours)."
        )

    return actionable_df


def validate_knapsack_inputs(values: np.ndarray, weights: np.ndarray, capacity: float) -> None:
    """Low-level validation applied immediately before handing raw arrays to a solver."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) != len(weights):
        raise KnapsackInputError(f"values (len={len(values)}) and weights (len={len(weights)}) length mismatch.")
    if len(values) == 0:
        raise KnapsackInputError("No items to optimize over (empty values/weights array).")
    if not np.isfinite(values).all():
        raise KnapsackInputError("values array contains NaN/infinite entries.")
    if not np.isfinite(weights).all():
        raise KnapsackInputError("weights array contains NaN/infinite entries.")
    if (values < 0).any():
        raise KnapsackInputError("values array contains negative risk scores.")
    if (weights <= 0).any():
        raise KnapsackInputError("weights array contains non-positive remediation_hours.")
    if capacity is None or not np.isfinite(capacity):
        raise KnapsackInputError(f"Invalid budget/capacity: {capacity!r} (must be a finite number).")
    if capacity <= 0:
        raise KnapsackInputError(f"Invalid budget/capacity: {capacity!r} (must be > 0).")


# ---------------------------------------------------------------------------
# Step 2: Exact solvers
# ---------------------------------------------------------------------------

def solve_knapsack_milp(values: np.ndarray, weights: np.ndarray, capacity: float):
    """
    Solves 0-1 Knapsack to provable global optimality using HiGHS branch-and-cut
    via scipy.optimize.milp, operating directly on continuous doubles (no
    fixed-point scaling needed for this solver).
    """
    validate_knapsack_inputs(values, weights, capacity)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n = len(values)
    c = -values
    A = weights.reshape(1, -1)
    constraints = LinearConstraint(A, lb=0, ub=float(capacity))
    integrality = np.ones(n, dtype=np.int32)
    bounds = Bounds(0, 1)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"scipy MILP solver failed: status={res.status} message={res.message}")

    x = np.round(res.x).astype(int)
    if not np.all((x == 0) | (x == 1)):
        raise RuntimeError("scipy MILP returned non-binary solution values; refusing to trust result.")

    chosen_indices = np.where(x == 1)[0]
    total_weight = float(np.sum(weights[chosen_indices])) if len(chosen_indices) else 0.0
    if total_weight - capacity > 1e-6:
        raise RuntimeError(
            f"scipy MILP solution violates budget: used {total_weight} > capacity {capacity}"
        )
    total_value = float(np.sum(values[chosen_indices])) if len(chosen_indices) else 0.0
    return chosen_indices, total_value, total_weight, "scipy_milp_highs"


def solve_knapsack_ortools(values: np.ndarray, weights: np.ndarray, capacity: float):
    """
    Solves 0-1 Knapsack using Google OR-Tools KnapsackSolver (exact
    branch-and-bound). Returns None if OR-Tools is not installed.

    Fixed-point rounding strategy (this is the correctness fix over the
    previous implementation, not just a precision tweak):
      - risk_score VALUES are rounded to the NEAREST integer on the scaled
        grid. Rounding direction here only affects objective accuracy, not
        feasibility, so nearest-rounding minimizes error.
      - remediation_hours WEIGHTS are rounded UP (ceiling).
      - capacity is rounded DOWN (floor).
    Ceiling the weights and flooring the capacity guarantees that any
    selection the integer solver considers feasible in scaled space is also
    feasible w.r.t. the ORIGINAL continuous hour budget:
        sum(w_i) <= sum(ceil(w_i * scale)) / scale
                  <= floor(capacity * scale) / scale
                  <= capacity
    The previous implementation used int(w * scale) (truncation, i.e. floor)
    for BOTH weights and capacity. Flooring the weights can understate an
    item's true remediation_hours, which -- combined with a floored
    capacity -- could let the scaled solver accept a combination that
    actually exceeds the true hour budget once you sum the real
    (unrounded) remediation_hours values.

    A second, subtler issue applies to naive ceil/floor on `w * scale`
    directly: IEEE-754 float multiplication noise (e.g. 0.27 * 10000 can
    land on 2700.0000000000005 instead of exactly 2700.0) gets pushed up an
    extra whole unit by a bare ceil(). That +1-per-item noise accumulates
    across many items and can make the scaled problem artificially tighter
    than the true budget -- concretely, this was observed to make OR-Tools
    reject a combination whose TRUE hours sum was exactly equal to the
    budget, while scipy's MILP (which works in continuous space with its
    own feasibility tolerance) accepted it, causing the two exact solvers to
    disagree. We subtract/add a small epsilon before ceil/floor to absorb
    floating-point noise while remaining conservative (the epsilon is many
    orders of magnitude smaller than the actual precision of
    remediation_hours values).
    """
    validate_knapsack_inputs(values, weights, capacity)
    try:
        from ortools.algorithms.python import knapsack_solver
    except ImportError:
        return None

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    scale = FIXEDPOINT_SCALE
    fp_eps = 1e-6  # absorbs float64 multiplication noise, see docstring above

    int_values = [int(round(v * scale)) for v in values]
    int_weights = [[int(math.ceil(w * scale - fp_eps)) for w in weights]]
    int_capacity = [int(math.floor(capacity * scale + fp_eps))]

    if int_capacity[0] < 0:
        raise KnapsackInputError("Scaled capacity is negative after rounding.")

    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "EnterpriseRemediationSolver",
    )
    solver.init(int_values, int_weights, int_capacity)
    solver.solve()

    chosen_indices = np.array(
        [i for i in range(len(values)) if solver.best_solution_contains(i)], dtype=int
    )
    total_weight = float(np.sum(weights[chosen_indices])) if len(chosen_indices) else 0.0
    if total_weight - capacity > 1e-6:
        # Should not happen given the ceil/floor rounding scheme above, but
        # verify rather than trust silently.
        raise RuntimeError(
            f"OR-Tools solution violates true budget after de-scaling: "
            f"{total_weight} > {capacity}"
        )
    total_value = float(np.sum(values[chosen_indices])) if len(chosen_indices) else 0.0
    return chosen_indices, total_value, total_weight, "google_ortools"


def solve_knapsack(values: np.ndarray, weights: np.ndarray, capacity: float):
    """
    Tries Google OR-Tools first; falls back to exact SciPy HiGHS MILP if
    OR-Tools is unavailable or errors. Both are exact solvers; see
    compare_solvers() to explicitly cross-check agreement rather than just
    silently trusting whichever one ran.
    """
    try:
        ortools_res = solve_knapsack_ortools(values, weights, capacity)
    except Exception:
        ortools_res = None
    if ortools_res is not None:
        return ortools_res
    return solve_knapsack_milp(values, weights, capacity)


def compare_solvers(values: np.ndarray, weights: np.ndarray, capacity: float,
                     value_tol: float = 1e-3):
    """
    Runs BOTH exact solvers on the same input and checks agreement on the
    optimal objective value (total risk_score addressed) and hours used.

    Item *sets* chosen may legitimately differ when there are ties (multiple
    optimal solutions with equal objective) -- this is NOT treated as
    disagreement as long as objective values match within value_tol.
    """
    milp_idx, milp_val, milp_hrs, _ = solve_knapsack_milp(values, weights, capacity)
    ortools_res = solve_knapsack_ortools(values, weights, capacity)

    result = {
        "capacity": capacity,
        "scipy_milp": {"objective": milp_val, "hours_used": milp_hrs, "n_items": int(len(milp_idx))},
    }

    if ortools_res is None:
        result["ortools"] = None
        result["agree"] = None
        result["note"] = "OR-Tools not available in this environment; comparison skipped."
        return result

    ort_idx, ort_val, ort_hrs, _ = ortools_res
    result["ortools"] = {"objective": ort_val, "hours_used": ort_hrs, "n_items": int(len(ort_idx))}
    value_diff = abs(milp_val - ort_val)
    result["value_diff"] = value_diff
    result["agree"] = bool(value_diff <= value_tol)
    return result


# ---------------------------------------------------------------------------
# Step 3: Baselines
# ---------------------------------------------------------------------------

def solve_greedy_cvss(df: pd.DataFrame, capacity: float):
    """
    NAIVE CVSS-ONLY GREEDY BASELINE.
    Sorts by cvss_score descending and remediates until the hour budget is
    exhausted. Ignores exploit probability, business impact, and exposure
    entirely. Included as an "industry status quo" reference point -- it
    optimizes a DIFFERENT objective than the exact solver (raw CVSS, not the
    composite risk_score), so its risk_score total is reported only for
    context, not as an apples-to-apples optimality comparison.
    """
    sorted_df = df.sort_values(by="cvss_score", ascending=False).reset_index()
    chosen_idx, used_hours = [], 0.0
    for _, row in sorted_df.iterrows():
        hrs = row["remediation_hours"]
        if used_hours + hrs <= capacity:
            used_hours += hrs
            chosen_idx.append(row["index"])
    chosen_idx = np.array(chosen_idx, dtype=int)
    tot_risk = float(df.loc[chosen_idx, "risk_score"].sum()) if len(chosen_idx) else 0.0
    return chosen_idx, tot_risk, float(used_hours)


def solve_greedy_risk_per_hour(df: pd.DataFrame, capacity: float):
    """
    RISK-PER-HOUR GREEDY BASELINE.
    Uses the SAME composite risk_score as the exact optimizer, sorted by
    risk_score / remediation_hours descending, filling the budget greedily.
    This is the fair greedy comparison point for the exact 0-1 knapsack
    solution, since it shares the exact solver's objective definition
    (unlike the CVSS-only baseline above). It is the classic fractional-
    knapsack-inspired greedy heuristic for 0-1 knapsack and is NOT guaranteed
    to be optimal, only a reasonable non-exact reference.
    """
    tmp = df.copy()
    tmp["_density"] = tmp["risk_score"] / tmp["remediation_hours"]
    sorted_df = tmp.sort_values(by="_density", ascending=False).reset_index()
    chosen_idx, used_hours = [], 0.0
    for _, row in sorted_df.iterrows():
        hrs = row["remediation_hours"]
        if used_hours + hrs <= capacity:
            used_hours += hrs
            chosen_idx.append(row["index"])
    chosen_idx = np.array(chosen_idx, dtype=int)
    tot_risk = float(df.loc[chosen_idx, "risk_score"].sum()) if len(chosen_idx) else 0.0
    return chosen_idx, tot_risk, float(used_hours)


# ---------------------------------------------------------------------------
# Step 4: Monotonicity check (exact solver only)
# ---------------------------------------------------------------------------

def check_monotonicity(budgets, objectives, tol: float = 1e-6) -> bool:
    """
    Verifies the EXACT optimal objective is non-decreasing as budget
    increases -- relaxing a knapsack's capacity can never make its optimum
    worse. Raises AssertionError if violated.

    Only meaningful for the exact solver's results. Greedy heuristics are
    NOT guaranteed to be monotone in budget and should not be checked here.
    """
    budgets = np.asarray(budgets, dtype=float)
    objectives = np.asarray(objectives, dtype=float)
    order = np.argsort(budgets)
    sorted_budgets = budgets[order]
    sorted_obj = objectives[order]
    for i in range(1, len(sorted_obj)):
        if sorted_obj[i] < sorted_obj[i - 1] - tol:
            raise AssertionError(
                f"Monotonicity violated: budget {sorted_budgets[i]} -> objective "
                f"{sorted_obj[i]:.6f} is less than budget {sorted_budgets[i - 1]} -> "
                f"objective {sorted_obj[i - 1]:.6f}"
            )
    return True


# ---------------------------------------------------------------------------
# Step 5: End-to-end benchmark
# ---------------------------------------------------------------------------

def run_enterprise_optimization(
    data_path: str = "data/processed/cyberguard_master_enterprise_dataset.csv",
    budgets=(5.0, 10.0, 20.0, 40.0),
    output_dir: str = "outputs",
    report_dir: str = "reports",
):
    """
    End-to-end benchmark: loads the dataset, computes risk scores, validates
    inputs, runs the exact optimizer + both greedy baselines across the given
    budgets, cross-checks the two exact solvers, verifies monotonicity of the
    exact objective, and writes CSV/Markdown outputs.

    Returns a dict with the long-format results DataFrame, the per-budget
    allocation DataFrame, the solver-agreement checks, and the monotonicity
    check result, so callers (including the notebook) can assert on them
    directly rather than re-deriving anything by hand.
    """
    import os

    print("=" * 78)
    print("  CYBERGUARD-AI: ENTERPRISE REMEDIATION 0-1 KNAPSACK OPTIMIZER")
    print("=" * 78)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Master enterprise dataset not found at {data_path}")

    raw_df = pd.read_csv(data_path)
    print(f"[Loading] {len(raw_df):,d} rows from {data_path}")

    df = compute_risk_scores(raw_df)
    total_enterprise_risk = float(df["risk_score"].sum())
    print(f"  Total enterprise MODELED risk surface (all rows): {total_enterprise_risk:,.2f}")

    actionable_df = get_actionable_dataframe(df)
    total_actionable_risk = float(actionable_df["risk_score"].sum())
    print(f"  Actionable (open/in_progress) remediation actions: {len(actionable_df):,d}")
    print(f"  Actionable MODELED risk surface: {total_actionable_risk:,.2f}")

    values = actionable_df["risk_score"].to_numpy()
    weights = actionable_df["remediation_hours"].to_numpy()

    long_rows = []
    allocation_records = []
    exact_objectives = []
    solver_agreement_checks = []

    print("\n" + "-" * 78)
    for b in budgets:
        # --- exact optimizer ---
        exact_idx, exact_val, exact_hrs, solver_name = solve_knapsack(values, weights, b)
        exact_objectives.append(exact_val)

        agreement = compare_solvers(values, weights, b)
        solver_agreement_checks.append(agreement)

        # --- baselines ---
        rph_idx, rph_val, rph_hrs = solve_greedy_risk_per_hour(actionable_df, b)
        cvss_idx, cvss_val, cvss_hrs = solve_greedy_cvss(actionable_df, b)

        for method, idx, val, hrs in [
            ("Exact 0-1 Knapsack", exact_idx, exact_val, exact_hrs),
            ("Risk-per-hour greedy baseline", rph_idx, rph_val, rph_hrs),
            ("Naive CVSS-only greedy baseline", cvss_idx, cvss_val, cvss_hrs),
        ]:
            pct_of_actionable = (val / total_actionable_risk * 100) if total_actionable_risk > 0 else 0.0
            long_rows.append({
                "method": method,
                "budget_hours": b,
                "hours_used": round(hrs, 4),
                "n_remediation_actions": int(len(idx)),
                "modeled_risk_score_addressed": round(val, 4),
                "pct_of_actionable_modeled_risk_addressed": round(pct_of_actionable, 4),
            })

        for idx in exact_idx:
            row = actionable_df.loc[idx]
            allocation_records.append({
                "budget_hours": b,
                "asset_id": row.get("asset_id"),
                "cve_id": row.get("cve_id"),
                "asset_name": row.get("asset_name"),
                "criticality": row.get("criticality"),
                "internet_exposure": row.get("internet_exposure"),
                "cvss_score": row.get("cvss_score"),
                "remediation_hours": row.get("remediation_hours"),
                "modeled_risk_score": round(float(row.get("risk_score")), 4),
            })

        agree_str = "AGREE" if agreement.get("agree") else ("N/A" if agreement.get("agree") is None else "DISAGREE")
        print(
            f"  budget={b:>5.1f}h  exact({solver_name})={exact_val:>10.3f}  "
            f"risk/hr-greedy={rph_val:>10.3f}  cvss-greedy={cvss_val:>10.3f}  "
            f"solver-agreement={agree_str}"
        )
    print("-" * 78)

    # --- monotonicity check on exact solver only ---
    check_monotonicity(list(budgets), exact_objectives)
    print(f"[Check] Monotonicity of exact objective across budgets: PASSED")

    disagreements = [c for c in solver_agreement_checks if c.get("agree") is False]
    if disagreements:
        raise AssertionError(f"OR-Tools and scipy MILP disagreed on {len(disagreements)} budget(s): {disagreements}")
    print("[Check] OR-Tools vs scipy HiGHS MILP objective agreement: PASSED (within tolerance)")

    results_df = pd.DataFrame(long_rows)
    alloc_df = pd.DataFrame(allocation_records)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    results_csv = os.path.join(output_dir, "knapsack_remediation_summary.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n[Saved] {results_csv}")

    alloc_csv = os.path.join(output_dir, "knapsack_remediation_allocations.csv")
    alloc_df.to_csv(alloc_csv, index=False)
    print(f"[Saved] {alloc_csv} ({len(alloc_df):,d} exact-optimizer remediation actions across all budgets)")

    _write_report(
        report_dir, results_df, total_enterprise_risk, total_actionable_risk,
        len(actionable_df), solver_agreement_checks,
    )

    return {
        "results_long": results_df,
        "allocations": alloc_df,
        "solver_agreement_checks": solver_agreement_checks,
        "monotonicity_passed": True,
        "total_enterprise_modeled_risk": total_enterprise_risk,
        "total_actionable_modeled_risk": total_actionable_risk,
        "actionable_df": actionable_df,
    }


def _write_report(report_dir, results_df, total_enterprise_risk, total_actionable_risk,
                   n_actionable, solver_agreement_checks):
    import os

    agree_rows = []
    for chk in solver_agreement_checks:
        if chk.get("ortools") is None:
            agree_rows.append(f"| {chk['capacity']}h | scipy only ({chk['scipy_milp']['objective']:.3f}) | N/A |")
        else:
            agree_rows.append(
                f"| {chk['capacity']}h | {chk['scipy_milp']['objective']:.3f} | "
                f"{chk['ortools']['objective']:.3f} | {'✅ agree' if chk['agree'] else '❌ DISAGREE'} |"
            )

    pivot = results_df.pivot(index="budget_hours", columns="method",
                              values=["hours_used", "n_remediation_actions",
                                      "modeled_risk_score_addressed",
                                      "pct_of_actionable_modeled_risk_addressed"])

    report_content = f"""# Enterprise Remediation 0-1 Knapsack Optimization Report

## Scope

Each optimization item is an **asset-CVE remediation action** (one dataset
row), not a unique CVE in isolation -- the same CVE affecting multiple
assets produces multiple independent remediation actions.

## Modeling assumptions (not empirically calibrated)

- KEV bonus to exploit probability: **+{KEV_BONUS}** additive, when `kev == 1`.
- Internet exposure multiplier: **{INTERNET_EXPOSURE_MULTIPLIER}x** for internet-facing
  assets, **{INTERNAL_EXPOSURE_MULTIPLIER}x** otherwise.

These are modeling choices carried over unchanged from the original
implementation. They are **not derived from an empirical study** of
exploitation rates or breach impact, and all "risk" figures below should be
read as **modeled risk exposure addressed under this specific formula**, not
as a measured or predicted real-world risk reduction.

## Totals

- Total actionable (open/in_progress) remediation actions: **{n_actionable:,d}**
- Total actionable modeled risk surface: **{total_actionable_risk:,.2f}**
- Total enterprise modeled risk surface (incl. non-actionable rows): **{total_enterprise_risk:,.2f}**

## Results by budget and method

{results_df.to_markdown(index=False)}

## Solver agreement (OR-Tools vs scipy HiGHS MILP), exact solver only

| Budget | scipy MILP objective | OR-Tools objective | Agreement |
| --- | --- | --- | --- |
{chr(10).join(agree_rows)}

## Checks performed

1. Exact solution respects the budget constraint (enforced in-code; a
   `RuntimeError` is raised otherwise, not silently ignored).
2. Every selected decision variable is binary (0/1); non-binary MILP
   solutions are rejected rather than rounded and trusted.
3. Duplicate asset-CVE actionable rows are rejected before optimization,
   so no remediation action can be double-counted.
4. Objective equals the sum of `risk_score` over selected actions (computed
   directly from the selected index set, not tracked separately).
5. Exact objective is verified non-decreasing across budgets ({', '.join(str(b) for b in sorted(results_df['budget_hours'].unique()))}) --
   see monotonicity check in the run log.
6. OR-Tools and scipy HiGHS MILP objective values are cross-checked for
   agreement within tolerance at every budget -- see the agreement table above.

## Methodology

1. **Composite risk formulation** (unchanged):
   risk_score = exploit_prob * business_impact_score * exposure_mult * (cvss_score / 10)

2. **0-1 Knapsack formulation**:
   maximize sum(risk_score_i * x_i) subject to sum(remediation_hours_i * x_i) <= budget, x_i in {{0,1}}
   Solved to exact, provable global optimality (branch-and-cut / branch-and-bound),
   not approximated.

3. **Baselines**:
   - *Naive CVSS-only greedy baseline*: sorts by `cvss_score` alone; optimizes a
     **different objective** than the exact solver, included only as an
     "industry status quo" reference point.
   - *Risk-per-hour greedy baseline*: sorts by `risk_score / remediation_hours`,
     i.e. the **same objective** as the exact solver, filled greedily instead
     of optimally. This is the fair greedy comparison point for the exact
     optimizer's gain.

## Limitations

- The composite risk formula's constants (KEV bonus, exposure multiplier)
  are modeling assumptions, not fitted/validated parameters.
- "Risk addressed" assumes remediating an action fully eliminates its
  modeled risk contribution; it does not model partial mitigation, residual
  risk after patching, or dependencies between remediation actions.
- This module does not incorporate ML-predicted exploit probabilities; it
  uses `epss_score` and `kev` directly from the dataset as provided.
"""
    path = os.path.join(report_dir, "knapsack_optimization_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[Saved] {path}")


if __name__ == "__main__":
    run_enterprise_optimization()
