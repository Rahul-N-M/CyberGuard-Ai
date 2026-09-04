from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def evaluate_ranking_quality(predictions: pd.DataFrame, actual: pd.Series | None = None) -> dict[str, float]:
    """Compute simple ranking-quality summary for a scored dataframe.

    This is a reusable placeholder for the later ML evaluation stage.
    """
    if "Risk_Score" not in predictions.columns:
        raise ValueError("Expected a 'Risk_Score' column for ranking evaluation.")

    ranked = predictions.sort_values("Risk_Score", ascending=False).reset_index(drop=True)
    summary = {
        "n_rows": int(len(ranked)),
        "top_1_score": float(ranked["Risk_Score"].iloc[0]) if len(ranked) else 0.0,
        "top_10_mean": float(ranked["Risk_Score"].head(10).mean()) if len(ranked) else 0.0,
    }
    if actual is not None:
        summary["actual_mean"] = float(actual.mean())
    return summary


def summarize_model_comparison(model_predictions: pd.DataFrame, baseline_predictions: pd.DataFrame) -> pd.DataFrame:
    """Create a compact comparison table between model and baseline outputs."""
    model = model_predictions.copy()
    baseline = baseline_predictions.copy()

    model["Model_Type"] = "LightGBM"
    baseline["Model_Type"] = baseline.get("Baseline", "Baseline")

    comparison = pd.concat([model, baseline], ignore_index=True)
    return comparison


def write_model_evaluation_report(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = """# Model Evaluation Plan

This document is a placeholder for the evaluation stage of the CyberGuard AI model.

The final model should be assessed using metrics appropriate to the chosen target.

Potential evaluation components:
- ranking quality
- high-risk recall
- calibration (if classification)
- false positives and false negatives
- feature importance review
- missing-value robustness

The selected target is future KEV appearance within a future time window.
"""
    output.write_text(content, encoding="utf-8")
    return str(output)
