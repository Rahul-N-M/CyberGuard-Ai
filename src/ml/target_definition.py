from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "risk_features.csv"
REPORT_PATH = ROOT / "reports" / "target_definition.md"


def load_risk_features(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summarize_candidate_targets() -> list[dict[str, str]]:
    return [
        {
            "candidate": "Future exploitation / future KEV appearance",
            "meaning": "The vulnerability becomes a publicly confirmed exploited issue in a future time window.",
            "data_needed": "Historical CVE metadata plus a later KEV/observed exploit label across a time horizon.",
            "leakage_risk": "Low if a strict temporal split is used; high if future labels are merged into current features.",
            "backtesting": "Yes, using time-based validation and forward-looking labeling.",
            "fit": "Strongest fit for the project objective: prioritize vulnerabilities likely to matter soon in a real enterprise environment.",
        },
        {
            "candidate": "Current KEV binary classification",
            "meaning": "The vulnerability is already known exploited at the snapshot date.",
            "data_needed": "Current KEV flag from CISA KEV, which already exists in the current dataset.",
            "leakage_risk": "Moderate to high because the label is a current fact and may correlate directly with the current feature set.",
            "backtesting": "Limited as a prospective operational target; the label is contemporaneous rather than predictive.",
            "fit": "Useful as a benchmark, but not the best final model target for future prioritization.",
        },
        {
            "candidate": "Risk regression",
            "meaning": "Predict a continuous risk score from technical and business features.",
            "data_needed": "Historical remediation urgency, exploit likelihood, or a validated proxy score.",
            "leakage_risk": "High unless the target is built from a future outcome and not from the same row-level evidence.",
            "backtesting": "Possible but demanding, and requires a defensible scalar target.",
            "fit": "Potentially valid, but weaker than an explicit future exploit target when the objective is prioritization under a limited budget.",
        },
        {
            "candidate": "Ranking formulation",
            "meaning": "Order vulnerabilities by importance rather than classify them directly.",
            "data_needed": "A historical ranking of exploited or business-critical vulnerabilities, often from expert review or an operational outcome.",
            "leakage_risk": "Moderate if produced using current signals; must be time-ordered and cleanly separated from training inputs.",
            "backtesting": "Yes, using rank-based metrics over a holdout period.",
            "fit": "Good for decision support, but it is not a direct label unless a valid historical ranking exists.",
        },
    ]


def choose_target() -> dict[str, Any]:
    return {
        "selected_target": "Future KEV appearance within a future time window",
        "why_selected": "It is the cleanest operational proxy for which vulnerabilities are likely to become urgent in the near term and aligns with the project's prioritization objective.",
        "data_requirement": "A temporal dataset with publication dates, exploit signals, and a later KEV occurrence label. The current vulnerability snapshot alone is not enough to train this target without careful temporal assembly.",
        "leakage_control": "Use a time-based split: all features come from an earlier observation window, and the target is derived from a later period. Never use the same future KEV or exploit label in the feature set.",
        "backtesting": "Yes, historical backtesting is feasible by defining an as-of date and predicting the probability of KEV appearance in the next 30/60/90/180 days.",
        "project_fit": "Strong. The project objective is to rank vulnerabilities for specific enterprise context under limited remediation resources, and future exploitation risk is a defensible operational target.",
        "recommended_label_definition": "y = 1 if a CVE is added to CISA KEV within the next 180 days after the feature observation date; otherwise y = 0.",
    }


def build_markdown_report() -> str:
    candidates = summarize_candidate_targets()
    selected = choose_target()

    sections = [
        "# Target Definition for CyberGuard AI",
        "",
        "## Objective",
        "",
        "The goal is to estimate which vulnerabilities are most important for a specific enterprise context, with the final model serving as an input to a constrained remediation optimizer. The target must therefore be operationally meaningful, time-valid, and resistant to leakage.",
        "",
        "## Candidate targets reviewed",
        "",
    ]

    for item in candidates:
        sections.append(f"### {item['candidate']}")
        for key in ["meaning", "data_needed", "leakage_risk", "backtesting", "fit"]:
            sections.append(f"- {key.replace('_', ' ').title()}: {item[key]}")
        sections.append("")

    sections.extend([
        "## Final selection",
        "",
        f"**Selected target:** {selected['selected_target']}",
        "",
        f"**Why selected:** {selected['why_selected']}",
        f"**Data requirement:** {selected['data_requirement']}",
        f"**Leakage control:** {selected['leakage_control']}",
        f"**Backtesting:** {selected['backtesting']}",
        f"**Project fit:** {selected['project_fit']}",
        f"**Recommended label definition:** {selected['recommended_label_definition']}",
        "",
        "## Recommended methodology",
        "",
        "1. Define an observation date for each vulnerability or asset-vulnerability row.",
        "2. Build the feature set from information available strictly before that date.",
        "3. Create the label using a later KEV observation window, such as 30, 60, 90, or 180 days.",
        "4. Use a time-based train/validation/test split to avoid look-ahead leakage.",
        "5. Evaluate with ranking and high-risk recall metrics appropriate for prioritization tasks.",
        "",
        "## Explicit constraints",
        "",
        "- Do not use current KEV as a feature in the same row if the target is future KEV appearance.",
        "- Do not use same-timestamp vulnerability severity or exploit metadata to define the target in a way that collapses the problem into a self-fulfilling label.",
        "- Do not train a final LightGBM model until the time-based target has been approved and documented.",
    ])

    return "\n".join(sections)


def write_report(path: str | Path = REPORT_PATH) -> str:
    report = build_markdown_report()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return str(output)


def main() -> None:
    df = load_risk_features()
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print("Candidate review complete. Selected target: Future KEV appearance within a future time window.")
    report_path = write_report()
    print(f"Target definition saved to: {report_path}")


if __name__ == "__main__":
    main()
