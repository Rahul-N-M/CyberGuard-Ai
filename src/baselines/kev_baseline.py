from __future__ import annotations

import pandas as pd


def kev_first_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline prioritization that prioritizes KEV vulnerabilities first.

    This baseline is intentionally simple and rule-based rather than statistical.
    """
    output = df[["CVE_ID", "KEV", "CVSS_Score", "EPSS_Score"]].copy()
    output["Baseline"] = "KEV-first"
    output["Risk_Score"] = (
        output["KEV"].fillna(0).astype(float) * 1000
        + output["CVSS_Score"].fillna(0).astype(float) * 10
        + output["EPSS_Score"].fillna(0).astype(float) * 100
    )
    output = output.sort_values(["KEV", "Risk_Score", "CVE_ID"], ascending=[False, False, True]).reset_index(drop=True)
    output["Rank"] = range(1, len(output) + 1)
    return output[["CVE_ID", "Baseline", "Risk_Score", "Rank"]]
