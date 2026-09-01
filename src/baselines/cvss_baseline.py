from __future__ import annotations

import pandas as pd


def cvss_only_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline prioritization based solely on CVSS score."""
    output = df[["CVE_ID", "CVSS_Score"]].copy()
    output["Baseline"] = "CVSS-only"
    output["Risk_Score"] = output["CVSS_Score"].fillna(0.0)
    output = output.sort_values(["Risk_Score", "CVE_ID"], ascending=[False, True]).reset_index(drop=True)
    output["Rank"] = range(1, len(output) + 1)
    return output[["CVE_ID", "Baseline", "Risk_Score", "Rank"]]
