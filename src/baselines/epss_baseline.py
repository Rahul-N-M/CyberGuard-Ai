from __future__ import annotations

import pandas as pd


def epss_only_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline prioritization based solely on EPSS score."""
    output = df[["CVE_ID", "EPSS_Score"]].copy()
    output["Baseline"] = "EPSS-only"
    output["Risk_Score"] = output["EPSS_Score"].fillna(0.0)
    output = output.sort_values(["Risk_Score", "CVE_ID"], ascending=[False, True]).reset_index(drop=True)
    output["Rank"] = range(1, len(output) + 1)
    return output[["CVE_ID", "Baseline", "Risk_Score", "Rank"]]
