from __future__ import annotations

import pandas as pd


def weighted_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Simple weighted score combining CVSS, EPSS, and KEV evidence."""
    output = df[["CVE_ID", "CVSS_Score", "EPSS_Score", "KEV"]].copy()
    output["Baseline"] = "Weighted"

    cvss_component = output["CVSS_Score"].fillna(0.0) / 10.0
    epss_component = output["EPSS_Score"].fillna(0.0)
    kev_component = output["KEV"].fillna(0).astype(float)

    output["Risk_Score"] = 0.5 * cvss_component + 0.3 * epss_component + 0.2 * kev_component
    output = output.sort_values(["Risk_Score", "CVE_ID"], ascending=[False, True]).reset_index(drop=True)
    output["Rank"] = range(1, len(output) + 1)
    return output[["CVE_ID", "Baseline", "Risk_Score", "Rank"]]
