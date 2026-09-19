# Baseline Prioritization Methods

These are baseline scoring methods used only for comparison with the eventual ML model. They are not machine learning models.

## 1. CVSS-only

- Rank vulnerability priority strictly by `CVSS_Score`.
- Interpretation: severity-only ordering.
- Strength: easy to explain and simple to audit.
- Weakness: ignores exploitation likelihood and enterprise context.

## 2. EPSS-only

- Rank vulnerability priority strictly by `EPSS_Score`.
- Interpretation: probability of exploitation-only ordering.
- Strength: captures current exploitation likelihood.
- Weakness: may miss severe vulnerabilities that are not currently exploited.

## 3. KEV-first

- Prioritize vulnerabilities in CISA KEV first.
- Use CVSS and EPSS as tie-breakers within the KEV group.
- Interpretation: confirmed exploited vulnerabilities are highest urgency.
- Strength: operationally meaningful and highly relevant for real-world remediation.
- Weakness: sparse coverage; many non-KEV vulnerabilities will be pushed down.

## 4. Weighted baseline

- Use a simple combination of normalized CVSS, EPSS, and KEV presence.
- Formula used in the current implementation:

  Risk_Score = 0.5 * (CVSS / 10) + 0.3 * EPSS + 0.2 * KEV

- Strength: balances technical severity, exploitation probability, and confirmed exploitation.
- Weakness: still a heuristic and not learned from historical outcomes.

## Output format

All baseline methods return a consistent structure:

- CVE_ID
- Baseline
- Risk_Score
- Rank

This makes them easy to compare directly against future ML outputs.
