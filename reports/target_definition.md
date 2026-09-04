# Target Definition for CyberGuard AI

## Objective

The goal is to estimate which vulnerabilities are most important for a specific enterprise context, with the final model serving as an input to a constrained remediation optimizer. The target must therefore be operationally meaningful, time-valid, and resistant to leakage.

## Candidate targets reviewed

### Future exploitation / future KEV appearance
- Meaning: The vulnerability becomes a publicly confirmed exploited issue in a future time window.
- Data Needed: Historical CVE metadata plus a later KEV/observed exploit label across a time horizon.
- Leakage Risk: Low if a strict temporal split is used; high if future labels are merged into current features.
- Backtesting: Yes, using time-based validation and forward-looking labeling.
- Fit: Strongest fit for the project objective: prioritize vulnerabilities likely to matter soon in a real enterprise environment.

### Current KEV binary classification
- Meaning: The vulnerability is already known exploited at the snapshot date.
- Data Needed: Current KEV flag from CISA KEV, which already exists in the current dataset.
- Leakage Risk: Moderate to high because the label is a current fact and may correlate directly with the current feature set.
- Backtesting: Limited as a prospective operational target; the label is contemporaneous rather than predictive.
- Fit: Useful as a benchmark, but not the best final model target for future prioritization.

### Risk regression
- Meaning: Predict a continuous risk score from technical and business features.
- Data Needed: Historical remediation urgency, exploit likelihood, or a validated proxy score.
- Leakage Risk: High unless the target is built from a future outcome and not from the same row-level evidence.
- Backtesting: Possible but demanding, and requires a defensible scalar target.
- Fit: Potentially valid, but weaker than an explicit future exploit target when the objective is prioritization under a limited budget.

### Ranking formulation
- Meaning: Order vulnerabilities by importance rather than classify them directly.
- Data Needed: A historical ranking of exploited or business-critical vulnerabilities, often from expert review or an operational outcome.
- Leakage Risk: Moderate if produced using current signals; must be time-ordered and cleanly separated from training inputs.
- Backtesting: Yes, using rank-based metrics over a holdout period.
- Fit: Good for decision support, but it is not a direct label unless a valid historical ranking exists.

## Final selection

**Selected target:** Future KEV appearance within a future time window

**Why selected:** It is the cleanest operational proxy for which vulnerabilities are likely to become urgent in the near term and aligns with the project's prioritization objective.
**Data requirement:** A temporal dataset with publication dates, exploit signals, and a later KEV occurrence label. The current vulnerability snapshot alone is not enough to train this target without careful temporal assembly.
**Leakage control:** Use a time-based split: all features come from an earlier observation window, and the target is derived from a later period. Never use the same future KEV or exploit label in the feature set.
**Backtesting:** Yes, historical backtesting is feasible by defining an as-of date and predicting the probability of KEV appearance in the next 30/60/90/180 days.
**Project fit:** Strong. The project objective is to rank vulnerabilities for specific enterprise context under limited remediation resources, and future exploitation risk is a defensible operational target.
**Recommended label definition:** y = 1 if a CVE is added to CISA KEV within the next 180 days after the feature observation date; otherwise y = 0.

## Recommended methodology

1. Define an observation date for each vulnerability or asset-vulnerability row.
2. Build the feature set from information available strictly before that date.
3. Create the label using a later KEV observation window, such as 30, 60, 90, or 180 days.
4. Use a time-based train/validation/test split to avoid look-ahead leakage.
5. Evaluate with ranking and high-risk recall metrics appropriate for prioritization tasks.

## Explicit constraints

- Do not use current KEV as a feature in the same row if the target is future KEV appearance.
- Do not use same-timestamp vulnerability severity or exploit metadata to define the target in a way that collapses the problem into a self-fulfilling label.
- Do not train a final LightGBM model until the time-based target has been approved and documented.