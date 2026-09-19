# ML Initial Audit

## 1. Repository inventory

### Data Files
- data\processed\asset_vulnerabilities.csv
- data\processed\cyberguard_dataset.csv
- data\processed\cyberguard_master_enterprise_dataset.csv
- data\processed\data_quality_report.txt
- data\processed\enterprise_assets.csv
- data\processed\enterprise_risk_summary.csv
- data\processed\missing_records_report.csv
- data\processed\risk_features.csv
- data\raw\epss_data.csv
- data\raw\kev_data.csv
- data\raw\nvd_data.csv

### Python Scripts
- src\data_collection\epss_collector.py
- src\data_collection\kev_collector.py
- src\data_collection\nvd_collector.py
- src\data_processing\data_quality.py
- src\data_processing\data_quality_report.py
- src\data_processing\feature_engineering.py
- src\data_processing\merge_data.py
- src\data_processing\missing_values_report.py
- src\database\__init__.py
- src\database\config.py
- src\database\connection.py
- src\database\models.py
- src\database\seed_database.py
- src\enterprise\__init__.py
- src\enterprise\assets_config.py
- src\enterprise\data_access.py
- src\enterprise\export_datasets.py
- src\enterprise\synthetic_generator.py
- src\ml\data_audit.py

### Notebooks
- None detected

### Requirements Files
- requirements.txt
- requirements.txt

### Ml Related Files
- src\data_processing\feature_engineering.py
- src\ml\data_audit.py

## 2. Dataset profile

- Shape: 6316 rows x 13 columns
- Columns: CVE_ID, CVSS_Score, CVSS_Version, Severity, Description, Published_Date, EPSS_Score, EPSS_Percentile, KEV, Vulnerability_Age_Days, Severity_Encoded, CVSS_EPSS_Interaction, KEV_EPSS_Interaction

### Data types

CVE_ID                     object
CVSS_Score                float64
CVSS_Version              float64
Severity                   object
Description                object
Published_Date             object
EPSS_Score                float64
EPSS_Percentile           float64
KEV                         int64
Vulnerability_Age_Days      int64
Severity_Encoded          float64
CVSS_EPSS_Interaction     float64
KEV_EPSS_Interaction      float64

### Missing values

CVSS_Score                300
Severity                  300
CVSS_Version              300
EPSS_Percentile           300
EPSS_Score                300
KEV_EPSS_Interaction      300
Severity_Encoded          300
CVSS_EPSS_Interaction     300
CVE_ID                      0
Description                 0
Published_Date              0
KEV                         0
Vulnerability_Age_Days      0

### Duplicate records

- Duplicate rows: 0

### Unique categorical values

- CVE_ID: unique=6316; examples=['CVE-2021-45929', 'CVE-2021-45944', 'CVE-2021-45945', 'CVE-2021-45946', 'CVE-2021-45947', 'CVE-2021-45948', 'CVE-2021-45949', 'CVE-2021-45950', 'CVE-2021-45951', 'CVE-2021-45952']
- Severity: unique=4; examples=['MEDIUM', 'CRITICAL', 'HIGH', 'LOW']
- Description: unique=5699; examples=['Wasm3 0.5.0 has an out-of-bounds write in CompileBlock (called from CompileElseBlock and Compile_If).', 'Ghostscript GhostPDL 9.50 through 9.53.3 has a use-after-free in sampled_data_sample (called from sampled_data_continue and interp).', 'Rejected reason: DO NOT USE THIS CANDIDATE NUMBER. ConsultIDs: none. Reason: This candidate was withdrawn by its CNA. Further investigation showed that it was not a security issue. Notes: none', 'Wasm3 0.5.0 has an out-of-bounds write in CompileBlock (called from Compile_LoopOrBlock and CompileBlockStatements).', 'Wasm3 0.5.0 has an out-of-bounds write in Runtime_Release (called from EvaluateExpression and InitDataSegments).', 'Open Asset Import Library (aka assimp) 5.1.0 and 5.1.1 has a heap-based buffer overflow in _m3d_safestr (called from m3d_load and Assimp::M3DWrapper::M3DWrapper).', 'Ghostscript GhostPDL 9.50 through 9.54.0 has a heap-based buffer overflow in sampled_data_finish (called from sampled_data_continue and interp).', 'LibreDWG 0.12.4.4313 through 0.12.4.4367 has an out-of-bounds write in dwg_free_BLOCK_private (called from dwg_free_BLOCK and dwg_free_object).', 'Dnsmasq 2.86 has a heap-based buffer overflow in check_bad_address (called from check_for_bogus_wildcard and FuzzCheckForBogusWildcard). NOTE: the vendor\'s position is that CVE-2021-45951 through CVE-2021-45957 "do not represent real vulnerabilities, to the best of our knowledge.', 'Dnsmasq 2.86 has a heap-based buffer overflow in dhcp_reply (called from dhcp_packet and FuzzDhcp). NOTE: the vendor\'s position is that CVE-2021-45951 through CVE-2021-45957 "do not represent real vulnerabilities, to the best of our knowledge.']
- Published_Date: unique=6316; examples=['2022-01-01 00:15:08.057000+00:00', '2022-01-01 00:15:08.183000+00:00', '2022-01-01 00:15:08.230000+00:00', '2022-01-01 00:15:08.277000+00:00', '2022-01-01 00:15:08.320000+00:00', '2022-01-01 00:15:08.367000+00:00', '2022-01-01 00:15:08.413000+00:00', '2022-01-01 00:15:08.460000+00:00', '2022-01-01 00:15:08.507000+00:00', '2022-01-01 00:15:08.553000+00:00']

### Numerical statistics

                         count         mean        std          min          25%          50%          75%         max
CVSS_Score              6016.0     7.059707   1.723647     1.900000     5.500000     7.400000     8.100000    10.00000
CVSS_Version            6016.0     3.100432   0.023560     3.000000     3.100000     3.100000     3.100000     4.00000
EPSS_Score              6016.0     0.031570   0.101907     0.000820     0.006330     0.010370     0.018160     0.99796
EPSS_Percentile         6016.0     0.597545   0.233463     0.002420     0.478790     0.614930     0.771525     0.99956
KEV                     6316.0     0.007283   0.085036     0.000000     0.000000     0.000000     0.000000     1.00000
Vulnerability_Age_Days  6316.0  1657.661495  25.613382  1612.000000  1634.000000  1660.000000  1682.000000  1702.00000
Severity_Encoded        6016.0     2.667553   0.764454     1.000000     2.000000     3.000000     3.000000     4.00000
CVSS_EPSS_Interaction   6016.0     0.255309   0.889423     0.003003     0.036335     0.069472     0.143570     9.93510
KEV_EPSS_Interaction    6016.0     0.003282   0.050248     0.000000     0.000000     0.000000     0.000000     0.99796

## 3. Existing feature-engineering logic

The current feature-engineering logic lives in `src/data_processing/feature_engineering.py`.

Observed transformations:
- `Published_Date` is converted to timezone-aware datetime (`utc=True`).
- `Vulnerability_Age_Days` = reference_date - `Published_Date` in days.
- `Severity_Encoded` maps `LOW -> 1`, `MEDIUM -> 2`, `HIGH -> 3`, `CRITICAL -> 4`.
- `CVSS_EPSS_Interaction` = `CVSS_Score * EPSS_Score`.
- `KEV_EPSS_Interaction` = `KEV * EPSS_Score`.

Important note: the 300 incomplete records are preserved rather than zero-filled or deleted.

## 4. Potential target and label candidates

### Future KEV appearance
- Meaning: Whether a CVE becomes a CISA KEV item in a future time window.
- Data needed: Historical CVEs, publication date, exploitation telemetry, and a future KEV label window.
- Leakage risk: High if the target is derived from the same time window as the features or if future exploit knowledge is used in training.
- Backtesting: Yes, with time-based splits and a forward-looking label definition.
- Fit: Strong conceptual fit for real-world prioritization if the label is future-looking and non-leaking.

### Binary classification: current KEV flag
- Meaning: A vulnerability is already in KEV (1) or not (0).
- Data needed: The current dataset already contains KEV as a binary label.
- Leakage risk: Moderate to high if used as a target with the same snapshot features, because it is strongly correlated with existing input features and is not a future-risk estimate.
- Backtesting: Limited, because it is a contemporaneous label rather than a forward-looking risk target.
- Fit: Useful for a benchmark but not ideal as the final enterprise prioritization target.

### Risk regression
- Meaning: Predict a numeric risk score from technical and business factors.
- Data needed: Target labels like remediation urgency, exploit likelihood, or a business risk score built from historical outcomes.
- Leakage risk: High if the numeric target is constructed from the same fields used as model inputs or if business risk is defined from the same period.
- Backtesting: Possible with a carefully defined historical target.
- Fit: A valid formulation only when the target is clearly defined and temporally aligned.

### Ranking formulation
- Meaning: Order vulnerabilities by expected importance rather than predict a single integer label.
- Data needed: A ranked list from historical outcomes or labeled expert reviews.
- Leakage risk: Moderate if ranking is derived from current signals; requires careful temporal splits.
- Backtesting: Yes, via rank-based retrospective evaluation.
- Fit: Good for prioritization, but not a direct supervised target unless a valid historical ranking is defined.

## 5. Potential data leakage risks

- The dataset is vulnerability-level and does not yet include enterprise asset context; it is not yet the final per-asset ML dataset.
- `CVSS_Score`, `EPSS_Score`, `Severity`, `KEV`, and interaction features are all derived from the same vulnerability snapshot and should not be treated as a future target when used on the same row.
- Using `Description` or `Published_Date` in the same model as a target label is not appropriate without careful temporal validation, because text and publication time may encode current exposure patterns.
- For future exploitation modeling, any target should be built from a later time window than the feature window to avoid look-ahead bias.
- The `Severity_Encoded` field is a deterministic encoding of `Severity`, so it should not be used as a target if `Severity` also appears in input features.
- If the future enterprise asset dataset is added later, the per-asset mapping must remain one vulnerability per asset row to avoid collapsing multiple assets into a single record.

## 6. Initial takeaways

- The current dataset is a vulnerability-centric feature set, not the final enterprise-level row-level ML dataset.
- Missingness is concentrated in `CVSS_Score`, `CVSS_Version`, `Severity`, `EPSS_Score`, `EPSS_Percentile`, `Severity_Encoded`, and interaction terms, exactly matching the 300 incomplete records described by the project documentation.
- The CVSS/EPSS missingness pattern is not random but is consistent with incomplete vulnerability records that lack score metadata.
- There are no duplicate CVEs, which is a good sign for record integrity.
- `KEV` is binary and sparse (46 of 6316 values), which makes it useful as a signal but not a reliable target without temporal logic.
- The key decision is to define a future-looking target before any LightGBM training is attempted.

## Recommended next step

- Define and document the approved target using a temporal split and a clearly justified label.
- Confirm the precise input feature set for the approved model after the enterprise context is available.
- Treat the 300 incomplete records as a missing-data problem, not as a zero-fill problem.
- Do not train a final LightGBM model until this target definition is reviewed and approved.