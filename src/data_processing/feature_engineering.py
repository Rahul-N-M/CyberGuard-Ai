import pandas as pd

# ==========================================
# FILES
# ==========================================

INPUT_FILE = "data/processed/cyberguard_dataset.csv"
OUTPUT_FILE = "data/processed/risk_features.csv"
KEV_FILE = "data/raw/kev_data.csv"

# ==========================================
# LOAD MERGED DATASET
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("==========================================")
print("CYBERGUARD TEMPORAL FEATURE ENGINEERING")
print("==========================================")
print("Original CVEs:", len(df))

# ==========================================
# CONVERT NVD PUBLICATION DATE
# ==========================================

df["Published_Date"] = pd.to_datetime(
    df["Published_Date"],
    errors="coerce",
    utc=True
)

# ==========================================
# LOAD KEV DATE
# ==========================================
#
# KEV information is used ONLY to construct
# the future target.
#
# It must NOT be used as a model feature.
# ==========================================

kev = pd.read_csv(KEV_FILE)

kev["Date_Added"] = pd.to_datetime(
    kev["Date_Added"],
    errors="coerce",
    utc=True
)

# Keep only the columns needed
kev_dates = kev[
    ["CVE_ID", "Date_Added"]
].drop_duplicates(
    subset=["CVE_ID"]
)

# Merge KEV dates
df = df.merge(
    kev_dates,
    on="CVE_ID",
    how="left"
)

# ==========================================
# REMOVE INVALID PUBLICATION DATES
# ==========================================

before = len(df)

df = df.dropna(
    subset=["Published_Date"]
).copy()

print(
    "Removed CVEs with missing publication date:",
    before - len(df)
)

# ==========================================
# DETERMINE OBSERVATION PERIOD
# ==========================================

min_date = df["Published_Date"].min()
max_date = df["Published_Date"].max()

# Convert to monthly timestamps.
#
# .to_period("M") removes timezone information,
# so we restore UTC using tz_localize().
# ==========================================

first_month = (
    min_date
    .to_period("M")
    .to_timestamp()
    .tz_localize("UTC")
)

last_month = (
    max_date
    .to_period("M")
    .to_timestamp()
    .tz_localize("UTC")
)

# ==========================================
# CREATE MONTHLY OBSERVATION DATES
# ==========================================

observation_dates = pd.date_range(
    start=first_month,
    end=last_month,
    freq="MS",
    tz="UTC"
)

print(
    "Observation months:",
    len(observation_dates)
)

# ==========================================
# CREATE TEMPORAL OBSERVATIONS
# ==========================================

frames = []

for observation_date in observation_dates:

    # Only vulnerabilities that were already
    # published at the observation date can
    # be considered.
    eligible = df[
        df["Published_Date"] <= observation_date
    ].copy()

    if eligible.empty:
        continue

    eligible["Observation_Date"] = observation_date

    # ======================================
    # VULNERABILITY AGE AT OBSERVATION TIME
    # ======================================

    eligible["Vulnerability_Age_Days"] = (
        observation_date
        - eligible["Published_Date"]
    ).dt.days

    frames.append(eligible)

# Combine all monthly observations
temporal = pd.concat(
    frames,
    ignore_index=True
)

print(
    "Temporal observations:",
    len(temporal)
)

# ==========================================
# CREATE 180-DAY FUTURE KEV TARGET
# ==========================================
#
# Target = 1 if the vulnerability enters KEV
# AFTER the observation date and within the
# following 180 days.
#
# Target = 0 otherwise.
#
# Date_Added is NOT used as a feature.
# ==========================================

temporal["Target_KEV_180d"] = (
    temporal["Date_Added"].notna()
    &
    (
        temporal["Date_Added"]
        > temporal["Observation_Date"]
    )
    &
    (
        temporal["Date_Added"]
        <=
        (
            temporal["Observation_Date"]
            + pd.Timedelta(days=180)
        )
    )
).astype(int)

# ==========================================
# ENCODE SEVERITY
# ==========================================

severity_mapping = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4
}

temporal["Severity_Encoded"] = (
    temporal["Severity"]
    .map(severity_mapping)
)

# ==========================================
# IMPORTANT:
# DO NOT CREATE KEV-BASED FEATURES
# ==========================================
#
# We intentionally DO NOT create:
#
# KEV_EPSS_Interaction
#
# because KEV is the outcome used to create
# Target_KEV_180d.
#
# Using KEV as a feature would cause leakage.
# ==========================================

# ==========================================
# REMOVE OBSERVATIONS WITHOUT FULL
# 180-DAY FOLLOW-UP
# ==========================================
#
# To determine whether a vulnerability becomes
# KEV within 180 days, we need at least 180
# days of subsequent KEV history.
#
# Example:
#
# Observation = 2024-12-01
# Need KEV history through = 2025-05-30
#
# If our KEV data ends earlier, we cannot safely
# label that observation as negative.
# ==========================================

latest_kev_date = temporal["Date_Added"].max()

if pd.notna(latest_kev_date):

    cutoff_date = (
        latest_kev_date
        - pd.Timedelta(days=180)
    )

    before_cutoff = len(temporal)

    temporal = temporal[
        temporal["Observation_Date"]
        <= cutoff_date
    ].copy()

    print(
        "Removed observations without full "
        "180-day follow-up:",
        before_cutoff - len(temporal)
    )

# ==========================================
# SELECT FINAL FEATURES
# ==========================================
#
# Date_Added is deliberately excluded.
# KEV is deliberately excluded.
#
# EPSS is also excluded from the temporal
# prediction features because the current
# EPSS dataset is not historical.
# ==========================================

feature_columns = [
    "CVE_ID",
    "Observation_Date",
    "CVSS_Score",
    "CVSS_Version",
    "Severity",
    "Severity_Encoded",
    "Description",
    "Published_Date",
    "Vulnerability_Age_Days",
    "Target_KEV_180d"
]

# Make sure only existing columns are selected
feature_columns = [
    column
    for column in feature_columns
    if column in temporal.columns
]

final_df = temporal[
    feature_columns
].copy()

# ==========================================
# SORT DATA
# ==========================================

final_df = final_df.sort_values(
    [
        "Observation_Date",
        "CVE_ID"
    ]
).reset_index(drop=True)

# ==========================================
# SAVE
# ==========================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==========================================
# FINAL SUMMARY
# ==========================================

print()
print("==========================================")
print("TEMPORAL FEATURE ENGINEERING COMPLETE")
print("==========================================")

print(
    "Final observations:",
    len(final_df)
)

print(
    "Unique CVEs:",
    final_df["CVE_ID"].nunique()
)

print(
    "Positive observations:",
    final_df["Target_KEV_180d"].sum()
)

print(
    "Positive unique CVEs:",
    final_df.loc[
        final_df["Target_KEV_180d"] == 1,
        "CVE_ID"
    ].nunique()
)

print()
print("Target distribution:")

print(
    final_df["Target_KEV_180d"]
    .value_counts()
)

print()
print("Features created:")

print("- Observation_Date")
print("- Vulnerability_Age_Days")
print("- Severity_Encoded")
print("- Target_KEV_180d")

print()
print("KEV and Date_Added are NOT model features.")
print("Current EPSS is NOT used as a temporal feature.")

print()
print("Saved to:")
print(OUTPUT_FILE)