import pandas as pd
from pathlib import Path

# ==========================================
# FILES
# ==========================================

INPUT_FILE = "data/processed/risk_features.csv"
OUTPUT_DIR = Path("data/processed/ml")

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

df["Observation_Date"] = pd.to_datetime(
    df["Observation_Date"],
    errors="coerce",
    utc=True
)

print("==========================================")
print("CYBERGUARD ML DATASET PREPARATION")
print("==========================================")

print("Total observations:", len(df))

# ==========================================
# REMOVE INVALID OBSERVATIONS
# ==========================================

before = len(df)

df = df.dropna(
    subset=[
        "Observation_Date",
        "Target_KEV_180d"
    ]
).copy()

print(
    "Removed invalid observations:",
    before - len(df)
)

# ==========================================
# CHRONOLOGICAL SPLIT
# ==========================================
#
# 2022 -> TRAIN
# 2023 -> VALIDATION
# 2024 -> TEST
#
# No random shuffling.
# ==========================================

df["Observation_Year"] = (
    df["Observation_Date"].dt.year
)

train = df[
    df["Observation_Year"] == 2022
].copy()

validation = df[
    df["Observation_Year"] == 2023
].copy()

test = df[
    df["Observation_Year"] == 2024
].copy()

# ==========================================
# REMOVE HELPER COLUMN
# ==========================================

train.drop(
    columns=["Observation_Year"],
    inplace=True
)

validation.drop(
    columns=["Observation_Year"],
    inplace=True
)

test.drop(
    columns=["Observation_Year"],
    inplace=True
)

# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# SAVE
# ==========================================

train.to_csv(
    TRAIN_FILE,
    index=False
)

validation.to_csv(
    VALIDATION_FILE,
    index=False
)

test.to_csv(
    TEST_FILE,
    index=False
)

# ==========================================
# SUMMARY FUNCTION
# ==========================================

def print_summary(name, data):

    total = len(data)

    positives = int(
        data["Target_KEV_180d"].sum()
    )

    negatives = total - positives

    unique_cves = data["CVE_ID"].nunique()

    positive_cves = data.loc[
        data["Target_KEV_180d"] == 1,
        "CVE_ID"
    ].nunique()

    positive_rate = (
        positives / total * 100
        if total > 0
        else 0
    )

    print()
    print("------------------------------------------")
    print(name)
    print("------------------------------------------")

    print("Observations:", total)
    print("Unique CVEs:", unique_cves)
    print("Positive observations:", positives)
    print("Negative observations:", negatives)
    print("Positive unique CVEs:", positive_cves)
    print(
        "Positive rate:",
        f"{positive_rate:.4f}%"
    )


# ==========================================
# PRINT RESULTS
# ==========================================

print_summary(
    "TRAIN (2022)",
    train
)

print_summary(
    "VALIDATION (2023)",
    validation
)

print_summary(
    "TEST (2024)",
    test
)

# ==========================================
# CHECK CHRONOLOGICAL ORDER
# ==========================================

print()
print("------------------------------------------")
print("DATE RANGES")
print("------------------------------------------")

for name, data in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:

    print(
        name + ":",
        data["Observation_Date"].min(),
        "to",
        data["Observation_Date"].max()
    )

# ==========================================
# CHECK FEATURE COLUMNS
# ==========================================

print()
print("------------------------------------------")
print("MODEL FEATURES")
print("------------------------------------------")

model_features = [
    "CVSS_Score",
    "CVSS_Version",
    "Severity_Encoded",
    "Vulnerability_Age_Days"
]

for feature in model_features:

    if feature in df.columns:
        print("OK:", feature)

    else:
        print("MISSING:", feature)

# ==========================================
# LEAKAGE CHECK
# ==========================================

print()
print("------------------------------------------")
print("LEAKAGE CHECK")
print("------------------------------------------")

for forbidden in [
    "KEV",
    "Date_Added",
    "Target_KEV_180d"
]:

    if forbidden in model_features:
        print(
            "WARNING:",
            forbidden,
            "is incorrectly listed as a feature."
        )
    else:
        print(
            "OK:",
            forbidden,
            "is not a model feature."
        )

# ==========================================
# FINAL MESSAGE
# ==========================================

print()
print("==========================================")
print("ML DATASET PREPARATION COMPLETE")
print("==========================================")

print("Train:", TRAIN_FILE)
print("Validation:", VALIDATION_FILE)
print("Test:", TEST_FILE)