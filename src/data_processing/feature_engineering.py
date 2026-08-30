import pandas as pd

# Load the merged CyberGuard dataset
input_file = "data/processed/cyberguard_dataset.csv"
output_file = "data/processed/risk_features.csv"

df = pd.read_csv(input_file)

print("Original records:", len(df))

# Convert published date to datetime
df["Published_Date"] = pd.to_datetime(
    df["Published_Date"],
    errors="coerce",
    utc=True
)

# Calculate vulnerability age
reference_date = pd.Timestamp.now(tz="UTC")

df["Vulnerability_Age_Days"] = (
    reference_date - df["Published_Date"]
).dt.days

# Encode severity
severity_mapping = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4
}

df["Severity_Encoded"] = df["Severity"].map(severity_mapping)

# Interaction between CVSS and EPSS
df["CVSS_EPSS_Interaction"] = (
    df["CVSS_Score"] * df["EPSS_Score"]
)

# Interaction between KEV and EPSS
df["KEV_EPSS_Interaction"] = (
    df["KEV"] * df["EPSS_Score"]
)

# Save feature-engineered dataset
df.to_csv(output_file, index=False)

print("Feature engineering completed.")
print("Final records:", len(df))
print("Features added:")
print("- Vulnerability_Age_Days")
print("- Severity_Encoded")
print("- CVSS_EPSS_Interaction")
print("- KEV_EPSS_Interaction")
print("Saved to:", output_file)