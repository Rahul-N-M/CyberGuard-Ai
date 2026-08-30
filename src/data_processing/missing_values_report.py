import pandas as pd

INPUT_FILE = "data/processed/cyberguard_dataset.csv"
OUTPUT_FILE = "data/processed/missing_records_report.csv"

df = pd.read_csv(INPUT_FILE)

# Fields required for the current risk/ML features
required_fields = [
    "CVSS_Score",
    "CVSS_Version",
    "Severity",
    "EPSS_Score",
    "EPSS_Percentile"
]

# Find records missing any required field
missing_records = df[df[required_fields].isnull().any(axis=1)].copy()

print("==============================")
print("CYBERGUARD MISSING DATA REPORT")
print("==============================")

print(f"Total records: {len(df)}")
print(f"Records with missing core fields: {len(missing_records)}")

print("\nMissing values by field:")
print(df[required_fields].isnull().sum())

print("\nMissing CVE IDs:")
print(missing_records["CVE_ID"].tolist())

# Save detailed report
missing_records.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")