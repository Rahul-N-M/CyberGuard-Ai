import pandas as pd

# Load the CyberGuard master dataset
df = pd.read_csv("data/processed/cyberguard_dataset.csv")

print("\n==============================")
print("CYBERGUARD DATA QUALITY REPORT")
print("==============================")

# Basic information
print("\nTotal records:", len(df))
print("Total columns:", len(df.columns))

# Duplicate CVEs
duplicates = df["CVE_ID"].duplicated().sum()
print("Duplicate CVEs:", duplicates)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# CVSS statistics
print("\nCVSS Statistics:")
print(df["CVSS_Score"].describe())

# EPSS statistics
print("\nEPSS Statistics:")
print(df["EPSS_Score"].describe())

# KEV distribution
print("\nKEV Distribution:")
print(df["KEV"].value_counts())

# Severity distribution
print("\nSeverity Distribution:")
print(df["Severity"].value_counts(dropna=False))

print("\n==============================")
print("DATA QUALITY CHECK COMPLETE")
print("==============================")