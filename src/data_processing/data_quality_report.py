import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/cyberguard_dataset.csv")

# Basic statistics
total_records = len(df)
total_columns = len(df.columns)
duplicates = df["CVE_ID"].duplicated().sum()

# Missing values
missing = df.isnull().sum()

# CVSS statistics
cvss = df["CVSS_Score"].describe()

# EPSS statistics
epss = df["EPSS_Score"].describe()

# KEV distribution
kev = df["KEV"].value_counts()

# Severity distribution
severity = df["Severity"].value_counts(dropna=False)

# Create report
with open("data/processed/data_quality_report.txt", "w") as file:

    file.write("CYBERGUARD AI - DATA QUALITY REPORT\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Total records: {total_records}\n")
    file.write(f"Total columns: {total_columns}\n")
    file.write(f"Duplicate CVEs: {duplicates}\n\n")

    file.write("MISSING VALUES\n")
    file.write("-" * 20 + "\n")
    file.write(missing.to_string())
    file.write("\n\n")

    file.write("CVSS STATISTICS\n")
    file.write("-" * 20 + "\n")
    file.write(cvss.to_string())
    file.write("\n\n")

    file.write("EPSS STATISTICS\n")
    file.write("-" * 20 + "\n")
    file.write(epss.to_string())
    file.write("\n\n")

    file.write("KEV DISTRIBUTION\n")
    file.write("-" * 20 + "\n")
    file.write(kev.to_string())
    file.write("\n\n")

    file.write("SEVERITY DISTRIBUTION\n")
    file.write("-" * 20 + "\n")
    file.write(severity.to_string())
    file.write("\n")

print("Data quality report created.")
print("Saved to: data/processed/data_quality_report.txt")