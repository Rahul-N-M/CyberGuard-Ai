import pandas as pd

# Load datasets
nvd = pd.read_csv("data/raw/nvd_data.csv")
epss = pd.read_csv("data/raw/epss_data.csv")
kev = pd.read_csv("data/raw/kev_data.csv")

print("NVD records:", len(nvd))
print("EPSS records:", len(epss))
print("KEV records:", len(kev))

# Merge NVD with EPSS
merged = nvd.merge(
    epss,
    on="CVE_ID",
    how="left"
)

# Add KEV information
merged = merged.merge(
    kev[["CVE_ID", "KEV"]],
    on="CVE_ID",
    how="left"
)

# CVEs not present in KEV are not known exploited
merged["KEV"] = merged["KEV"].fillna(0).astype(int)

# Save combined dataset
merged.to_csv(
    "data/processed/cyberguard_dataset.csv",
    index=False
)

print("\nFinal records:", len(merged))
print("EPSS matches:", merged["EPSS_Score"].notna().sum())
print("KEV vulnerabilities:", merged["KEV"].sum())
print("\nSaved to: data/processed/cyberguard_dataset.csv")