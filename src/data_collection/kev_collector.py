import requests
import pandas as pd

url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

response = requests.get(url)

print("Status:", response.status_code)

data = response.json()

records = []

for item in data["vulnerabilities"]:
    records.append({
        "CVE_ID": item["cveID"],
        "KEV": 1,
        "Vendor": item["vendorProject"],
        "Product": item["product"],
        "Date_Added": item["dateAdded"]
    })

df = pd.DataFrame(records)

df.to_csv("data/raw/kev_data.csv", index=False)

print("KEV vulnerabilities:", len(df))
print("Saved to: data/raw/kev_data.csv")