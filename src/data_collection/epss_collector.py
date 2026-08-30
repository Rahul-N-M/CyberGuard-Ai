import requests
import pandas as pd
import time

# Load NVD CVEs
nvd = pd.read_csv("data/raw/nvd_data.csv")

cve_ids = nvd["CVE_ID"].dropna().unique().tolist()

url = "https://api.first.org/data/v1/epss"

records = []

# Process CVEs in batches
batch_size = 100

for i in range(0, len(cve_ids), batch_size):

    batch = cve_ids[i:i + batch_size]

    params = {
        "cve": ",".join(batch)
    }

    response = requests.get(url, params=params)

    print(
        f"Batch {i // batch_size + 1} "
        f"({i + 1}-{min(i + batch_size, len(cve_ids))}) "
        f"Status: {response.status_code}"
    )

    if response.status_code != 200:
        print("EPSS API request failed.")
        print(response.text)
        continue

    data = response.json()

    for item in data.get("data", []):

        records.append({
            "CVE_ID": item["cve"],
            "EPSS_Score": float(item["epss"]),
            "EPSS_Percentile": float(item["percentile"])
        })

    time.sleep(0.2)

df = pd.DataFrame(records)

df.to_csv("data/raw/epss_data.csv", index=False)

print("\n==============================")
print("EPSS COLLECTION COMPLETE")
print("==============================")
print("NVD CVEs:", len(cve_ids))
print("EPSS matches:", len(df))
print("Saved to: data/raw/epss_data.csv")