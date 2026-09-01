import requests
import pandas as pd
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================

NVD_FILE = "data/raw/nvd_data.csv"
OUTPUT_FILE = "data/raw/epss_data.csv"
TEMP_FILE = "data/raw/epss_data_temp.csv"

API_URL = "https://api.first.org/data/v1/epss"

BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_WAIT = 5
REQUEST_DELAY = 0.5

# ==========================================
# LOAD NVD CVEs
# ==========================================

nvd = pd.read_csv(NVD_FILE)

cve_ids = (
    nvd["CVE_ID"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

print("==========================================")
print("EPSS COLLECTION")
print("==========================================")
print("NVD CVEs:", len(cve_ids))
print("Batch size:", BATCH_SIZE)

# ==========================================
# COLLECT EPSS
# ==========================================

records = []

total_batches = (
    len(cve_ids) + BATCH_SIZE - 1
) // BATCH_SIZE

for batch_number, i in enumerate(
    range(0, len(cve_ids), BATCH_SIZE),
    start=1
):

    batch = cve_ids[i:i + BATCH_SIZE]

    params = {
        "cve": ",".join(batch)
    }

    success = False

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                API_URL,
                params=params,
                timeout=60
            )

            print(
                f"Batch {batch_number}/{total_batches} "
                f"| Attempt {attempt} "
                f"| Status: {response.status_code}"
            )

            if response.status_code == 200:

                data = response.json()

                batch_data = data.get(
                    "data",
                    []
                )

                for item in batch_data:

                    records.append({
                        "CVE_ID": item["cve"],
                        "EPSS_Score": float(item["epss"]),
                        "EPSS_Percentile": float(
                            item["percentile"]
                        )
                    })

                success = True

                print(
                    f"    EPSS records received: "
                    f"{len(batch_data)}"
                )

                break

            else:

                print(
                    "    API error:",
                    response.text[:300]
                )

        except requests.RequestException as e:

            print("    Request error:", e)

        if attempt < MAX_RETRIES:

            print(
                f"    Retrying in {RETRY_WAIT} seconds..."
            )

            time.sleep(RETRY_WAIT)

    if not success:

        print()
        print("==========================================")
        print("EPSS COLLECTION FAILED")
        print("==========================================")
        print(
            f"Failed batch: "
            f"{batch_number}/{total_batches}"
        )
        print(
            "Existing epss_data.csv "
            "was NOT changed."
        )

        raise SystemExit(1)

    time.sleep(REQUEST_DELAY)

# ==========================================
# BUILD DATAFRAME
# ==========================================

df = pd.DataFrame(records)

if df.empty:

    print("No EPSS records collected.")
    print("Existing dataset was NOT changed.")
    raise SystemExit(1)

# Remove duplicate CVEs
df = df.drop_duplicates(
    subset=["CVE_ID"]
)

# Sort
df = df.sort_values(
    "CVE_ID"
).reset_index(drop=True)

# ==========================================
# SAVE TEMPORARY FILE FIRST
# ==========================================

df.to_csv(
    TEMP_FILE,
    index=False
)

# Verify temporary file exists
if not os.path.exists(TEMP_FILE):

    print("Temporary EPSS file was not created.")
    print("Existing dataset was NOT changed.")
    raise SystemExit(1)

# ==========================================
# REPLACE MAIN FILE
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==========================================
# SUMMARY
# ==========================================

print()
print("==========================================")
print("EPSS COLLECTION COMPLETE")
print("==========================================")

print("NVD CVEs:", len(cve_ids))
print("EPSS matches:", len(df))
print(
    "EPSS coverage:",
    f"{(len(df) / len(cve_ids) * 100):.2f}%"
)

print("Saved to:", OUTPUT_FILE)
print("Temporary copy:", TEMP_FILE)