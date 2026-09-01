import requests
import pandas as pd
import time
from datetime import datetime

URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

OUTPUT_FILE = "data/raw/nvd_data.csv"
BACKUP_FILE = "data/raw/nvd_data_expanded_backup.csv"

RESULTS_PER_PAGE = 2000
MAX_RETRIES = 3


def get_month_ranges(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    ranges = []

    for date in dates:
        month_start = date.strftime("%Y-%m-%dT00:00:00.000")

        next_month = date + pd.offsets.MonthBegin(1)
        month_end = (
            next_month - pd.Timedelta(milliseconds=1)
        ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        ranges.append((month_start, month_end))

    return ranges


def request_with_retry(params):

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = requests.get(
                URL,
                params=params,
                timeout=60
            )

            print(
                f"    Attempt {attempt}: "
                f"HTTP {response.status_code}"
            )

            if response.status_code == 200:
                return response

            print(response.text[:500])

        except requests.RequestException as e:
            print("    Request error:", e)

        if attempt < MAX_RETRIES:
            wait = attempt * 5
            print(f"    Waiting {wait} seconds...")
            time.sleep(wait)

    return None


# ==========================================
# MAIN COLLECTION
# ==========================================

all_records = []

month_ranges = get_month_ranges(
    START_DATE,
    END_DATE
)

print("==========================================")
print("NVD EXPANDED COLLECTION")
print("==========================================")
print("Period:", START_DATE, "to", END_DATE)
print("Months:", len(month_ranges))


for month_number, (start_date, end_date) in enumerate(
    month_ranges, start=1
):

    print()
    print("------------------------------------------")
    print(
        f"Month {month_number}/{len(month_ranges)}"
    )
    print(start_date, "to", end_date)
    print("------------------------------------------")

    start_index = 0
    month_records = []
    total_results = None

    while True:

        params = {
            "pubStartDate": start_date,
            "pubEndDate": end_date,
            "startIndex": start_index,
            "resultsPerPage": RESULTS_PER_PAGE
        }

        response = request_with_retry(params)

        if response is None:
            print("FAILED MONTH - stopping collection.")
            print("Existing nvd_data.csv will NOT be changed.")
            raise SystemExit(1)

        data = response.json()

        vulnerabilities = data.get(
            "vulnerabilities",
            []
        )

        total_results = data.get(
            "totalResults",
            0
        )

        print(
            "    Page:",
            start_index,
            "| Received:",
            len(vulnerabilities),
            "| Total:",
            total_results
        )

        for item in vulnerabilities:

            cve = item["cve"]
            metrics = cve.get("metrics", {})

            cvss_score = None
            cvss_version = None
            severity = None

            if "cvssMetricV40" in metrics:
                cvss = metrics["cvssMetricV40"][0]["cvssData"]
                cvss_score = cvss["baseScore"]
                cvss_version = "4.0"
                severity = cvss.get("baseSeverity")

            elif "cvssMetricV31" in metrics:
                cvss = metrics["cvssMetricV31"][0]["cvssData"]
                cvss_score = cvss["baseScore"]
                cvss_version = "3.1"
                severity = cvss.get("baseSeverity")

            elif "cvssMetricV30" in metrics:
                cvss = metrics["cvssMetricV30"][0]["cvssData"]
                cvss_score = cvss["baseScore"]
                cvss_version = "3.0"
                severity = cvss.get("baseSeverity")

            elif "cvssMetricV2" in metrics:
                cvss = metrics["cvssMetricV2"][0]["cvssData"]
                cvss_score = cvss["baseScore"]
                cvss_version = "2.0"
                severity = cvss.get("baseSeverity")

            descriptions = cve.get(
                "descriptions",
                []
            )

            description = ""

            for d in descriptions:
                if d.get("lang") == "en":
                    description = d.get("value", "")
                    break

            month_records.append({
                "CVE_ID": cve["id"],
                "CVSS_Score": cvss_score,
                "CVSS_Version": cvss_version,
                "Severity": severity,
                "Description": description,
                "Published_Date": cve["published"]
            })

        start_index += len(vulnerabilities)

        if start_index >= total_results:
            break

        time.sleep(1)

    print(
        f"Month completed: {len(month_records)} CVEs"
    )

    all_records.extend(month_records)

    # Small delay between months
    time.sleep(2)


# ==========================================
# BUILD DATASET
# ==========================================

if not all_records:

    print("No CVEs collected.")
    print("Existing dataset was NOT changed.")
    raise SystemExit(1)


df = pd.DataFrame(all_records)

# Remove duplicates
df = df.drop_duplicates(
    subset=["CVE_ID"]
)

# Sort by publication date
df["Published_Date"] = pd.to_datetime(
    df["Published_Date"],
    errors="coerce"
)

df = df.sort_values(
    "Published_Date"
)

# Convert date back to ISO string
df["Published_Date"] = df[
    "Published_Date"
].astype(str)


# ==========================================
# SAVE BACKUP OF NEW DATASET
# ==========================================

df.to_csv(
    BACKUP_FILE,
    index=False
)

# Only after successful creation,
# replace the main dataset.
df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("==========================================")
print("NVD COLLECTION COMPLETE")
print("==========================================")
print("Total unique CVEs:", len(df))
print("First publication:", df["Published_Date"].min())
print("Last publication:", df["Published_Date"].max())
print("Saved to:", OUTPUT_FILE)
print("Backup:", BACKUP_FILE)