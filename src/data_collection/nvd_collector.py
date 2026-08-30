import requests
import pandas as pd
import time

url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

start_date = "2022-01-01T00:00:00.000"
end_date = "2022-03-31T23:59:59.999"

all_records = []
start_index = 0
results_per_page = 2000

while True:

    params = {
        "pubStartDate": start_date,
        "pubEndDate": end_date,
        "startIndex": start_index,
        "resultsPerPage": results_per_page
    }

    response = requests.get(url, params=params)

    print("Request:", start_index, "Status:", response.status_code)

    if response.status_code != 200:
        print("API request failed.")
        print(response.text)

        # Do NOT overwrite the existing dataset
        break

    data = response.json()

    for item in data["vulnerabilities"]:

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

        description = cve["descriptions"][0]["value"]

        all_records.append({
            "CVE_ID": cve["id"],
            "CVSS_Score": cvss_score,
            "CVSS_Version": cvss_version,
            "Severity": severity,
            "Description": description,
            "Published_Date": cve["published"]
        })

    total = data["totalResults"]

    start_index += len(data["vulnerabilities"])

    print("Collected:", len(all_records), "/", total)

    if start_index >= total:
        break

    time.sleep(0.6)

# Only save if we actually collected data
if len(all_records) > 0:

    df = pd.DataFrame(all_records)

    df.to_csv("data/raw/nvd_data.csv", index=False)

    print("\n==============================")
    print("NVD COLLECTION COMPLETE")
    print("==============================")
    print("Total CVEs:", len(df))
    print("Saved to: data/raw/nvd_data.csv")

else:
    print("\nNo CVEs collected.")
    print("Existing nvd_data.csv was NOT changed.")