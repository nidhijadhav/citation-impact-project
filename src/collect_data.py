"""
collect_data.py
---------------
Fetches Computer Science papers (2018–2020) from the OpenAlex API
and saves raw results as a JSON file.
"""

import requests
import json
import time
import os
import urllib.parse
from datetime import datetime

# CONFIG =========================================================================

EMAIL = "your_email@northeastern.edu"
API_KEY = "your_api_key_here"          # get free key at openalex.org/settings/api

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "openalex_cs_papers.json")

BASE_URL = "https://api.openalex.org/works"

# OpenAlex field ID for Computer Science
# See: https://openalex.org/fields/17
CS_FIELD_ID = "17"

YEARS = [2018, 2019, 2020]
PER_PAGE = 100
MAX_PAGES = 18
SLEEP = 0.12


# HELPER FUNCTIONS ===============================================================

def build_params(year: int) -> dict:
    return {
        "filter": (
            f"topics.field.id:{CS_FIELD_ID},"
            f"publication_year:{year},"
            "type:article"
        ),
        "select": (
            "id,title,publication_year,cited_by_count,"
            "authorships,primary_location"
        ),
        "sort":     "cited_by_count:desc",
        "per_page": PER_PAGE,
        "api_key":  API_KEY,
        "mailto":   EMAIL,
    }


def fetch_page(params: dict, cursor: str) -> dict:
    query = urllib.parse.urlencode(params) + f"&cursor={urllib.parse.quote(cursor, safe='*')}"
    url   = f"{BASE_URL}?{query}"
    resp  = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  API error {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    data = resp.json()
    if cursor == "*":
        print(f"  API reports {data.get('meta', {}).get('count', '?')} total results")
    return data


# MAIN COLLECTION ================================================================

def collect_papers() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_papers = []

    for year in YEARS:
        print(f"\n=== Fetching {year} papers ===")
        params   = build_params(year)
        cursor   = "*"
        page_num = 0

        while page_num < MAX_PAGES:
            try:
                data = fetch_page(params, cursor)
            except requests.HTTPError as e:
                print(f"  HTTP error on page {page_num + 1}: {e}. Retrying in 5s...")
                time.sleep(5)
                continue
            except requests.RequestException as e:
                print(f"  Request failed: {e}. Skipping page.")
                break

            results = data.get("results", [])
            if not results:
                print(f"  No more results at page {page_num + 1}.")
                break

            all_papers.extend(results)
            page_num += 1
            print(f"  Page {page_num:02d} | +{len(results)} papers | total so far: {len(all_papers)}")

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            time.sleep(SLEEP)

    return all_papers


# ENTRY POINT ====================================================================

if __name__ == "__main__":
    print(f"Starting collection at {datetime.now().strftime('%H:%M:%S')}")
    papers = collect_papers()

    payload = {
        "collected_at": datetime.now().isoformat(),
        "total_papers": len(papers),
        "years": YEARS,
        "papers": papers,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nDone. {len(papers)} papers saved to {OUTPUT_FILE}")