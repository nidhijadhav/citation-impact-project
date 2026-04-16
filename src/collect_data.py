"""
collect_data.py
---------------
Fetches Computer Science papers (2018–2020) from the OpenAlex API and saves the raw results as a JSON file.
"""

import requests
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# CONFIG =========================================================================

API_KEY = os.getenv("OPENALEX_API_KEY", "")
if not API_KEY:
    raise ValueError("OPENALEX_API_KEY not set. Add it to your .env file.")


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "openalex_cs_papers.json")

BASE_URL = "https://api.openalex.org/works"

# OpenAlex field ID for Computer Science
# See: https://openalex.org/fields/17
CS_FIELD_ID = "17"

YEARS = [2018, 2019, 2020]
SAMPLE_SIZE = 1800
SEED = 42 
PER_PAGE = 100
SLEEP = 0.12


# HELPER FUNCTIONS ===============================================================

def build_params(year: int, page: int) -> dict:
    return {
        "filter": (
            f"topics.field.id:{CS_FIELD_ID},"
            f"publication_year:{year},"
            "type:article"
        ),
        "select": (
            "id,title,publication_year,cited_by_count,"
            "authorships,primary_location,open_access,type"
        ),
        "sample": SAMPLE_SIZE,
        "seed": SEED,
        "per-page": PER_PAGE,
        "page": page,
    }


def fetch_page(params: dict) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"  API error {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    data = resp.json()

    if params["page"] == 1:
        print(f"  API reports {data.get('meta', {}).get('count', '?')} total in sample")

    return data


def deduplicate_papers(papers: list) -> list:
    seen_ids = set()
    unique_papers = []

    for paper in papers:
        paper_id = paper.get("id")
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(paper)

    return unique_papers


def summarize_by_year(papers: list) -> dict:
    summary = {}
    for year in YEARS:
        summary[year] = sum(1 for p in papers if p.get("publication_year") == year)
    return summary


# FETCH PAPERS ================================================================

def collect_papers() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_papers = []

    total_pages = SAMPLE_SIZE // PER_PAGE

    for year in YEARS:
        print(f"\n=== Fetching {year} papers ===")

        for page in range(1, total_pages + 1):
            params = build_params(year, page)

            try:
                data = fetch_page(params)
            except requests.HTTPError as e:
                print(f"  HTTP error on page {page}: {e}. Retrying in 5s...")
                time.sleep(5)
                try:
                    data = fetch_page(params)
                except requests.RequestException:
                    print(f"  Retry failed. Skipping page {page}.")
                    continue
            except requests.RequestException as e:
                print(f"  Request failed: {e}. Skipping page {page}.")
                continue

            results = data.get("results", [])
            if not results:
                print(f"  No results on page {page}, stopping early.")
                break

            all_papers.extend(results)
            print(f"  Page {page:02d} | +{len(results)} papers | total so far: {len(all_papers)}")
            time.sleep(SLEEP)

    return all_papers


# MAIN ====================================================================

if __name__ == "__main__":
    print(f"Starting collection at {datetime.now().strftime('%H:%M:%S')}")

    papers = collect_papers()
    unique_papers = deduplicate_papers(papers)
    year_counts = summarize_by_year(unique_papers)

    payload = {
        "collected_at": datetime.now().isoformat(),
        "total_papers_raw": len(papers),
        "total_papers_unique": len(unique_papers),
        "years": YEARS,
        "counts_by_year": year_counts,
        "papers": unique_papers,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nCollection summary:")
    print(f"  Raw papers collected:    {len(papers)}")
    print(f"  Unique papers collected: {len(unique_papers)}")
    print(f"  Counts by year:          {year_counts}")
    print(f"\nDone. Saved to {OUTPUT_FILE}")
