"""
fetch_institution_ids.py
------------------------
Looks up OpenAlex IDs for a list of elite institutions and prints
them ready to paste into build_dataset.py.

"""

import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY", "")
if not API_KEY:
    raise ValueError("OPENALEX_API_KEY not set. Add it to your .env file.")

INSTITUTIONS = [
    # US — top CS programs
    "Massachusetts Institute of Technology",
    "Stanford University",
    "Harvard University",
    "University of California Berkeley",
    "Carnegie Mellon University",
    "Princeton University",
    "Cornell University",
    "Columbia University",
    "University of Illinois Urbana-Champaign",
    "University of Michigan",
    "University of Washington",
    "Georgia Institute of Technology",
    "University of Texas at Austin",
    "University of California San Diego",
    "University of California Los Angeles",
    "New York University",
    "Yale University",
    "Caltech",
    "Duke University",
    "University of Pennsylvania",
    # International — top CS programs
    "University of Oxford",
    "University of Cambridge",
    "ETH Zurich",
    "Tsinghua University",
    "Peking University",
    "University of Toronto",
    "National University of Singapore",
    "Nanyang Technological University",
    "Imperial College London",
    "University College London",
    "Ecole Polytechnique Federale de Lausanne",
    "Technische Universitat Munchen",
    "University of Edinburgh",
    "University of Amsterdam",
    "Seoul National University",
    "Tokyo University",
    "Australian National University",
    "McGill University",
    "Universite de Montreal",
    # Industry research labs
    "Google",
    "Microsoft",
    "Meta",
    "Apple",
    "Amazon",
    "DeepMind",
    "OpenAI",
    "IBM",
    "Adobe",
    "Nvidia",
]


def lookup(name: str) -> tuple[str, str] | None:
    url = "https://api.openalex.org/institutions"
    params = {"search": name, "per_page": 1, "api_key": API_KEY}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None
    r = results[0]
    return r["id"], r["display_name"]


if __name__ == "__main__":
    print("Fetching institution IDs from OpenAlex...\n")
    found, missing = {}, []

    for name in INSTITUTIONS:
        result = lookup(name)
        if result:
            openalex_id, display_name = result
            found[display_name] = openalex_id
            print(f"  {display_name:50s}  {openalex_id}")
        else:
            missing.append(name)
            print(f"  {name} — not found")
        time.sleep(0.15)

    print("\n\n# ===== Add to build_dataset.py \n")
    print("ELITE_IDS = {")
    for display_name, oid in found.items():
        short = oid.split("/")[-1]
        print(f'    "{oid}",  # {display_name}')
    print("}")

    if missing:
        print(f"\n# Not found: {missing}")