"""
build_dataset.py
----------------
Loads raw OpenAlex JSON, engineers features, and saves a clean CSV ready for modeling.
"""

import json
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime


# PATHS =========================================================================

BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
RAW_FILE     = os.path.join(BASE_DIR, "data", "raw", "openalex_cs_papers.json")
OUT_FILE     = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")
CURRENT_YEAR = datetime.now().year


# ELITE INSTITUTIONS ============================================================

ELITE_IDS = {
    "https://openalex.org/I63966007",    # Massachusetts Institute of Technology
    "https://openalex.org/I97018004",    # Stanford University
    "https://openalex.org/I136199984",   # Harvard University
    "https://openalex.org/I95457486",    # University of California, Berkeley
    "https://openalex.org/I74973139",    # Carnegie Mellon University
    "https://openalex.org/I20089843",    # Princeton University
    "https://openalex.org/I205783295",   # Cornell University
    "https://openalex.org/I78577930",    # Columbia University
    "https://openalex.org/I157725225",   # University of Illinois Urbana-Champaign
    "https://openalex.org/I27837315",    # University of Michigan
    "https://openalex.org/I201448701",   # University of Washington
    "https://openalex.org/I130701444",   # Georgia Institute of Technology
    "https://openalex.org/I86519309",    # The University of Texas at Austin
    "https://openalex.org/I36258959",    # University of California San Diego
    "https://openalex.org/I161318765",   # University of California, Los Angeles
    "https://openalex.org/I57206974",    # New York University
    "https://openalex.org/I32971472",    # Yale University
    "https://openalex.org/I122411786",   # California Institute of Technology
    "https://openalex.org/I170897317",   # Duke University
    "https://openalex.org/I79576946",    # University of Pennsylvania
    "https://openalex.org/I40120149",    # University of Oxford
    "https://openalex.org/I241749",      # University of Cambridge
    "https://openalex.org/I35440088",    # ETH Zurich
    "https://openalex.org/I99065089",    # Tsinghua University
    "https://openalex.org/I20231570",    # Peking University
    "https://openalex.org/I185261750",   # University of Toronto
    "https://openalex.org/I165932596",   # National University of Singapore
    "https://openalex.org/I172675005",   # Nanyang Technological University
    "https://openalex.org/I47508984",    # Imperial College London
    "https://openalex.org/I45129253",    # University College London
    "https://openalex.org/I98677209",    # University of Edinburgh
    "https://openalex.org/I887064364",   # University of Amsterdam
    "https://openalex.org/I139264467",   # Seoul National University
    "https://openalex.org/I161296585",   # Tokyo University of Science
    "https://openalex.org/I118347636",   # Australian National University
    "https://openalex.org/I5023651",     # McGill University
    "https://openalex.org/I1291425158",  # Google (United States)
    "https://openalex.org/I1290206253",  # Microsoft (United States)
    "https://openalex.org/I2252078561",  # Meta (Israel)
    "https://openalex.org/I1311269955",  # Apple (Israel)
    "https://openalex.org/I1311688040",  # Amazon (United States)
    "https://openalex.org/I4210090411",  # Google DeepMind (United Kingdom)
    "https://openalex.org/I4210161460",  # OpenAI (United States)
    "https://openalex.org/I1341412227",  # IBM (United States)
    "https://openalex.org/I1306409833",  # Adobe Systems (United States)
    "https://openalex.org/I4210127875",  # Nvidia (United States)
}


# FEATURE EXTRACTION ============================================================

def extract_features(paper: dict) -> dict | None:
    paper_id    = paper.get("id", "")
    year        = paper.get("publication_year")
    citations   = paper.get("cited_by_count")
    authorships = paper.get("authorships", [])

    if year is None or citations is None:
        return None

    # Collaboration features 
    num_authors     = len(authorships)
    institution_ids = set()
    inst_types      = []
    country_codes   = set()

    for authorship in authorships:
        for cc in authorship.get("countries", []):
            if cc:
                country_codes.add(cc)
        for inst in authorship.get("institutions", []):
            inst_id = inst.get("id")
            if inst_id:
                institution_ids.add(inst_id)
            inst_type = inst.get("type", "")
            if inst_type:
                inst_types.append(inst_type)

    num_institutions = len(institution_ids)
    num_countries    = len(country_codes)
    is_multi_inst    = int(num_institutions > 1)
    is_international = int(num_countries > 1)

    # Institutional features 
    has_elite       = int(bool(institution_ids & ELITE_IDS))
    has_industry    = int("company" in inst_types)
    has_gov_nonprof = int("government" in inst_types or "nonprofit" in inst_types)
    has_us_inst     = int("US" in country_codes)

    # Venue features 
    primary_loc = paper.get("primary_location") or {}
    is_oa       = int(bool(primary_loc.get("is_oa")))
    source      = primary_loc.get("source") or {}
    is_journal  = int(source.get("type", "") == "journal")

    # Temporal features 
    paper_age = CURRENT_YEAR - year

    return {
        "paper_id":             paper_id,
        "publication_year":     year,
        "cited_by_count":       citations,
        "num_authors":          num_authors,
        "num_institutions":     num_institutions,
        "num_countries":        num_countries,
        "is_multi_institution": is_multi_inst,
        "is_international":     is_international,
        "has_elite_affiliation":has_elite,
        "has_industry":         has_industry,
        "has_gov_nonprofit":    has_gov_nonprof,
        "has_us_institution":   has_us_inst,
        "is_open_access":       is_oa,
        "is_journal":           is_journal,
        "paper_age":            paper_age,
    }


# IMPACT LABELS ==================================================================

def add_high_impact_label(df: pd.DataFrame, percentile: float = 90.0) -> pd.DataFrame:
    df["high_impact"] = 0
    for year, group in df.groupby("publication_year"):
        threshold = np.percentile(group["cited_by_count"], percentile)
        mask = (df["publication_year"] == year) & (df["cited_by_count"] >= threshold)
        df.loc[mask, "high_impact"] = 1
        print(f"  {year}: threshold = {threshold:.0f} citations | "
              f"high-impact papers = {mask.sum()} / {len(group)}")
    return df


# LOG TRANSFORMATIONS ============================================================

def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["num_authors", "num_institutions", "num_countries"]:
        df[f"log_{col}"] = np.log1p(df[col])
    return df


# MAIN ===========================================================================

def main():
    print(f"Loading raw data from {RAW_FILE} ...")
    with open(RAW_FILE) as f:
        raw = json.load(f)

    papers = raw.get("papers", [])
    print(f"  {len(papers)} raw records loaded.")

    rows    = []
    skipped = 0
    for paper in papers:
        features = extract_features(paper)
        if features is None:
            skipped += 1
        else:
            rows.append(features)

    df = pd.DataFrame(rows)

    before = len(df)
    df = df[df["num_authors"] > 0].reset_index(drop=True)
    df = add_high_impact_label(df)
    df = add_log_features(df)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"\nDataset saved to {OUT_FILE}")
    print(f"Shape: {df.shape}")
    print(f"\nClass balance:\n{df['high_impact'].value_counts(normalize=True).round(3)}")
    print(f"\nSample:\n{df.head(3).to_string()}")


if __name__ == "__main__":
    main()
