# Predicting High-Impact Scientific Papers Using Institutional and Collaboration Features 
---

## Overview

Not all scientific papers get cited equally. A small fraction of publications end up accumulating the vast majority of citations, while most of other papers get relatively little attention. This project explores if the structural features of a paper can predict whether a paper ends up in the top 10% of citations in its field.

We're focusing on Computer Science papers published between 2018 and 2020, using metadata pulled from the OpenAlex open academic graph. The idea is to only use features that would be observable at the time of publication, before any citations have accumulated, which makes this a more realistic prediction problem.

---

## Research Question

Can institutional affiliation and collaboration-related features help predict whether a scientific paper will become highly cited?

We define the target as:

```
high_impact = 1  if citation count >= 90th percentile within publication year
high_impact = 0  otherwise
```

---

## Dataset

**Source:** OpenAlex API (https://docs.openalex.org/)

**Scope:** ~5,000 Computer Science papers published between 2018 and 2020, randomly sampled using OpenAlex's sampling endpoint to get a representative distribution across citation counts.

**Fields collected:**
- Paper ID, title, publication year, citation count
- Author list and institutional affiliations per author
- Country codes per authorship
- Open access status and venue type (journal vs. other)

**Why 2018-2020?** Papers from this window have had at least 6 years to accumulate citations, which makes the citation distribution more stable. We also calculated the 90th percentile threshold within each year separately to avoid penalizing newer papers that haven't had a lot of time to accumulate citations.

---

## Features

All features are observable at publication time.

| Feature | Description |
|---|---|
| `num_authors` | Total number of authors |
| `num_institutions` | Number of distinct institutional affiliations |
| `num_countries` | Number of distinct countries represented |
| `is_multi_institution` | 1 if more than one institution is involved |
| `is_international` | 1 if authors come from more than one country |
| `has_elite_affiliation` | 1 if any author is affiliated with a top CS institution |
| `has_industry` | 1 if any author is affiliated with a company (Google, Meta, etc.) |
| `has_gov_nonprofit` | 1 if any author is from a government or nonprofit lab |
| `has_us_institution` | 1 if any author is from a US institution |
| `is_open_access` | 1 if the paper is open access |
| `is_journal` | 1 if published in a journal (vs. conference or other) |

Log-transformed versions (`log_num_authors`, `log_num_institutions`, `log_num_countries`) are added by `build_dataset.py` and used as model inputs.

Elite institutions are matched using OpenAlex's official institution IDs. The list includes 36 institutions covering top US programs, international universities, and major industry research labs.

NOTE ON MISSING DATA: OpenAlex doesn't always report institution or country information for every author. `num_institutions` and `num_countries` are set to 0 instead of flooring to 1, to avoid misrepresenting unknown data.

---

## Models

**Decision threshold:** Some models use a threshold of 0.1 (rather than 0.5) for converting predicted probabilities to binary labels. With ~10% positive class rate, 0.5 suppresses nearly all positive predictions. So by setting the threshold near the base rate, allows for some more meaningful recall. 

**Baseline: Logistic Regression (Python)**
sklearn LogisticRegression with `class_weight='balanced'`, evaluated at threshold 0.1 to match the MLP.

**Model 1: MLP (Python)** 
A two-layer neural network implemented from scratch using NumPy. Uses ReLU activations in the hidden layer and a sigmoid output for binary classification, trained with weighted binary cross-entropy loss and gradient descent.

**Model 2: Bayesian Logistic Regression (R)**
Bayesian logistic regression with a N(0, 25) prior on coefficients, posterior sampled via Metropolis-Hastings MCMC (15,000 iterations, 5,000 burn-in). Inference uses the posterior mean. Evaluated at threshold 0.5.

---

## Repo Structure

```
citation-impact/
│
├── data/
│   ├── raw/                      # Raw API responses (JSON, gitignored)
│   └── processed/                # Feature matrix (CSV, gitignored)
│
├── notebooks/
│   └── 00_eda.ipynb              # Exploratory data analysis
│
├── src/
│   ├── collect_data.py           # Pulls data from OpenAlex API
│   ├── build_dataset.py          # Feature engineering
│   ├── scripts/
│   │   └── fetch_institution_ids.py  # Verify/fetch elite institution IDs
│   └── mlp/
│       ├── mlp.py                # NumPy MLP implementation
│       ├── train_mlp.py          # MLP training script
│       ├── baseline_lr.py        # LR baseline (same split as MLP)
│       └── test_mlp.py           # Sanity check on synthetic data
│
├── results/                      # Saved figures, metrics, scaler
├── .env.example
└── README.md
```

---

## How to Run

1. Copy `.env.example` to `.env` and add your OpenAlex API key (free at openalex.org/settings/api)
2. Run `python src/collect_data.py` to pull raw data
3. Run `python src/build_dataset.py` to build the feature matrix
4. Run `python -m src.mlp.train_mlp` to train the MLP
5. Run `python -m src.mlp.baseline_lr` to run the LR baseline on the same split
6. Run `Rscript src/bayesian_lr.R` to run the Bayesian model
