# Predicting High-Impact Scientific Papers via Institutional and Collaboration Signals


---

## Overview

Not all scientific papers are cited equally. A small fraction of publications accumulate the vast majority of citations, shaping research agendas and funding decisions for years after publication. What structural signals — present at the time of publication — distinguish these high-impact papers from the rest?

This project investigates whether **institutional prestige** and **collaboration structure** can predict whether a Computer Science paper will become highly cited. Using metadata from the [OpenAlex](https://openalex.org/) open academic graph, we treat citation impact as a binary classification problem and evaluate the predictive power of features that are observable at the time of publication, before any downstream citation accumulation occurs.

---

## Research Question

> *Can institutional prestige and collaboration network features predict whether a scientific paper will fall within the top 10% of citations in its cohort?*

The prediction target is defined as:

```
high_impact = 1  if citation count ≥ 90th percentile of dataset
high_impact = 0  otherwise
```

---

## Dataset

**Source:** [OpenAlex API](https://docs.openalex.org/)

**Scope:** ~3,000–5,000 Computer Science papers published between 2018 and 2020

**Fields used:**
- Title, publication year, citation count
- Author list and count
- Institutional affiliations per author

**Rationale for time window:** Papers from 2018–2020 have had 4–6+ years to accumulate citations, making the citation distribution more stable than for very recent work, while avoiding older papers whose citation patterns reflect a structurally different publishing landscape.

> Raw and processed data are stored in `data/`. See `notebooks/01_data_collection.ipynb` for the full collection and filtering pipeline.

---

## Feature Engineering

Features are designed to be fully observable at the time of publication — no post-publication signals are used.

| Feature | Description |
|---|---|
| `num_authors` | Total number of authors on the paper |
| `num_institutions` | Number of distinct institutional affiliations |
| `is_multi_institution` | Binary flag: does the paper involve more than one institution? |
| `has_elite_affiliation` | Binary flag: does any author affiliate with a top-tier institution? |
| `num_elite_affiliations` | Count of elite institutions represented |
| `publication_year` | Year of publication (2018, 2019, or 2020) |
| `paper_age` | Years elapsed since publication as of data collection date |

**Elite institutions (initial list):** MIT, Stanford, Harvard, UC Berkeley, Carnegie Mellon University

> This list is a modeling choice and is discussed further in the Limitations section below.

---

## Methods

### Model 1 — Manual Logistic Regression (Python)

A logistic regression classifier implemented from scratch using **NumPy only** (no Scikit-learn or similar). The model estimates:

```
P(high_impact = 1 | x) = σ(xᵀβ)
```

where σ is the sigmoid function and β is learned via gradient descent. This implementation includes:
- Mini-batch gradient descent with configurable learning rate and epochs
- Binary cross-entropy loss tracking
- Evaluation on a held-out test set (accuracy, ROC-AUC, confusion matrix)

### Model 2 — Bayesian Logistic Regression (R)

A Bayesian treatment of the same classification problem, implemented in **R** using Stan/brms or a manual MCMC approach. Rather than point estimates, the Bayesian model yields full posterior distributions over each coefficient, allowing us to:
- Quantify uncertainty in the effect of institutional prestige
- Examine whether the `has_elite_affiliation` effect is credibly non-zero
- Compare posterior predictive accuracy with the frequentist baseline

Weakly informative priors (e.g., Normal(0, 1) on log-odds scale) will be used unless domain knowledge suggests otherwise.

---

## Workflow

```
1. Data Collection       →  OpenAlex API query, filtering to CS papers 2018–2020
2. Preprocessing         →  Deduplication, missing value handling, citation percentile computation
3. Feature Engineering   →  Author/institution counts, elite affiliation flags, temporal features
4. EDA                   →  Citation distribution, feature correlations, class balance check
5. Model Training        →  Manual logistic regression (Python) + Bayesian logistic regression (R)
6. Evaluation            →  Accuracy, ROC-AUC, confusion matrix, posterior credible intervals
7. Interpretation        →  Coefficient analysis, uncertainty quantification, comparison across models
```

---

## Repository Structure

```
citation-impact/
│
├── data/
│   ├── raw/                  # Raw API responses (JSON)
│   └── processed/            # Cleaned feature matrices (CSV)
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   └── 03_results_visualization.ipynb
│
├── src/
│   ├── collect_data.py       # OpenAlex API collection script
│   ├── build_dataset.py      # Feature engineering pipeline
│   └── logistic_regression.py  # Manual logistic regression (NumPy)
│
├── r/
│   └── bayesian_logit.R      # Bayesian logistic regression (R/Stan)
│
├── docs/
│   └── model_planning.md     # Notes on modeling decisions and prior choices
│
└── README.md
```

---

## Evaluation Metrics

- **Accuracy** — overall classification performance
- **ROC-AUC** — discriminative ability under class imbalance
- **Confusion Matrix** — breakdown of false positives / false negatives
- **Posterior Credible Intervals** (Bayesian model) — uncertainty in coefficient estimates

---

## Limitations and Known Challenges

- **Elite institution list is manually defined.** Membership is subjective and U.S.-centric; papers from strong non-U.S. institutions (e.g., ETH Zurich, Tsinghua) may be systematically underrepresented in the prestige signal.
- **Citation counts are a lagging signal.** Papers from 2020 have had fewer years to accumulate citations than those from 2018, which may introduce bias in the 90th-percentile threshold unless computed within-year.
- **Class imbalance.** By construction, only ~10% of papers are labeled `high_impact = 1`. This may require threshold tuning or class-weighted training.
- **Omitted variables.** Citation impact is also driven by research topic, venue prestige, and paper quality — none of which are captured here. The models measure correlation, not causation.

---

## Requirements

**Python:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `requests`

**R:** `tidyverse`, `brms` (or `rstan`), `bayesplot`

Install Python dependencies:
```bash
pip install -r requirements.txt
```
