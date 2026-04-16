import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Config — keep in sync with train_mlp.py ========================================
DATA_PATH = "data/processed/papers.csv"
TARGET_COL = "high_impact"

FEATURE_COLS = [
    "log_num_authors",
    "log_num_institutions",
    "log_num_countries",
    "is_multi_institution",
    "is_international",
    "has_elite_affiliation",
    "has_industry",
    "has_gov_nonprofit",
    "has_us_institution",
    "is_open_access",
    "is_journal",
]

RANDOM_STATE = 42

# Must match train_mlp.py
DECISION_THRESHOLD = 0.1


# Main ============================================================================
def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    data = pd.concat([X, y], axis=1).dropna()
    X = data[FEATURE_COLS]
    y = data[TARGET_COL].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    numeric_cols = ["log_num_authors", "log_num_institutions", "log_num_countries"]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)

    accuracy  = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test, preds, zero_division=0)

    print(f"Logistic Regression Baseline (threshold={DECISION_THRESHOLD}):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


if __name__ == "__main__":
    main()