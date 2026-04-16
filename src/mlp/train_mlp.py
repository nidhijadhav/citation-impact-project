import json
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
from src.mlp.mlp import MLP


# Config =========================================================================
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

EPOCHS = 300
LEARNING_RATE = 0.05
HIDDEN_DIM = 16
RANDOM_STATE = 42

# Positive class weight: upweights minority class in the loss.
POS_WEIGHT = 6.0

# L2 regularization: keeps weights small and closes the train/val gap.
WEIGHT_DECAY = 0.01

# Decision threshold: 0.52 is the best-F1 point from the PR curve.
DECISION_THRESHOLD = 0.52

METRICS_PATH = "results/mlp_metrics.json"
LOSS_FIG_PATH = "results/mlp_loss.png"
CM_FIG_PATH = "results/mlp_confusion_matrix.png"
PR_FIG_PATH  = "results/mlp_pr_curve.png"


# Helpers =========================================================================

def plot_loss(train_losses, val_losses, path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("MLP Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(cm, path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_pr_curve(y_test, y_probs, threshold, path):
    prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
    f1s = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    best_idx = f1s.argmax()
    chosen_idx = np.argmin(np.abs(thresholds - threshold))
 
    plt.figure(figsize=(8, 5))
    plt.plot(rec, prec, color="#2c7fb8", linewidth=1.5, label="PR curve")
    plt.scatter(
        [rec[best_idx]], [prec[best_idx]], color="#d7301f", zorder=5,
        label=f"Best F1 threshold={thresholds[best_idx]:.2f} (F1={f1s[best_idx]:.3f})",
    )
    plt.scatter(
        [rec[chosen_idx]], [prec[chosen_idx]], color="orange", zorder=5, marker="D",
        label=f"Chosen threshold={threshold}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# Main ============================================================================
def main():
    # load dataset
    df = pd.read_csv(DATA_PATH)

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # select features and target
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # convert to numeric and coerce any errors
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # drop missing rows
    data = pd.concat([X, y], axis=1).dropna()
    X = data[FEATURE_COLS]
    y = data[TARGET_COL].astype(int)

    print("Dataset loaded.")
    print("X shape:", X.shape)
    print("Positive class rate:", y.mean())

    # train-val-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # normalize numeric features
    scaler = StandardScaler()

    numeric_cols = [
        "log_num_authors",
        "log_num_institutions",
        "log_num_countries",
    ]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    X_train = X_train.to_numpy(dtype=float)
    X_val = X_val.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    # init model
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        seed=RANDOM_STATE,
    )

    # train model
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # forward + loss + backward + update
        train_loss = model.train_step(X_train, y_train, learning_rate=LEARNING_RATE, pos_weight=POS_WEIGHT, weight_decay=WEIGHT_DECAY)
        train_losses.append(train_loss)

        # validation loss
        val_probs = model.predict_proba(X_val)
        val_loss = model.compute_loss(y_val, val_probs, pos_weight=POS_WEIGHT)
        val_losses.append(val_loss)

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        

    # eval on test set
    y_probs = model.predict_proba(X_test).flatten()
    y_pred = (y_probs >= DECISION_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    print("MLP Metrics:")
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # save metrics to JSON
    metrics = {
        "decision_threshold": DECISION_THRESHOLD,
        "accuracy":  float(accuracy),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "auc":       float(auc),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # plot loss curves
    plot_loss(train_losses, val_losses, LOSS_FIG_PATH)

    # plot confusion matrix
    plot_confusion_matrix(cm, CM_FIG_PATH)

    # plot precision-recall curve
    plot_pr_curve(y_test, y_probs, DECISION_THRESHOLD, PR_FIG_PATH)

    print("\nSaved outputs:")
    print(METRICS_PATH)
    print(LOSS_FIG_PATH)
    print(CM_FIG_PATH)
    print(PR_FIG_PATH)

if __name__ == "__main__":
    main()