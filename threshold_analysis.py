
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from churn_model import build_pipeline, prepare_X

# 1. Load and prepare the data
df = pd.read_csv("data/European_Bank.csv")

X = prepare_X(df)
y = df["Exited"].astype(int)

# 2. Create a stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 3. Generate out-of-fold probabilities using training data only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

model = build_pipeline("Gradient Boosting", random_state=42)

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

# 4. Compare possible thresholds using out-of-fold predictions
thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

rows = []

for threshold in thresholds:
    predictions = (oof_probabilities >= threshold).astype(int)

    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_train, predictions, zero_division=0
        ),
    })

results = pd.DataFrame(rows)

print("\n=== TRAINING CROSS-VALIDATION: THRESHOLD COMPARISON ===")
print(results.round(4).to_string(index=False))

# 5. Select the threshold with the highest cross-validated F1
best_threshold = float(
    results.loc[results["F1"].idxmax(), "Threshold"]
)

print(f"\nSelected threshold (highest CV F1): {best_threshold:.2f}")

# 6. Fit on the complete training split and evaluate on the test set
model.fit(X_train, y_train)

test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

print("\n=== HELD-OUT TEST RESULTS ===")
print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, test_predictions, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_probabilities):.4f}")

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

print("\n=== CONFUSION MATRIX COUNTS ===")
print(f"True negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")

results.to_csv("threshold_comparison.csv", index=False)
print("\nSaved threshold comparison to threshold_comparison.csv")
