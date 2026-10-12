
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from churn_model import build_pipeline, prepare_X

# 1. Load and validate the target
df = pd.read_csv("data/European_Bank.csv")

if df["Exited"].isna().any():
    raise ValueError("The target contains missing values.")

if not set(df["Exited"].unique()).issubset({0, 1}):
    raise ValueError("Exited must contain only 0 and 1.")

X = prepare_X(df)
y = df["Exited"].astype(int)

# 2. Create a stratified holdout test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 3. Cross-validate only on the training data
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

models = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
]

results = []

for name in models:
    pipeline = build_pipeline(name, random_state=42)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )

    # Fit on the complete training partition
    pipeline.fit(X_train, y_train)

    # Evaluate once on the holdout test partition
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "Model": name,
        "CV ROC-AUC Mean": cv_scores.mean(),
        "CV ROC-AUC Std": cv_scores.std(),
        "Test Accuracy": accuracy_score(y_test, predictions),
        "Test Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Test Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Test F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Test ROC-AUC": roc_auc_score(y_test, probabilities),
    })

results_df = pd.DataFrame(results)

print("\n=== MODEL VALIDATION RESULTS ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("validation_results.csv", index=False)
print("\nSaved results to validation_results.csv")
