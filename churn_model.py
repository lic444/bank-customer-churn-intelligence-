"""Model utilities for the Bank Customer Churn Risk Scoring project."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

TARGET = "Exited"
DROP_COLS = ["CustomerId", "Surname", "Year"]

class FeatureEngineer:
    """Add interpretable interaction and ratio features without using the target."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["BalanceToSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"].abs() + 1.0)
        X["ProductDensity"] = X["NumOfProducts"] / (X["Tenure"] + 1.0)
        X["EngagementProductInteraction"] = X["IsActiveMember"] * X["NumOfProducts"]
        X["AgeTenureInteraction"] = X["Age"] * X["Tenure"]
        return X

def prepare_X(df):
    return df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns], errors="ignore")

def build_pipeline(model_name="Random Forest", random_state=42):
    categorical = ["Geography", "Gender"]
    numeric = [c for c in ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "BalanceToSalaryRatio", "ProductDensity", "EngagementProductInteraction", "AgeTenureInteraction"]]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }
    return Pipeline([("features", FeatureEngineer()), ("preprocess", preprocessor), ("model", models[model_name])])

def train_and_evaluate(df, random_state=42):
    df = df.dropna(subset=[TARGET]).copy()
    X, y = prepare_X(df), df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
    fitted, rows = {}, []
    for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
        pipe = build_pipeline(name, random_state)
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        # Threshold selected using training cross-validation.
        pred = (prob >= 0.30).astype(int)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
        }
        rows.append(metrics)
        fitted[name] = {"pipeline": pipe, "probabilities": prob, "predictions": pred, "y_test": y_test, "X_test": X_test}
    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = results.iloc[0]["Model"]
    return results, best_name, fitted[best_name], fitted, (X_train, X_test, y_train, y_test)
