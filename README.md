# Bank Customer Churn Intelligence

An interactive machine-learning dashboard for exploring historical bank
customer churn, comparing classification models, estimating churn risk,
and interpreting model predictions.

## Project overview

Customer churn occurs when a customer stops using a company's services.
This project explores historical customer data to build a model that
estimates churn probability and presents the results in a Streamlit
dashboard.

**Project type:** Machine Learning / Classification / Interactive
Dashboard\
**Dataset:** `data/European_Bank.csv`\
**Target column:** `Exited` (`1` = churned, `0` = retained)

> **Important:** This is an educational and portfolio prototype. Its
> predictions are estimates, not guarantees, and should not be used to
> make real customer decisions without further validation, governance,
> privacy review, fairness assessment, calibration, and monitoring.

## Dashboard features

-   **Executive Overview:** Dataset size, observed churn rate, churn
    count, and churn distribution.
-   **Explore Customers:** Filter customers by geography, status, and
    age; inspect records and download filtered data.
-   **Churn Risk Calculator:** Enter a hypothetical customer profile and
    estimate churn probability.
-   **What-if Simulator:** Compare model scores for a sample customer
    under changed profile inputs. These comparisons are model
    associations, not causal treatment effects.
-   **Model Performance:** Compare Logistic Regression, Random Forest,
    and Gradient Boosting using classification metrics, a confusion
    matrix, and ROC curve.
-   **Feature Insights:** Use permutation importance to show how
    shuffling a feature affects the model's ROC-AUC.

## Dataset

The project uses these columns:

-   `CreditScore`
-   `Geography`
-   `Gender`
-   `Age`
-   `Tenure`
-   `Balance`
-   `NumOfProducts`
-   `HasCrCard`
-   `IsActiveMember`
-   `EstimatedSalary`
-   `Exited` --- prediction target

The dataset also contains `CustomerId`, `Surname`, and `Year`.
Identifier columns and the constant `Year` field are excluded from model
inputs.

Dataset checks performed during development found 10,000 rows, no
missing values, no duplicate full rows, no duplicate customer IDs, and a
target distribution of 7,963 retained customers and 2,037 churned
customers (20.37% churn). Verify these values if the dataset file is
replaced.

## Methodology

1.  Load the CSV dataset and separate the target `Exited` from the input
    features.
2.  Exclude `CustomerId`, `Surname`, and `Year` from model inputs.
3.  Add deterministic interaction features:
    -   `BalanceToSalaryRatio`
    -   `ProductDensity`
    -   `EngagementProductInteraction`
    -   `AgeTenureInteraction`
4.  Apply preprocessing inside a scikit-learn pipeline:
    -   Median imputation and standardization for numeric features.
    -   Most-frequent imputation and one-hot encoding for categorical
        features.
5.  Split the data into an 80% training set and a 20% held-out test set,
    stratified by the target, using `random_state=42`.
6.  Compare Logistic Regression, Random Forest, and Gradient Boosting.
7.  Evaluate with accuracy, precision, recall, F1-score, and ROC-AUC.
8.  Use five-fold stratified cross-validation on the training data to
    compare ROC-AUC and select a classification threshold for Gradient
    Boosting.
9.  Calculate permutation importance on the held-out test set.

## Model results

The following held-out test metrics were obtained at a classification
threshold of `0.30`:

  Model                   Accuracy   Precision   Recall   F1-score   ROC-AUC
  --------------------- ---------- ----------- -------- ---------- ---------
  Gradient Boosting         0.8490      0.6241   0.6486     0.6361    0.8674
  Random Forest             0.6995      0.3905   0.8501     0.5352    0.8598
  Logistic Regression       0.4785      0.2686   0.9066     0.4144    0.7759

Gradient Boosting had the highest F1-score and ROC-AUC in this
comparison. The `0.30` threshold was selected to maximize
cross-validated F1 for Gradient Boosting. It is used as a common
comparison threshold in the dashboard; it was not independently
optimized for each model.

### Threshold and confusion matrix

For Gradient Boosting at threshold `0.30`, the held-out test confusion
matrix counts were:

-   True negatives: 1,434
-   False positives: 159
-   False negatives: 143
-   True positives: 264

Threshold choice affects precision and recall. A lower threshold
generally flags more customers as potential churners, which can catch
more actual churners but also create more false positives. An
operational threshold should be chosen based on the cost of missed
churn, outreach capacity, and the cost of contacting customers who would
not churn.

## Feature importance

The leading permutation-importance values observed in the dashboard
were:

  Feature             Mean decrease in ROC-AUC
  ----------------- --------------------------
  Age                                   0.1223
  NumOfProducts                         0.1136
  IsActiveMember                        0.0335
  Balance                               0.0265
  Geography                             0.0162
  Gender                                0.0078
  CreditScore                           0.0027
  EstimatedSalary                       0.0004
  HasCrCard                            -0.0002
  Tenure                               -0.0008

Permutation importance measures the model's predictive reliance on a
feature in this evaluation. It does **not** establish causality,
fairness, or that changing a feature would change a customer's churn
outcome. Correlated features may share importance, and small or negative
values should be interpreted cautiously.

## Technology stack

-   Python
-   pandas and NumPy
-   scikit-learn
-   Streamlit
-   Plotly

## Project structure

``` text
bank_churn_project/
├── app.py
├── churn_model.py
├── validate_model.py
├── threshold_analysis.py
├── validation_results.csv
├── threshold_comparison.csv
└── data/
    └── European_Bank.csv
```

The structure above describes the files used during development; include
only files that are present in your submitted project.

## Run locally

### 1. Create and activate a virtual environment

On Windows PowerShell:

``` powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, use the virtual environment's Python executable directly
if activation is blocked.

### 2. Install dependencies

``` powershell
python -m pip install pandas numpy scikit-learn streamlit plotly
```

### 3. Start the dashboard

Run from the project root:

``` powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`.

## Validation and limitations

-   The held-out test split provides an estimate of performance on
    unseen rows from this dataset; it does not guarantee performance on
    future customers or another bank.
-   The dataset has a class imbalance: approximately 20.37% of records
    are churners.
-   The threshold is selected for F1-score, not from a formal
    bank-specific cost-benefit analysis.
-   Predicted probabilities have not necessarily been calibrated.
-   Permutation importance is model-specific and does not establish
    causation.
-   Assess fairness and possible proxy effects before considering
    real-world use.
-   Do not commit private customer data, secrets, or credentials to a
    public repository.

## Future improvements

-   Add automated tests for data validation and prediction inputs.
-   Compare precision-recall curves and business-cost-based thresholds.
-   Evaluate probability calibration and threshold stability.
-   Add model versioning and reproducible dependency versions.
-   Add fairness analysis and monitoring for data or performance drift.
-   Deploy only after appropriate privacy, security, governance, and
    business validation.

## Author

Add your name, contact details, and portfolio links here before
publishing.
