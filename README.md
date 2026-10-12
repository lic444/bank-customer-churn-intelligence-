# Predictive Modeling and Risk Scoring for Bank Customer Churn

An end-to-end machine-learning portfolio project that estimates bank customer churn probabilities and presents exploratory analytics, model evaluation, individual risk scoring, and what-if scenarios in Streamlit.

## Features
- Data exploration and churn distribution analysis
- Feature engineering: balance-to-salary ratio, product density, engagement-product interaction, age-tenure interaction
- Comparison of Logistic Regression, Random Forest, and Gradient Boosting
- Stratified 80/20 train-test split and Accuracy, Precision, Recall, F1, ROC-AUC
- Individual churn probability and illustrative risk bands
- What-if scenario simulator
- Permutation feature importance and downloadable filtered data

## Dataset
`data/European_Bank.csv` (10,000 records). Target: `Exited` (1 = churned, 0 = retained). The supplied dataset has a constant `Year` column; it is dropped along with identifiers (`CustomerId`, `Surname`) to avoid using non-predictive identifiers.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```text
bank_churn_project/
├── app.py
├── churn_model.py
├── requirements.txt
├── README.md
├── research_paper.md
└── data/
    └── European_Bank.csv
```

## Methodology and responsible use
The test set is stratified to preserve class proportions. The app compares models using a fixed 0.5 threshold and ranks candidate models by ROC-AUC. This is a reproducible baseline, not proof that the model is production-ready. Tune thresholds using business costs, assess calibration and subgroup performance, and validate on temporally later data before real deployment. What-if changes are model associations, not causal estimates.

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload the project files.
2. Ensure `app.py`, `churn_model.py`, `requirements.txt`, and `data/European_Bank.csv` are in the repository.
3. Open Streamlit Community Cloud, connect GitHub, select the repository and `app.py`, then deploy.
4. Test every page using the public URL before submitting it.

## Submission checklist
- [ ] GitHub repository URL
- [ ] Research paper link (publish the final paper in the repository or a shareable document)
- [ ] Deployed Streamlit app URL
- [ ] Project feedback video URL (upload an explanation/demo to a video host)

## Limitations
The supplied file's provenance and representativeness are not established by this project. Predictions may reflect historical correlations and may not generalize. Do not use model scores as the sole basis for denying services or making consequential financial decisions.
