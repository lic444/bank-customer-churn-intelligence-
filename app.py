from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve
from churn_model import train_and_evaluate, build_pipeline, prepare_X

st.set_page_config(page_title="Bank Churn Intelligence", page_icon="🏦", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "European_Bank.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models(data_json):
    df = pd.read_json(StringIO(data_json), orient="split")
    return train_and_evaluate(df)

df = load_data()
results, best_name, best, all_models, split = load_models(df.to_json(orient="split"))
X_train, X_test, y_train, y_test = split
model = best["pipeline"]

st.title("🏦 Bank Customer Churn Intelligence")
st.caption("Predictive modeling • Customer risk scoring • Explainable retention insights")
st.info("Decision-support prototype using a historical dataset. Scores are estimates, not guarantees; validate fairness, calibration and business impact before production use.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose module", ["Executive Overview", "Explore Customers", "Churn Risk Calculator", "What-if Simulator", "Model Performance", "Feature Insights"])
    st.divider()
    st.caption(
    "Risk bands: Low < 30%, Medium 30%–59.99%, High >= 60%. "
    "The 30% boundary was selected using training cross-validation "
    "to maximize F1-score. These bands are not yet validated against "
    "business retention costs."
)
churn_rate = df["Exited"].mean() * 100
if page == "Executive Overview":
    st.subheader("Executive Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Observed churn rate", f"{churn_rate:.2f}%")
    c3.metric("Churned customers", f"{int(df.Exited.sum()):,}")
    c4.metric("Selected model", best_name)
    left,right = st.columns(2)
    with left:
        counts = df["Exited"].map({0:"Retained",1:"Churned"}).value_counts().rename_axis("Customer status").reset_index(name="Customers")
        st.plotly_chart(px.pie(counts, names="Customer status", values="Customers", hole=.55, title="Customer churn distribution"), use_container_width=True)
    with right:
        geo = df.groupby("Geography", as_index=False)["Exited"].mean()
        geo["Churn rate (%)"] = geo["Exited"]*100
        st.plotly_chart(px.bar(geo, x="Geography", y="Churn rate (%)", title="Observed churn rate by geography", text_auto=".1f"), use_container_width=True)
    st.markdown("#### Initial business questions")
    st.markdown("- Which customer profiles receive high predicted churn probabilities?\n- How do age, activity and product count relate to churn?\n- Which model best separates churners from retained customers?\n- Which retention scenarios are worth testing with controlled experiments?")
    st.caption("Observed associations are not proof of causation. Use controlled experiments to estimate the effect of retention offers.")

elif page == "Explore Customers":
    st.subheader("Customer Explorer")
    c1,c2,c3 = st.columns(3)
    geography = c1.multiselect("Geography", sorted(df.Geography.unique()), default=sorted(df.Geography.unique()))
    status = c2.selectbox("Customer status", ["All", "Churned", "Retained"])
    age_range = c3.slider("Age range", int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())))
    filtered = df[df.Geography.isin(geography) & df.Age.between(*age_range)]
    if status != "All": filtered = filtered[filtered.Exited == (1 if status == "Churned" else 0)]
    st.metric("Matching customers", f"{len(filtered):,}")
    st.dataframe(filtered.drop(columns=["Surname"]), use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_customers.csv", "text/csv")
    left,right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(df, x="Age", color=df["Exited"].map({0:"Retained",1:"Churned"}), barmode="overlay", opacity=.7, title="Age distribution by churn status", labels={"color":"Status"}), use_container_width=True)
    with right:
        st.plotly_chart(px.box(df, x=df["Exited"].map({0:"Retained",1:"Churned"}), y="NumOfProducts", title="Products held by churn status", labels={"x":"Status"}), use_container_width=True)

elif page == "Churn Risk Calculator":
    st.subheader("Individual Customer Risk Calculator")
    st.write("Enter customer details to estimate churn probability. Use realistic values and treat the output as decision support.")
    with st.form("risk_form"):
        a,b,c = st.columns(3)
        credit = a.slider("Credit score", 300, 900, 650)
        geography = b.selectbox("Geography", sorted(df.Geography.unique()))
        gender = c.selectbox("Gender", sorted(df.Gender.unique()))
        age = a.slider("Age", 18, 100, 40)
        tenure = b.slider("Tenure (years)", 0, 10, 5)
        balance = c.number_input("Balance", min_value=0.0, value=75000.0, step=5000.0)
        products = a.selectbox("Number of products", [1,2,3,4], index=1)
        card = b.selectbox("Has credit card?", ["Yes", "No"])
        active = c.selectbox("Active member?", ["Yes", "No"])
        salary = a.number_input("Estimated annual salary", min_value=0.0, value=80000.0, step=5000.0)
        submitted = st.form_submit_button("Calculate churn risk", type="primary")
    if submitted:
        customer = pd.DataFrame([{"CreditScore":credit,"Geography":geography,"Gender":gender,"Age":age,"Tenure":tenure,"Balance":balance,"NumOfProducts":products,"HasCrCard":1 if card=="Yes" else 0,"IsActiveMember":1 if active=="Yes" else 0,"EstimatedSalary":salary}])
        probability = float(model.predict_proba(customer)[0,1])
        band = "High" if probability >= .60 else "Medium" if probability >= .30 else "Low"
        c1,c2 = st.columns([1,2])
        c1.metric("Estimated churn probability", f"{probability:.1%}")
        c1.metric("Illustrative risk band", band)
        with c2:
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=probability*100, number={"suffix":"%"}, title={"text":"Churn probability"}, gauge={"axis":{"range":[0,100]},"steps":[{"range":[0,30]},{"range":[30,60]},{"range":[60,100]}],"threshold":{"line":{"width":4},"value":probability*100}})), use_container_width=True)
        st.warning("Risk bands are illustrative thresholds, not validated business cut-offs. Check calibration and retention costs before operational use.")

elif page == "What-if Simulator":
    st.subheader("What-if Scenario Simulator")
    st.write("Explore how model scores change when customer profile inputs change. These are model associations, not causal treatment effects.")
    customer_id = st.selectbox("Choose a sample customer", df.index, format_func=lambda i: f"Row {i+1} — customer {df.loc[i, 'CustomerId']}")
    row = df.loc[[customer_id]]
    base = prepare_X(row)
    scenario = base.copy()
    c1,c2,c3 = st.columns(3)
    active = c1.selectbox("Scenario: active member", [int(row.iloc[0].IsActiveMember), 1-int(row.iloc[0].IsActiveMember)], format_func=lambda x:"Active" if x else "Inactive")
    products = c2.selectbox("Scenario: number of products", [1,2,3,4], index=int(row.iloc[0].NumOfProducts)-1)
    tenure = c3.slider("Scenario: tenure", 0, 10, int(row.iloc[0].Tenure))
    scenario.loc[:,"IsActiveMember"] = active
    scenario.loc[:,"NumOfProducts"] = products
    scenario.loc[:,"Tenure"] = tenure
    p0 = float(model.predict_proba(base)[0,1]); p1 = float(model.predict_proba(scenario)[0,1])
    c1,c2,c3 = st.columns(3)
    c1.metric("Baseline risk", f"{p0:.1%}")
    c2.metric("Scenario risk", f"{p1:.1%}")
    c3.metric("Score difference", f"{(p1-p0)*100:+.1f} pp")
    st.caption("Changing a field in the simulator does not prove that an intervention will cause churn risk to change by this amount.")

elif page == "Model Performance":
    st.subheader("Model Performance Comparison")
    st.dataframe(results.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1-score","ROC-AUC"]}), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(results.melt(id_vars="Model", value_vars=["Precision","Recall","F1-score","ROC-AUC"], var_name="Metric", value_name="Score"), x="Model", y="Score", color="Metric", barmode="group", title="Evaluation metrics at the 0.30 threshold"), use_container_width=True)
    ytrue = best["y_test"]
    cm = confusion_matrix(ytrue, best["predictions"])
    left,right = st.columns(2)
    with left:
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted retained","Predicted churn"], y=["Actual retained","Actual churn"], labels={"x":"Prediction","y":"Actual","color":"Count"}, title=f"Confusion matrix — {best_name}"), use_container_width=True)
    with right:
        fpr,tpr,_ = roc_curve(ytrue, best["probabilities"])
        st.plotly_chart(go.Figure([go.Scatter(x=fpr,y=tpr,mode="lines",name=best_name),go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line={"dash":"dash"})]).update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate"), use_container_width=True)
    st.caption("Metrics use a stratified 80/20 holdout split (random_state=42) and a 0.30 classification threshold. That threshold was selected using training cross-validation for Gradient Boosting; it is a common comparison threshold, not a separately optimized threshold for every model. Choose an operational threshold based on retention capacity and the costs of false positives and false negatives.")

else:
    st.subheader("Feature Insights")
    st.write(
        "Permutation importance measures how much the model's ROC-AUC changes "
        "when one input feature is randomly shuffled. A larger positive value "
        "means the model relies more on that feature for ranking churn risk."
    )

    from sklearn.inspection import permutation_importance

    with st.spinner("Calculating permutation importance on the held-out test set..."):
        imp = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1,
        )

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": imp.importances_mean,
        "Std": imp.importances_std,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    top_features = importance.head(12).sort_values("Importance", ascending=True)

    st.plotly_chart(
        px.bar(
            top_features,
            x="Importance",
            y="Feature",
            error_x="Std",
            orientation="h",
            title="Top 12 Features by Permutation Importance",
            labels={
                "Importance": "Mean decrease in ROC-AUC",
                "Feature": "Customer feature",
            },
        ),
        use_container_width=True,
    )

    st.subheader("Importance values")
    st.dataframe(
        importance.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Error bars show variation across repeated shuffles. Correlated features "
        "can share importance; near-zero or negative values do not prove a feature "
        "has no real-world relationship with churn. Importance is predictive, "
        "not causal, and does not establish fairness."
    )

st.divider()
st.caption("Built for educational and portfolio use • Do not use for real customer decisions without validation, governance, privacy review and monitoring.")
