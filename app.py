import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Required for loading our custom transformer
from bankinsight.feature_engineering import BankFeatureEngineer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bankinsight_xgboost.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

st.set_page_config(
    page_title="BankInsight AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

threshold = float(metadata["decision_threshold"])
metrics = metadata["test_metrics"]


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🏦 BankInsight AI")
    st.caption("Customer Intelligence Platform")

    st.divider()

    st.markdown("### Model Information")
    st.write("Algorithm: XGBoost")
    st.write("Dataset: UCI Bank Marketing")
    st.write("Input Features: 11")
    st.write("Prediction: Term Deposit Subscription")

    st.divider()

    st.markdown("### Model Performance")
    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
    st.metric(
        "Average Precision",
        f"{metrics['Average Precision']:.4f}"
    )
    st.metric("F1-score", f"{metrics['F1-score']:.4f}")

    st.caption(
        "Metrics are based on the held-out historical test set."
    )


# ---------------- HEADER ----------------

st.title("BankInsight AI")
st.markdown(
    "#### Intelligent Customer Subscription Prediction"
)

st.write(
    "Analyze a customer's historical banking information "
    "and estimate their likelihood of subscribing to "
    "a term deposit."
)

st.info(
    "This is a machine learning research demonstration. "
    "Predictions are not guarantees or approved banking decisions."
)

st.divider()


# ---------------- CUSTOMER FORM ----------------

st.subheader("Customer Profile")

with st.form("prediction_form"):

    st.markdown("#### Personal Information")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=110,
            value=35
        )

    with c2:
        job = st.selectbox(
            "Occupation",
            [
                "management", "technician", "entrepreneur",
                "blue-collar", "unknown", "retired",
                "admin.", "services", "self-employed",
                "unemployed", "housemaid", "student"
            ]
        )

    with c3:
        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )

    with c4:
        education = st.selectbox(
            "Education",
            ["primary", "secondary", "tertiary", "unknown"]
        )

    st.markdown("#### Financial Information")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        balance = st.number_input(
            "Average Yearly Balance (€)",
            min_value=-100000,
            max_value=1000000,
            value=1500,
            step=100
        )

    with c6:
        housing = st.selectbox(
            "Housing Loan",
            ["no", "yes"]
        )

    with c7:
        loan = st.selectbox(
            "Personal Loan",
            ["no", "yes"]
        )

    with c8:
        default = st.selectbox(
            "Credit Default",
            ["no", "yes"]
        )

    st.markdown("#### Previous Campaign History")

    c9, c10, c11 = st.columns(3)

    with c9:
        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            max_value=500,
            value=0,
            step=1
        )

    with c10:
        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=-1,
            max_value=2000,
            value=-1,
            step=1
        )

    with c11:
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "success", "failure", "other"]
        )

    submitted = st.form_submit_button(
        "Analyze Customer",
        type="primary",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if submitted:

    # Validate historical campaign consistency
    if previous == 0 and pdays != -1:
        st.error(
            "When previous contacts are zero, "
            "days since previous contact must be -1."
        )
        st.stop()

    if previous > 0 and pdays == -1:
        st.error(
            "Customers with previous contacts must have "
            "a valid days-since-contact value."
        )
        st.stop()

    if previous == 0 and poutcome != "unknown":
        st.error(
            "Previous campaign outcome must be unknown "
            "when there is no previous contact."
        )
        st.stop()

    customer_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    probability = float(
        model.predict_proba(customer_df)[0, 1]
    )

    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "Model Subscription Score",
            f"{probability * 100:.2f}%"
        )

    with result2:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.2f}%"
        )

    with result3:
        st.metric(
            "Customer Classification",
            "Higher Priority" if prediction else "Lower Priority"
        )

    st.progress(min(max(probability, 0.0), 1.0))

    if prediction:
        st.success(
            "This customer's model score exceeds the "
            "selected decision threshold."
        )
    else:
        st.warning(
            "This customer's model score is below "
            "the selected decision threshold."
        )

    st.caption(
        "The score is an estimate from historical data. "
        "It is not a guaranteed outcome or a causal estimate "
        "of the benefit of contacting this customer."
    )

    with st.expander("View Customer Input Data"):
        st.dataframe(
            customer_df,
            use_container_width=True,
            hide_index=True
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "BankInsight AI | XGBoost Machine Learning Project | "
    "UCI Bank Marketing Dataset"
)
