import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set page layout and title
st.set_page_config(
    page_title="Loan Approval Predictor", page_icon="💳", layout="centered"
)

# Custom CSS for UI styling and colors
st.markdown(
    """
    <style>
    /* Main app background color */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* Title card container */
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-card h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2rem;
    }
    .header-card p {
        color: #e0e0e0;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Style for the predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 12px 20px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="header-card">
        <h1>💳 Loan Eligibility Predictor</h1>
        <p>Enter applicant details below to assess approval probability</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Load the pre-trained Logistic Regression Model
@st.cache_resource
def load_model():
    with open("logistic_regression_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "❌ `logistic_regression_model.pkl` not found. Please make sure the file is in the same directory as this script."
    )
    st.stop()

# Input Form Controls using standard Streamlit widgets
st.subheader("📋 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income ($)",
        min_value=0,
        max_value=2000000,
        value=350000,
        step=5000,
        help="Total annual income in USD",
    )

    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700,
        help="FICO Credit Score range (300-850)",
    )

with col2:
    employment_years = st.number_input(
        "Employment Duration (Years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Number of consecutive years employed",
    )

    debt_ratio = st.slider(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.01,
        help="Ratio of total monthly debt payments to gross monthly income",
    )

st.markdown("---")

# Prediction Trigger
if st.button("Evaluate Application"):
    # Organize input array matching feature order: income, credit_score, employment_years, debt_ratio
    features = np.array(
        [[income, credit_score, employment_years, debt_ratio]]
    )

    # Perform Prediction
    prediction = model.predict(features)[0]

    # Calculate Probability if supported by model
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

    st.subheader("📊 Decision Result")

    if prediction == 1:
        st.success("🎉 **Application Approved!**")
        if probabilities is not None:
            st.metric(
                label="Approval Probability",
                value=f"{probabilities[1] * 100:.1f}%",
            )
            st.progress(float(probabilities[1]))
    else:
        st.error("❌ **Application Rejected**")
        if probabilities is not None:
            st.metric(
                label="Approval Probability",
                value=f"{probabilities[1] * 100:.1f}%",
            )
            st.progress(float(probabilities[1]))