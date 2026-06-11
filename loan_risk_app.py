"""
Access Bank Loan Risk Prediction App
Simple app to predict if a loan applicant is high-risk or low-risk
"""

import pickle
import streamlit as st
import numpy as np

# Load the trained Random Forest model
@st.cache_resource
def load_model():
    with open('module-04-introduction-to-machine-learning/week-06/module-demo/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# App title
st.title("🏦 Loan Risk Predictor")
st.write("Enter applicant information to predict loan risk")

# Input form
st.subheader("Applicant Information")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income (NGN)", min_value=100000, value=500000, step=50000)
loan_amount = st.number_input("Loan Amount (NGN)", min_value=50000, value=1000000, step=100000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
employment_years = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
num_dependants = st.number_input("Number of Dependants", min_value=0, max_value=10, value=2)

# Calculate debt ratio
debt_ratio = loan_amount / income

st.info(f"Debt-to-Income Ratio: {debt_ratio:.2f}")

# Predict button
if st.button("Predict Risk", type="primary"):
    # Prepare input for model
    # Order: age, income, loan_amount, credit_score, employment_years, num_dependants, debt_ratio
    features = np.array([[age, income, loan_amount, credit_score,
                         employment_years, num_dependants, debt_ratio]])

    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Probability of high-risk

    # Show result
    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("✅ LOW RISK - Applicant is likely to repay the loan")
    else:
        st.error("⚠️ HIGH RISK - Applicant may have difficulty repaying")

    st.metric("High-Risk Probability", f"{probability:.1%}")

    # Simple explanation
    if probability < 0.3:
        st.write("💡 Very safe to approve this loan")
    elif probability < 0.7:
        st.write("💡 Moderate risk - review carefully")
    else:
        st.write("💡 High risk - consider rejecting or require collateral")
