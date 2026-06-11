"""
Access Bank Loan Risk Prediction App - Gradio Version
Simple app to predict if a loan applicant is high-risk or low-risk
"""

import pickle
import gradio as gr
import numpy as np

# Load the trained Random Forest model
with open('module-04-introduction-to-machine-learning/week-06/module-demo/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_loan_risk(age, income, loan_amount, credit_score, employment_years, num_dependants):
    """
    Predict loan risk based on applicant information

    Returns: Risk classification and probability
    """
    # Calculate debt ratio
    debt_ratio = loan_amount / income

    # Prepare input for model
    # Order: age, income, loan_amount, credit_score, employment_years, num_dependants, debt_ratio
    features = np.array([[age, income, loan_amount, credit_score,
                         employment_years, num_dependants, debt_ratio]])

    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Probability of high-risk

    # Create result message
    if prediction == 0:
        risk_status = "✅ LOW RISK"
        message = "Applicant is likely to repay the loan"
        color = "green"
    else:
        risk_status = "⚠️ HIGH RISK"
        message = "Applicant may have difficulty repaying"
        color = "red"

    # Add recommendation
    if probability < 0.3:
        recommendation = "💡 Very safe to approve this loan"
    elif probability < 0.7:
        recommendation = "💡 Moderate risk - review carefully"
    else:
        recommendation = "💡 High risk - consider rejecting or require collateral"

    # Format output
    result = f"""
    <h2 style='color: {color};'>{risk_status}</h2>
    <p><strong>{message}</strong></p>
    <p><strong>High-Risk Probability:</strong> {probability:.1%}</p>
    <p><strong>Debt-to-Income Ratio:</strong> {debt_ratio:.2f}</p>
    <hr>
    <p>{recommendation}</p>
    """

    return result


# Create Gradio interface
with gr.Blocks(title="Loan Risk Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🏦 Loan Risk Predictor")
    gr.Markdown("Enter applicant information to predict loan risk")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Applicant Information")

            age = gr.Number(
                label="Age",
                value=35,
                minimum=18,
                maximum=100
            )

            income = gr.Number(
                label="Annual Income (NGN)",
                value=500000,
                minimum=100000
            )

            loan_amount = gr.Number(
                label="Loan Amount (NGN)",
                value=1000000,
                minimum=50000
            )

            credit_score = gr.Number(
                label="Credit Score",
                value=650,
                minimum=300,
                maximum=850
            )

            employment_years = gr.Number(
                label="Years of Employment",
                value=5,
                minimum=0,
                maximum=50
            )

            num_dependants = gr.Number(
                label="Number of Dependants",
                value=2,
                minimum=0,
                maximum=10
            )

            predict_btn = gr.Button("Predict Risk", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### Prediction Result")
            output = gr.HTML(label="Result")

    # Connect button to prediction function
    predict_btn.click(
        fn=predict_loan_risk,
        inputs=[age, income, loan_amount, credit_score, employment_years, num_dependants],
        outputs=output
    )

    gr.Markdown("""
    ---
    **Model Information:**
    - Algorithm: Random Forest Classifier
    - Training Data: 500 historical loan records
    - Model Accuracy: 93%
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
