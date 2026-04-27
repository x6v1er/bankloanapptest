import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("my_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")

st.markdown("Enter applicant details below:")

# -----------------------------
# Numeric inputs
# -----------------------------
Requested_Loan_Amount = st.number_input("Requested Loan Amount")
FICO_score = st.number_input("FICO Score")
Monthly_Gross_Income = st.number_input("Monthly Gross Income")
Monthly_Housing_Payment = st.number_input("Monthly Housing Payment")

Ever_Bankrupt_or_Foreclose = st.selectbox(
    "Ever Bankrupt or Foreclosed?",
    ["No", "Yes"]
)

# -----------------------------
# Categorical inputs (RAW ONLY)
# -----------------------------
Reason = st.selectbox(
    "Reason",
    [
        "credit_card_refinancing",
        "debt_conslidation",
        "home_improvement",
        "major_purchase",
        "other"
    ]
)

Employment_Status = st.selectbox(
    "Employment Status",
    ["full_time", "part_time", "unemployed"]
)

Employment_Sector = st.selectbox(
    "Employment Sector",
    [
        "consumer_discretionary",
        "consumer_staples",
        "energy",
        "financials",
        "health_care",
        "industrials",
        "information_technology",
        "materials",
        "real_estate",
        "utilities"
    ]
)

Lender = st.selectbox("Lender", ["A", "B", "C"])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Outcome"):

    # Build RAW input dataframe (NO manual dummy columns)
    input_df = pd.DataFrame([{
        "Requested_Loan_Amount": Requested_Loan_Amount,
        "FICO_score": FICO_score,
        "Monthly_Gross_Income": Monthly_Gross_Income,
        "Monthly_Housing_Payment": Monthly_Housing_Payment,
        "Ever_Bankrupt_or_Foreclose": 1 if Ever_Bankrupt_or_Foreclose == "Yes" else 0,
        "Reason": Reason,
        "Employment_Status": Employment_Status,
        "Employment_Sector": Employment_Sector,
        "Lender": Lender
    }])

    # ONE-HOT ENCODING (this replaces ALL manual feature columns)
    input_df = pd.get_dummies(input_df)

    # ALIGN WITH TRAINING FEATURES (CRITICAL FIX)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    prediction = model.predict(input_df)

    # Output
    st.subheader("Result")

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
        st.balloons()
    else:
        st.error("Loan Not Approved ❌")

    st.write("Prediction value:", prediction[0])