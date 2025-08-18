import streamlit as st
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Load model and preprocessing tools
gb_gridd = joblib.load("loan_data.pkl")        # GradientBoostingClassifier
scaler = joblib.load("scaler.pkl")               # StandardScaler

# Load encoders
le1 = joblib.load("le1.pkl")  # person_gender
le2 = joblib.load("le2.pkl")  # person_education
le3 = joblib.load("le3.pkl")  # person_home_ownership
le4 = joblib.load("le4.pkl")  # loan_intent
le5 = joblib.load("le5.pkl")  # previous_loan_defaults_on_file

# UI Title
st.title("💸 Loan Default Prediction App")
st.markdown("Fill in the details below to predict the likelihood of **loan repayment**.")

# Input Fields
person_age = st.number_input("Age", min_value=18, max_value=100, step=1)
person_gender = st.selectbox("Gender", le1.classes_.tolist())
person_education = st.selectbox("Education Level", le2.classes_.tolist())
person_income = st.number_input("Annual Income ($)", min_value=1000.0, step=100.0)
person_emp_exp = st.number_input("Years of Employment", min_value=0, max_value=100, step=1)
person_home_ownership = st.selectbox("Home Ownership", le3.classes_.tolist())
loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, step=100.0)
loan_intent = st.selectbox("Loan Purpose", le4.classes_.tolist())
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, step=0.1)
loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, step=0.01)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, step=1)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Default", le5.classes_.tolist())

# Encode categorical inputs
try:
    gender_encoded = le1.transform([person_gender])[0]
    education_encoded = le2.transform([person_education])[0]
    home_encoded = le3.transform([person_home_ownership])[0]
    intent_encoded = le4.transform([loan_intent])[0]
    default_encoded = le5.transform([previous_loan_defaults_on_file])[0]
except Exception as e:
    st.error(f"Encoding Error: {e}")
    st.stop()

# Combine features
input_data = [[
    person_age, gender_encoded, education_encoded, person_income, person_emp_exp,
    home_encoded, loan_amnt, intent_encoded, loan_int_rate, loan_percent_income,
    cb_person_cred_hist_length, credit_score, default_encoded
]]

# Scale the input
try:
    input_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Scaling Error: {e}")
    st.stop()

# Predict
if st.button("🔍 Predict Loan Status"):
    prediction = gb_gridd.predict(input_scaled)[0]
    probability = gb_gridd.predict_proba(input_scaled)[0][1]  # probability of class 1 (repay)

    if prediction == 1:
        st.success("✅ The applicant is **likely to repay** the loan.")
    else:
        st.error("❌ The applicant is **likely to default** on the loan.")

    # st.markdown(f"**Confidence Score:** {probability:.2%}")