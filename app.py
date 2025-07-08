import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# === Page Config ===
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("💼 Customer Churn Prediction App")

st.markdown("""
This app predicts whether a bank customer will **churn** or not, based on their profile.  
It uses a trained **Artificial Neural Network** model built with Keras and preprocessed features.
""")

# === Load Model and Scaler ===
model = load_model("model/churn_ann_model.h5")
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

le = LabelEncoder()
le.fit(['Female', 'Male'])

st.markdown("### 🧾 Enter Customer Details Below")

# === Input Fields (Grouped using Columns) ===
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
    age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
    balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=100.0)
    products = st.selectbox("Number of Products", [1, 2, 3, 4])
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

with col2:
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3, step=1)
    salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=100.0)
    has_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    active_member = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
    geo = st.selectbox("Geography", ["France", "Germany", "Spain"])

# === Submit Button ===
st.markdown("---")
if st.button("🔍 Predict Churn", use_container_width=True):
    # Build input
    data = {
        'CreditScore': credit_score,
        'Gender': le.transform([gender])[0],
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': products,
        'HasCrCard': 1 if has_card == "Yes" else 0,
        'IsActiveMember': 1 if active_member == "Yes" else 0,
        'EstimatedSalary': salary,
        'Geography_France': 1 if geo == "France" else 0,
        'Geography_Germany': 1 if geo == "Germany" else 0,
        'Geography_Spain': 1 if geo == "Spain" else 0
    }

    df = pd.DataFrame([data])
    final_order = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                   'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                   'Geography_France', 'Geography_Germany', 'Geography_Spain']
    df = df[final_order]

    scaled = scaler.transform(df)
    prob = model.predict(scaled)[0][0]
    churn = prob > 0.5

    # === Result Display ===
    st.markdown("### 🔎 Prediction Result")
    st.metric(label="Churn Probability", value=f"{prob:.2%}")

    if churn:
        st.error("⚠️ Prediction: Customer is likely to churn.")
    else:
        st.success("✅ Prediction: Customer will stay.")

