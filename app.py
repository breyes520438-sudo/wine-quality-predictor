import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🍷 Wine Quality Prediction App")
st.write("Enter the chemical properties of the wine sample to predict if it is **Good** (quality ≥ 7) or **Not Good**.")

# Define input fields for features
feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"{feature}", value=0.0, format="%.3f")

# Prediction button
if st.button("Predict Quality"):
    input_df = pd.DataFrame([inputs])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"✅ This wine is predicted to be GOOD quality with confidence {probability:.2f}")
    else:
        st.error(f"❌ This wine is predicted to be NOT GOOD quality with confidence {probability:.2f}")
