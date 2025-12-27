import streamlit as st
import requests
import os

st.set_page_config(page_title="Fraud Detection", layout="centered")

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Fraud Detection App")
st.write("Enter feature values for prediction")

values = st.text_input(
    "Feature values (comma separated)",
    "0.1, 1.2, 3.4, 5.6"
)

if st.button("Predict"):
    try:
        payload = {
            "values": [float(v.strip()) for v in values.split(",")]
        }

        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        result = response.json()

        st.success(f"Prediction: {result['prediction']}")
        st.write("Fraud Probability:", result["proba_fraud"])

    except Exception as e:
        st.error(f"Error: {e}")

        if len(payload["values"]) != 30:
    st.error("Model expects exactly 30 features")
    st.stop()

