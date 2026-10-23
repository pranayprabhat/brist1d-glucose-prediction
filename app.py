
import streamlit as st
import pandas as pd
import joblib
import gdown
from pathlib import Path

st.set_page_config(
    page_title="Blood Glucose Prediction",
    page_icon="🩸"
)

st.title("Blood Glucose Prediction")
st.write("Enter the measurements below to predict blood glucose one hour ahead.")

# Model file location
model_path = Path(__file__).parent / "glucose_model.pkl"

# Download the model from Google Drive if it is not available locally
if not model_path.exists():
    file_id = "1LU4pe3Z-zrRaR1BNQW5duxNRCBQi96B8"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner("Downloading trained model. Please wait..."):
        result = gdown.download(
            download_url,
            str(model_path),
            quiet=False
        )

    if result is None or not model_path.exists():
        st.error("Model download failed. Check the Google Drive sharing settings.")
        st.stop()

# Load the trained model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

st.subheader("Enter your values")

with st.form("prediction_form"):
    bg_5 = st.number_input("Glucose 5 minutes ago", min_value=0.0, value=6.2)
    bg_10 = st.number_input("Glucose 10 minutes ago", min_value=0.0, value=6.0)
    bg_15 = st.number_input("Glucose 15 minutes ago", min_value=0.0, value=5.9)
    bg_20 = st.number_input("Glucose 20 minutes ago", min_value=0.0, value=5.8)
    bg_25 = st.number_input("Glucose 25 minutes ago", min_value=0.0, value=5.7)
    bg_30 = st.number_input("Glucose 30 minutes ago", min_value=0.0, value=5.6)

    insulin = st.number_input("Insulin", min_value=0.0, value=0.0)
    carbs = st.number_input("Carbohydrates", min_value=0.0, value=10.0)
    hr = st.number_input("Heart rate", min_value=0.0, value=78.0)
    steps = st.number_input("Steps", min_value=0.0, value=120.0)
    cals = st.number_input("Calories", min_value=0.0, value=2.5)

    submitted = st.form_submit_button("Predict future glucose")

if submitted:
    input_data = pd.DataFrame([{
        "bg-0:05": bg_5,
        "bg-0:10": bg_10,
        "bg-0:15": bg_15,
        "bg-0:20": bg_20,
        "bg-0:25": bg_25,
        "bg-0:30": bg_30,
        "insulin-0:05": insulin,
        "carbs-0:05": carbs,
        "hr-0:05": hr,
        "steps-0:05": steps,
        "cals-0:05": cals
    }])

    prediction = float(model.predict(input_data)[0])

    st.subheader("Prediction result")
    st.metric(
        "Predicted blood glucose after one hour",
        f"{prediction:.2f}"
    )

    st.warning(
        "This is an experimental ML prediction, not medical advice. "
        "Do not use it to make insulin or treatment decisions."
    )
