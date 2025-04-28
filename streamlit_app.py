import streamlit as st
import pandas as pd
import requests
import subprocess
import os

# Constants
DATA_PATH = "data/processed/nifty50_5min_features.csv"
SEQ_LENGTH = 30
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# Initialize session state variables
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# Title
st.title("Nifty50 Stock Price Prediction App")

# --- Load Data ---
try:
    df = pd.read_csv(DATA_PATH)

    st.subheader("Uploaded Data Preview (Full Columns):")
    st.dataframe(df.tail(50))

    # --- Predict Section ---
    st.header("Predict Next Close Price")

    if "Close" in df.columns:
        if len(df) >= SEQ_LENGTH:
            input_data = df["Close"].tail(SEQ_LENGTH).values.tolist()
            input_data = [[value] for value in input_data]  # wrap each value

            if st.button("Predict Next Close Price"):
                payload = {"data": input_data}

                try:
                    response = requests.post(FASTAPI_URL, json=payload)
                    response.raise_for_status()
                    prediction = response.json()["prediction"]

                    st.success(f"Predicted Next Close Price: ₹ {prediction:.2f}")

                    # Mark prediction as made
                    st.session_state.prediction_made = True
                    st.session_state.feedback = None

                except Exception as e:
                    st.error(f"Failed to fetch prediction. Error: {e}")
        else:
            st.warning(f"Need at least {SEQ_LENGTH} rows to predict! (Currently have {len(df)} rows)")
    else:
        st.error(f"Uploaded CSV must contain a 'Close' column for prediction!")

except FileNotFoundError:
    st.error("Processed data file not found. Please upload data first.")
except Exception as e:
    st.error(f"An error occurred: {e}")

# --- Feedback Section ---
st.header("Feedback on Prediction")

# Define functions
def mark_satisfied():
    st.session_state.feedback = "satisfied"
    st.session_state.prediction_made = False

def mark_not_satisfied():
    st.session_state.feedback = "not_satisfied"
    st.session_state.prediction_made = False

def retrain_model():
    with st.spinner("Retraining model... Please wait"):
        try:
            retrain_result = subprocess.run(
                ["python", "ml_pipeline/train.py"],
                capture_output=True,
                text=True,
                check=True
            )
            st.success("Model retrained successfully!")
            st.code(retrain_result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"Error during model retraining: {e}")
            st.code(e.stderr)

# Feedback buttons
col1, col2 = st.columns(2)
with col1:
    st.button("Satisfied", on_click=mark_satisfied)
with col2:
    st.button("Not Satisfied", on_click=mark_not_satisfied)

# Show feedback based on state
if st.session_state.feedback == "satisfied" and not st.session_state.prediction_made:
    st.success("Thank you for your feedback! 🙏")

elif st.session_state.feedback == "not_satisfied" and not st.session_state.prediction_made:
    st.warning("You chose 'Not Satisfied'.")
    
    confirm_retrain = st.checkbox("⚡ I confirm I want to retrain the model")
    if confirm_retrain:
        if st.button("Start Retraining"):
            retrain_model()

# --- Data Update Section ---
st.header("Update Data")

if st.button("Update Data Pipeline"):
    with st.spinner("Running DVC pipeline and pushing data..."):
        try:
            # Run dvc repro
            repro_result = subprocess.run(
                ["dvc", "repro"],
                capture_output=True,
                text=True,
                check=True
            )
            st.success("DVC Repro successful!")
            st.code(repro_result.stdout)

            # Run dvc push
            push_result = subprocess.run(
                ["dvc", "push", "-r", "remote_ai_app"],
                capture_output=True,
                text=True,
                check=True
            )
            st.success("DVC Push successful!")
            st.code(push_result.stdout)

        except subprocess.CalledProcessError as e:
            st.error(f"Error during DVC operations: {e}")
            st.code(e.stderr)
