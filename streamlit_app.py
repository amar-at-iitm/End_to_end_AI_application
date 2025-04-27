import streamlit as st
import pandas as pd
import requests

# Title
st.title(" Nifty50 Stock Price Prediction App")

# Sidebar to upload your CSV
st.sidebar.header("Upload Processed Data")
uploaded_file = st.sidebar.file_uploader("Upload your processed nifty50_5min_features.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview:")
    st.write(df.tail(50))  # show last 50 rows

    # Prepare input
    feature_columns = ["Open", "High", "Low", "Close", "Volume"]

    if all(col in df.columns for col in feature_columns):
        SEQ_LENGTH = 30

        if len(df) >= SEQ_LENGTH:
            input_data = df[feature_columns].tail(SEQ_LENGTH).values.tolist()

            # Button to predict
            if st.button("Predict Next Close Price"):
                url = "http://127.0.0.1:8000/predict"
                payload = {"data": input_data}

                try:
                    response = requests.post(url, json=payload)
                    prediction = response.json()["prediction"]

                    st.success(f"Predicted Next Close Price: ₹ {prediction}")

                except Exception as e:
                    st.error(f" Failed to fetch prediction. Error: {e}")

        else:
            st.warning(f"Need at least {SEQ_LENGTH} rows to predict!")
    else:
        st.error(f"Uploaded CSV must contain columns: {feature_columns}")

else:
    st.info(" Upload your processed Nifty50 data to begin prediction.")
