import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import requests
import subprocess
import smtplib
from email.message import EmailMessage
from datetime import datetime
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

    # st.subheader("Uploaded Data Preview (Full Columns):")
    # st.dataframe(df.tail(50))
    st.subheader(" Data Preview ")

    if 'Close' in df.columns and 'Datetime' in df.columns:
        # Convert 'Datetime' column to datetime objects
        df['Datetime'] = pd.to_datetime(df['Datetime'])

        # Get available dates
        available_dates = df['Datetime'].dt.date.unique()
        available_dates = sorted(available_dates, reverse=True)  # Latest first

        # Calendar selector: Default is today if available, else latest available
        default_date = pd.Timestamp.now().date()
        if default_date not in available_dates:
            default_date = available_dates[0]

        selected_date = st.date_input(
            "Choose a date to display the chart:", 
            value=default_date,
            min_value=min(available_dates),
            max_value=max(available_dates)
        )

        # Filter data based on selected date
        selected_data = df[df['Datetime'].dt.date == selected_date]

        if selected_data.empty:
            st.warning(f"No data available for {selected_date}.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=selected_data['Datetime'],
                y=selected_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='deepskyblue')
            ))
            fig.update_layout(
                title=f"Close Price Movement on {selected_date}",
                xaxis_title="Time",
                yaxis_title="Close Price (₹)",
                template="plotly_dark",
                autosize=True,
                xaxis=dict(
                    tickformat='%H:%M'  # Only show time (Hour:Minute)
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Add Refresh Button at the Top ---
        if st.button("🔄 Refresh Data Now"):
            st.rerun()


    else:
        st.warning("Required columns ('Close', 'Datetime') not found in uploaded data!")

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # --- Predict Section ---
    st.header("Predict Next 30 Close Price")

    if "Close" in df.columns:
        if len(df) >= SEQ_LENGTH:
            input_data = df["Close"].tail(SEQ_LENGTH).values.tolist()
            input_data = [[value] for value in input_data]  # wrap each value

            if st.button("Predict Next 30 Close Price"):
                payload = {"data": input_data}

                try:
                    response = requests.post(FASTAPI_URL, json=payload)
                    response.raise_for_status()
                    prediction = response.json()["prediction"]

                    #st.success(f"Predicted Next Close Price: ₹ {prediction:.2f}")        # for single prediction


                    #///////////////////////////////////////////////////////////
                    # Display prediction in a line chart
                    # Create x and y values
                    days = list(range(1, len(prediction) + 1))
                    prices = [round(p, 2) for p in prediction]

                    # Create plotly figure
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=days,
                        y=prices,
                        mode='lines+markers',
                        name='Predicted Close Price',
                        line=dict(color='royalblue', width=2),
                        marker=dict(size=6)
                    ))

                    fig.update_layout(
                        title="Predicted Close Prices for the Next 30 Time Steps",
                        xaxis_title="Time Steps",
                        yaxis_title="Price (₹)",
                        template="plotly_white"
                    )

                    # Display in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                   #///////////////////////////////////////////////////////////////////////

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


#//////////////////////////////////////////////////////////////////////////////////////////////////
# --- Feedback Section ---
st.header("Feedback on Prediction")



# Define functions
def mark_satisfied():
    st.session_state.feedback = "satisfied"
    st.session_state.prediction_made = False

def mark_not_satisfied():
    st.session_state.feedback = "not_satisfied"
    st.session_state.prediction_made = False

# Function to notify developer instead of retraining
def notify_developer(feedback_details="Prediction was not satisfactory."):
    msg = EmailMessage()
    msg.set_content(f"User Feedback - Model Issue\n\nTimestamp: {datetime.now()}\nDetails: {feedback_details}")
    msg["Subject"] = " Feedback: Model prediction unsatisfactory"
    msg["From"] = "abhijit912813@gmail.com"
    msg["To"] = "amar8409849358@gmail.com"

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login("abhijit912813@gmail.com", "cbmghskpqrozbyyy")  # Use app-specific password for Gmail
            smtp.send_message(msg)
        st.success("Feedback sent to the developer. Thank you!")
    except Exception as e:
        st.error(f"Failed to send feedback: {e}")

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
    
    confirm_retrain = st.checkbox("⚡ I confirm I want to send this to the developer")
    if confirm_retrain:
        if st.button("Send Feedback to Developer"):
            notify_developer("User marked the prediction as unsatisfactory. Please review the model output.")

# # --- Data Update Section ---
# st.header("Update Data")

# if st.button("Update Data Pipeline"):
#     with st.spinner("Running data pipeline..."):
#         try:
#             # Run the custom data update script
#             update_result = subprocess.run(
#                 ["python", "data_pipeline/pipeline.py"],
#                 capture_output=True,
#                 text=True,
#                 check=True
#             )
#             st.success("Data pipeline ran successfully!")
#             st.code(update_result.stdout)

#         except subprocess.CalledProcessError as e:
#             st.error("Error during data pipeline execution.")
#             st.code(e.stderr)
