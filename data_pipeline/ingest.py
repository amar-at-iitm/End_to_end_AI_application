# data_pipeline/ingest.py
import yfinance as yf
import pandas as pd

def download_intraday_data(ticker="NIFTYBEES.NS", interval="5m", period="60d"):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df.reset_index(inplace=True)
    return df

def save_raw_data(df, path="data/raw/nifty50_5min.csv"):
    df.to_csv(path, index=False)
    print(f"[INFO] Raw data saved to {path}")

    # remove non-numeric rows after saving
    df_clean = pd.read_csv(path)

    # Select only numeric rows
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    for col in numeric_cols:
        df_clean = df_clean[pd.to_numeric(df_clean[col], errors="coerce").notna()]

    # Save cleaned version
    df_clean.to_csv(path, index=False)
    print(f"[INFO] Cleaned non-numeric rows and re-saved to {path}")
