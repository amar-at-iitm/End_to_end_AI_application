# data_pipeline/ingest.py
import yfinance as yf
import pandas as pd
import os


def download_intraday_data(interval="5m", period="60d"):
    ticker = "^NSEI"
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df.reset_index(inplace=True)
    return df


def save_raw_data(df, path="data/raw/nifty50_5min.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Raw NIFTY 50 index data saved to {path}")

    # Re-read and clean non-numeric rows
    df_clean = pd.read_csv(path)
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    for col in numeric_cols:
        df_clean = df_clean[pd.to_numeric(df_clean[col], errors="coerce").notna()]

    df_clean.to_csv(path, index=False)
    print(f"[INFO] Cleaned non-numeric rows and re-saved to {path}")

