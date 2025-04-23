# data_pipeline/transform.py
import pandas as pd

def add_features(df):
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df.dropna(inplace=True)
    return df

def save_transformed_data(df, path="data/processed/nifty50_5min_features.csv"):
    df.to_csv(path, index=False)
    print(f"[INFO] Transformed data saved to {path}")
