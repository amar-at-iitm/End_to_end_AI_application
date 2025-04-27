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

    # remove non-numeric rows after saving
    df_clean = pd.read_csv(path)

    # Select only numeric rows (ignore rows with strings etc.)
    numeric_cols = ["Close", "High", "Low", "Open", "Volume", "MA_20", "MA_50"]
    for col in numeric_cols:
        df_clean = df_clean[pd.to_numeric(df_clean[col], errors="coerce").notna()]

    # Save cleaned version
    df_clean.to_csv(path, index=False)
    print(f"[INFO] Cleaned non-numeric rows and re-saved to {path}")

