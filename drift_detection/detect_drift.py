import pandas as pd
import json
from scipy.stats import ks_2samp
from datetime import datetime

def load_training_data(path="data/processed/nifty50_5min_features.csv"):
    df = pd.read_csv(path)
    return df['Close'].values

def load_recent_data(path="data/raw/nifty50_5min.csv", last_n=100):
    df = pd.read_csv(path)
    return df['Close'].values[-last_n:]  # last N entries

def detect_drift(train_data, recent_data, threshold=0.05):
    stat, p_value = ks_2samp(train_data, recent_data)
    drift = bool(p_value < threshold)  # ensure native Python bool
    return {
        "drift_detected": drift,
        "p_value": float(p_value),      # convert to native float
        "statistic": float(stat),
        "timestamp": datetime.now().isoformat()
    }


def save_report(report, path="drift_detection/drift_report.json"):
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    train = load_training_data()
    recent = load_recent_data()
    report = detect_drift(train, recent)
    save_report(report)
    print("Drift report saved:", report)
