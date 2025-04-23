# data_pipeline/pipeline.py

from ingest import download_intraday_data, save_raw_data
from validate import validate_data
from transform import add_features, save_transformed_data

def run_pipeline():
    print("[PIPELINE] Starting data pipeline...")
    
    df = download_intraday_data()
    save_raw_data(df)
    
    validate_data(df)
    
    df_transformed = add_features(df)
    save_transformed_data(df_transformed)

    print("[PIPELINE] Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
