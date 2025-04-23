# data_pipeline/validate.py

def validate_data(df):
    assert not df.empty, "DataFrame is empty!"
    assert df.isnull().sum().sum() == 0, "Data contains missing values!"
    assert "Datetime" in df.columns or "Date" in df.columns, "Datetime column missing!"
    print("[INFO] Data validation passed")
