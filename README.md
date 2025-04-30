# Nifty50 Stock Price Prediction & Monitoring App (MLOps Project)

This project is an end-to-end AI application that predicts Nifty50 stock prices using 5-minute interval data. It is designed with MLOps principles in mind — focusing on automation, reproducibility, monitoring, and modular pipelines.

---

## Features
- Fetches 5-minute interval stock data for `NIFTYBEES.NS` via `yfinance`
- Validates and transforms data with rolling averages
- 🛠Modular pipeline structure (Ingest → Validate → Transform)
- Version-controlled data pipeline with DVC
- Local storage of raw and processed datasets
- Ready for integration with MLflow, Docker, Prometheus & Grafana


## Architecture 
```
+-------------------------+       +---------------------+      +-------------------+
|   Frontend Web App     | <---> | Backend (Model API) | <--> |  Model Inference   |
|     (Streamlit)        |       |     (FastAPI)       |      |(MLflow, .pth, .pkl)|
+-------------------------+       +---------------------+      +-------------------+
                                           |
                                           v
                                 +----------------------+
                                 | Prometheus Monitoring|
                                 +----------------------+
                                           |
                                           v
                                   +------------------+
                                   | Grafana Dashboard|
                                   +------------------+  
All Dockerized together:
- Frontend
- Backend
- Prometheus
- Metrics Exporter
- Grafana

```


## Folder Structure
```
.
├── data/
│   ├── raw/
│   │   └── nifty50_5min.csv
│   └── processed/
│       └── nifty50_5min_features.csv

├── model/                      
│   ├── best_model.pth

├── data_pipeline/
│   ├── ingest.py
│   ├── validate.py
│   ├── transform.py
│   └── pipeline.py

ml_pipeline/
|    ├── model_architecture.py    # model class only
|    └── train.py                 # train script

├── app/                      # FastAPI Inference API
│   ├── main.py               # FastAPI app (entry point)
│   ├── model.py              # Load model and predict
│   ├── schemas.py            # Request/Response data models
│
├── system-metric-exporter
│   ├── Dockerfile
│   ├── metrics_exporter.py
│   └── requirements.txt
├── dvc.yaml
├── .dvc/
├── .git/
├── docker-compose.yml
├── dockerfile
└── requirements.txt
```


## Setup Instructions

### Clone the Repo and Set Up The Environment

```bash
git clone https://github.com/amar-at-iitm/End_to_end_AI_application
cd nifty50-mlops
```
````bash
python -m venv venv
source venv/bin/activate
````
```bash
pip install -r requirements.txt
```
