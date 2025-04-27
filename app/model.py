# app/model.py
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI

from app.schemas import PredictionRequest  # (if needed later)
from ml_pipeline.model_architecture import StockPriceModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Load your trained best model
model = StockPriceModel(input_size= input_size)
model.load_state_dict(torch.load("model/best_model.pth", map_location=torch.device('cpu')))
model.eval()  # VERY IMPORTANT to set eval mode

def predict(input_tensor):
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction


Instrumentator().instrument(app).expose(app)


# --- Config ---
MODEL_PATH = "model/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 5
SEQ_LENGTH = 60

# --- Load model once at server start ---
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    model = YourModelClass(input_size=INPUT_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    try:
        scaler = joblib.load(SCALER_PATH)
    except:
        scaler = None

load_model_and_scaler()

# --- Predict function ---
def predict_stock_price(input_sequence: np.ndarray) -> float:
    model_input = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = model(model_input)

    prediction = prediction.cpu().numpy().flatten()

    if scaler is not None:
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

    return prediction[0]
