# app/model.py
import torch
import numpy as np
import configparser
import os

from ml_pipeline.model_architecture import LSTMModel

# --- Load config ---
config = configparser.ConfigParser()
config_filepath = os.path.join('./ml_pipeline', 'config.ini')
config.read(config_filepath)

# config = configparser()
# config.read('./ml_pipeline/config.ini') 

INPUT_SIZE = config.getint('MODEL', 'input_size')
HIDDEN_SIZE = config.getint('MODEL', 'hidden_size')
NUM_LAYERS = config.getint('MODEL', 'num_layers')

# --- Config ---
MODEL_PATH = config.get('model', 'model_path', fallback="./models/best_model.pth")  # Make this more flexible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model once at server start ---
model = None

def load_model():
    global model
    model = LSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

load_model()

# --- Predict function ---
def predict_stock_price(input_sequence: np.ndarray) -> float:
    model_input = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = model(model_input)

    prediction = prediction.cpu().numpy().flatten()
    prediction = np.clip(prediction, 0, None)  # Ensures non-negative predictions
    prediction = np.round(prediction, 3)  # Round-off to 3 decimal places
    return prediction[0]
