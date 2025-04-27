# app/main.py
from fastapi import FastAPI
from app.model import predict
from app.schemas import PredictRequest, PredictResponse
import torch

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def get_prediction(request: PredictRequest):
    input_data = torch.tensor(request.data).float().unsqueeze(0)  # assuming 2D input, make it batch size 1
    output = predict(input_data)
    return PredictResponse(prediction=output.item())
