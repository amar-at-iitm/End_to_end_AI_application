# app/main.py
from fastapi import FastAPI
from app import model, schemas
from app.model import predict
from app.schemas import PredictRequest, PredictResponse
import torch

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def get_prediction(request: PredictRequest):
    input_data = torch.tensor(request.data).float().unsqueeze(0) 
    output = predict(input_data)
    return PredictResponse(prediction=output.item())



@app.get("/")
def read_root():
    return {"message": "Stock Prediction API running successfully!"}

@app.post("/predict", response_model=schemas.PredictionOutput)
def predict(stock_input: schemas.StockInput):
    pred = model.predict_next(stock_input.data)
    return {"prediction": pred}
