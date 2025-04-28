# app/main.py
import torch
from fastapi import FastAPI
import numpy as np
from app.model import predict_stock_price
from app.schemas import PredictRequest, PredictResponse
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()
Instrumentator().instrument(app).expose(app)

# @app.post("/predict", response_model=PredictResponse)
# def get_prediction(request: PredictRequest):
#     # Convert the request data into a NumPy array
#     input_data = np.array(request.data)  # Convert List[List[float]] to np.array

#     # Convert the NumPy array to a PyTorch tensor and add batch dimension
#     input_data = torch.tensor(input_data).float().unsqueeze(0)

#     # Call the predict function
#     output = predict_stock_price(input_data)

#     # Return the prediction as a response
#     return PredictResponse(prediction=output.item())


@app.post("/predict", response_model=PredictResponse)
def get_prediction(request: PredictRequest):
    input_data = np.array(request.data)  # Just convert to np.array
    output = predict_stock_price(input_data)  # pass np.array directly
    return PredictResponse(prediction=output)
