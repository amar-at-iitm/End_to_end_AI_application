# app/main.py
from fastapi import FastAPI
import numpy as np
from model import predict_stock_price
from app.schemas import PredictRequest, PredictResponse
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()
Instrumentator().instrument(app).expose(app)


# @app.post("/predict", response_model=PredictResponse)
# def get_prediction(request: PredictRequest):
#     input_data = np.array(request.data)  # Just convert to np.array
#     #output = predict_stock_price(input_data)                 # for single prediction
#     output = predict_stock_price(input_data, steps=30)        # for multiple predictions
#     return PredictResponse(prediction=[float(output[0])])


@app.post("/predict", response_model=PredictResponse)
def get_prediction(request: PredictRequest):
    input_data = np.array(request.data)
    output = predict_stock_price(input_data, steps=30)  # Returns a list
    return PredictResponse(
        prediction=[float(val) for val in output]
    )  # Return full list
