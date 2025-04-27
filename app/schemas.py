# app/schemas.py
from pydantic import BaseModel
from typing import List

class StockInput(BaseModel):
    data: List[float]  # assuming you send list of recent Close prices or features

class PredictionOutput(BaseModel):
    prediction: float  # or int if it's classification
