# app/schemas.py

from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    data: List[List[float]]  # 2D List: (seq_length, input_size)

class PredictResponse(BaseModel):
    prediction: float
