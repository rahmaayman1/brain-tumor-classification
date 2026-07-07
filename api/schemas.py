from pydantic import BaseModel
from typing import Dict

class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    description: str
    all_probabilities: Dict[str, float]

class PredictionResponse(BaseModel):
    filename: str
    prediction: PredictionResult