from pydantic import BaseModel, Field, field_validator
from typing import List


class PredictionInput(BaseModel):
    """Model untuk input prediksi (numeric-encoded)"""

    region: int = Field(..., example=0)
    category: int = Field(..., example=1)
    parameter: int = Field(..., example=0)
    mode: int = Field(..., example=2)
    powertrain: int = Field(..., example=1)
    year: int = Field(..., ge=2000, le=2100, example=2023)

    @field_validator("year")
    def validate_year(cls, v):
        if not (2000 <= v <= 2100):
            raise ValueError("Year must be between 2000 and 2100")
        return v


class BatchPredictionInput(BaseModel):
    """Model untuk batch prediction"""

    instances: List[PredictionInput] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Model untuk response prediksi"""

    predictions: List[float]
    latency_seconds: float
    timestamp: float


class HealthResponse(BaseModel):
    """Model untuk health check response"""

    status: str
    model_available: bool
    system_info: dict
