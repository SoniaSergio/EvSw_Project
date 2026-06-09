from pydantic import BaseModel, field_validator
from typing import List, Optional

class PredictionRequest(BaseModel):
    signal: List[float]
    ground_truth: Optional[str] = "Non specificato"

    @field_validator("signal")
    @classmethod
    def check_length(cls, v):
        if len(v) != 187:
            raise ValueError("Il segnale deve avere esattamente 187 campioni")
        return v