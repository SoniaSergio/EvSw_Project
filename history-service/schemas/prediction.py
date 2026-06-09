from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime


class ModelOutput(BaseModel):
    diagnosi: str
    confidenza: float
    distribuzione: Dict[str, float]
    stato_affidabilita: str


class PredictionRecord(BaseModel):
    id: str = Field(alias="_id")
    timestamp: datetime
    cnn: ModelOutput
    rf: ModelOutput

    model_config = {"populate_by_name": True}


class PredictionRecordPublic(BaseModel):
    #Schema pubblico esposto dall'API: omette il segnale grezzo per alleggerire la risposta.
    id: str
    timestamp: datetime
    cnn: ModelOutput
    rf: ModelOutput