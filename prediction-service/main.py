from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routers import predict
from services.db_service import connect_db, close_db
from pydantic import BaseModel
from typing import List, Optional

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()

app = FastAPI(
    title="ECG Prediction Service",
    description="Inferenza CNN 1D e Random Forest su segnali ECG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    signal: List[float]
    ground_truth: Optional[str] = None

app.include_router(predict.router, prefix="/predict", tags=["prediction"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "prediction-service"}