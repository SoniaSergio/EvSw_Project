from fastapi import APIRouter, HTTPException, Query
from services.db_service import get_predictions, get_prediction_by_id, get_random_ecg_sample

router = APIRouter()

@router.get("/")
async def list_predictions(
    limit: int = Query(default=50, ge=1, le=200),
    skip: int = Query(default=0, ge=0),
):
    records = await get_predictions(limit=limit, skip=skip)
    return {"total": len(records), "skip": skip, "limit": limit, "data": records}

@router.get("/random")
async def get_random_sample():
    sample = await get_random_ecg_sample()
    if not sample:
        raise HTTPException(status_code=404, detail="Nessun campione disponibile")
    return sample

@router.get("/{prediction_id}")
async def get_prediction(prediction_id: str):
    record = await get_prediction_by_id(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Predizione non trovata")
    return record