from fastapi import APIRouter
from schemas.ecg import PredictionRequest
from services.cnn_service import predict_cnn
from services.rf_service import predict_rf
from services.db_service import save_prediction
import numpy as np

router = APIRouter()

CLASS_NAMES = [
    "N (Normale)", "S (Sopraventricolare)",
    "V (Ventricolare)", "F (Fusion)", "Q (Non classificabile)"
]

CONFIDENCE_THRESHOLD = 0.60

def build_output(probs: list, class_names: list) -> dict:
    idx = int(np.argmax(probs))
    max_conf = float(probs[idx])
    return {
        "diagnosi": class_names[idx],
        "confidenza": round(max_conf, 4),
        "distribuzione": {
            class_names[i]: round(float(probs[i]), 4)
            for i in range(len(probs))
        },
        "stato_affidabilita": (
            "Bassa confidenza (Revisione clinica raccomandata)"
            if max_conf < CONFIDENCE_THRESHOLD
            else "Diagnosi ad alta confidenza"
        )
    }

@router.post("/")
async def predict(req: PredictionRequest):
    signal = req.signal  # lista di 187 float
    # Recuperiamo il ground_truth dallo schema, con default "Non specificato"
    ground_truth = req.ground_truth if req.ground_truth else "Non specificato"

    probs_cnn = predict_cnn(signal)
    probs_rf  = predict_rf(signal)

    result = {
        "cnn": build_output(probs_cnn, CLASS_NAMES),
        "rf":  build_output(probs_rf,  CLASS_NAMES)
    }

    # Passiamo anche il ground_truth al servizio di salvataggio
    await save_prediction(signal, result, ground_truth=ground_truth)
    
    return result