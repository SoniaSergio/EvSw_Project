import os
import json
from datetime import datetime, timezone
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from cryptography.fernet import Fernet

_client: AsyncIOMotorClient | None = None
_db = None

secret_key = os.getenv("ENCRYPTION_KEY")
if not secret_key:
    raise ValueError("Manca la ENCRYPTION_KEY nel file .env!")
cipher = Fernet(secret_key.encode('utf-8'))

async def connect_db():
    global _client, _db
    uri = os.getenv("MONGO_URI", "mongodb://mongo:27017/ecgdb")
    _client = AsyncIOMotorClient(uri)
    _db = _client.get_default_database()
    print(f"[prediction-service] MongoDB connesso: {uri}")

async def close_db():
    if _client:
        _client.close()

async def save_prediction(signal: list, result: dict, ground_truth: Optional[str] = None) -> str:
    # Serializzazione
    sig_str = json.dumps(signal).encode('utf-8')
    cnn_str = json.dumps(result["cnn"]).encode('utf-8')
    rf_str = json.dumps(result["rf"]).encode('utf-8')
    # Serializzazione del ground truth se presente
    gt_str = json.dumps(ground_truth).encode('utf-8') if ground_truth else b"null"
    
    doc = {
        "timestamp": datetime.now(timezone.utc),
        "signal": cipher.encrypt(sig_str).decode('utf-8'),
        "cnn": cipher.encrypt(cnn_str).decode('utf-8'),
        "rf": cipher.encrypt(rf_str).decode('utf-8'),
        "ground_truth": cipher.encrypt(gt_str).decode('utf-8') # Ora è cifrato come il resto
    }
    
    inserted = await _db.predictions.insert_one(doc)
    return str(inserted.inserted_id) 
