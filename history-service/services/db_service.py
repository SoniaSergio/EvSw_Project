import os
import json
import random
from datetime import datetime
from bson import ObjectId
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
    print(f"[history-service] MongoDB connesso: {uri}")

async def close_db():
    if _client:
        _client.close()

def _decrypt_doc(doc: dict, exclude_signal: bool = False) -> dict:
    doc["id"] = str(doc.pop("_id"))
    if isinstance(doc.get("timestamp"), datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()
    if "cnn" in doc and isinstance(doc["cnn"], str):
        doc["cnn"] = json.loads(cipher.decrypt(doc["cnn"].encode('utf-8')).decode('utf-8'))
    if "rf" in doc and isinstance(doc["rf"], str):
        doc["rf"] = json.loads(cipher.decrypt(doc["rf"].encode('utf-8')).decode('utf-8'))
    if "ground_truth" in doc and isinstance(doc["ground_truth"], str):
        try:
            gt_data = cipher.decrypt(doc["ground_truth"].encode('utf-8')).decode('utf-8')
            doc["ground_truth"] = json.loads(gt_data)
        except:
            pass
    if exclude_signal:
        doc.pop("signal", None)
    elif "signal" in doc and isinstance(doc["signal"], str):
        doc["signal"] = json.loads(cipher.decrypt(doc["signal"].encode('utf-8')).decode('utf-8'))
    return doc

async def get_predictions(limit: int = 20, skip: int = 0) -> list[dict]:
    cursor = (
        _db.predictions
        .find({}, {"signal": 0})
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )
    results = []
    async for doc in cursor:
        results.append(_decrypt_doc(doc, exclude_signal=True))
    return results

async def get_prediction_by_id(prediction_id: str) -> dict | None:
    try:
        oid = ObjectId(prediction_id)
    except Exception:
        return None
    doc = await _db.predictions.find_one({"_id": oid})
    if doc is None:
        return None
    return _decrypt_doc(doc, exclude_signal=False)

async def count_predictions() -> int:
    return await _db.predictions.count_documents({})

async def delete_prediction(prediction_id: str) -> bool:
    try:
        oid = ObjectId(prediction_id)
    except Exception:
        return False
    result = await _db.predictions.delete_one({"_id": oid})
    return result.deleted_count == 1

async def get_stats() -> dict:
    cursor = _db.predictions.find({}, {"signal": 0})
    cnn_stats = {}
    rf_stats = {}
    total = 0
    async for doc in cursor:
        total += 1
        decrypted = _decrypt_doc(doc, exclude_signal=True)
        cnn_diag = decrypted["cnn"]["diagnosi"]
        rf_diag = decrypted["rf"]["diagnosi"]
        cnn_stats[cnn_diag] = cnn_stats.get(cnn_diag, 0) + 1
        rf_stats[rf_diag] = rf_stats.get(rf_diag, 0) + 1
    cnn_stats = dict(sorted(cnn_stats.items(), key=lambda item: item[1], reverse=True))
    rf_stats = dict(sorted(rf_stats.items(), key=lambda item: item[1], reverse=True))
    return {
        "total": total,
        "cnn_distribution": cnn_stats,
        "rf_distribution": rf_stats,
    }

async def get_random_ecg_sample() -> dict | None:
    pipeline = [{"$sample": {"size": 1}}]
    docs = await _db.ecg_samples.aggregate(pipeline).to_list(1)
    if not docs:
        return None
    doc = docs[0]
    signal = json.loads(cipher.decrypt(doc['signal_encrypted'].encode()).decode())
    return {
        "id":           str(doc['_id']),
        "signal":       signal,
        "ground_truth": doc.get('ground_truth', None),
    }
