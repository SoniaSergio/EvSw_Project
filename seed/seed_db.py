import pandas as pd
from pymongo import MongoClient
from cryptography.fernet import Fernet
import os, json

MONGO_URI      = os.environ['MONGO_URI']
ENCRYPTION_KEY = os.environ['ENCRYPTION_KEY']

MIT_LABEL_MAP = {
    0: 'N (Normale)',
    1: 'S (Sopraventricolare)',
    2: 'V (Ventricolare)',
    3: 'F (Fusion)',
    4: 'Q (Non classificabile)',
}

client  = MongoClient(MONGO_URI)
db      = client['ecgdb']
fernet  = Fernet(ENCRYPTION_KEY.encode())

if db.ecg_samples.count_documents({}) > 0:
    print("Seed gia eseguito, skip.")
    exit(0)

df = pd.read_csv('/data/mitbih_test.csv', header=None)
docs = []

for _, row in df.iterrows():
    signal    = row[:187].tolist()
    label     = int(row[187])
    encrypted = fernet.encrypt(json.dumps(signal).encode()).decode()
    docs.append({
        'signal_encrypted': encrypted,
        'ground_truth':     MIT_LABEL_MAP[label],
        'source':           'mitbih_seed',
    })

db.ecg_samples.insert_many(docs)
print(f"Inseriti {len(docs)} campioni in ecg_samples.")
