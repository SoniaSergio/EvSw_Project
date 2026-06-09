from signal import signal

import numpy as np
import pandas as pd
import joblib

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load("/app/models/ecg_rf_model.pkl")
    return _model

def extract_features(signal: list) -> pd.DataFrame:
    s = pd.DataFrame([signal])
    feats = pd.DataFrame()
    feats['media']          = s.mean(axis=1)
    feats['dev_std']        = s.std(axis=1)
    feats['skewness']       = s.skew(axis=1)
    feats['kurtosis']       = s.kurt(axis=1)
    feats['valore_massimo'] = s.max(axis=1)
    feats['valore_minimo']  = s.min(axis=1)
    feats['range']          = feats['valore_massimo'] - feats['valore_minimo']
    feats['energia']        = (s ** 2).sum(axis=1)
    feats['zero_crossing']  = (s.values[:, :-1] * s.values[:, 1:] < 0).sum(axis=1)
    return feats

def predict_rf(signal: list) -> list:
    feats = extract_features(signal)
    probs = get_model().predict_proba(feats)[0]
    return probs.tolist()