import numpy as np
from tensorflow.keras.models import load_model
_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_model("/app/models/ecg_cnn_model.h5", compile=False)
    return _model
def predict_cnn(signal: list) -> list:
    arr = np.array(signal).reshape(1, 187, 1)
    probs = get_model().predict(arr, verbose=0)[0]
    return probs.tolist()
