# api/model.py
from typing import Union, List, Tuple
import joblib
import os
import numpy as np

MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

def load_model() -> Tuple[any, any]:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict(text: Union[str, List[str]], model, vectorizer) -> Union[int, List[int]]:
    if isinstance(text, str):
        X = vectorizer.transform([text])
        return int(model.predict(X)[0])
    else:
        X = vectorizer.transform(text)
        return model.predict(X).tolist()
