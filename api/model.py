# api/model.py
from typing import Union, List, Tuple, Any
import joblib
import os
import numpy as np

MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


def load_model() -> Tuple[Any, Any]:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict(text: Union[str, List[str]], model: Any, vectorizer: Any) -> Union[int, List[int]]:
    if isinstance(text, str):
        X = vectorizer.transform([text])
        return int(model.predict(X)[0])
    else:
        X = vectorizer.transform(text)
        return model.predict(X).tolist()