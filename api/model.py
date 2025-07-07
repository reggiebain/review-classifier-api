# api/model.py
import joblib
from typing import Union, List


def load_model():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer


def predict(text: Union[str, List[str]], model, vectorizer):
    if isinstance(text, str):
        X = vectorizer.transform([text])
        return model.predict(X)[0]
    else:
        X = vectorizer.transform(text)
        return model.predict(X)
