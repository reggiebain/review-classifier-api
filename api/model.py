# api/model.py
import joblib
import os

def load_model():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

def predict(text, model, vectorizer):
    X = vectorizer.transform([text])
    return model.predict(X)[0]