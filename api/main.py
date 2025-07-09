# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.model import load_model, predict
import sqlite3
from typing import List
from datetime import datetime
import os

app = FastAPI(
    title="Review Classifier API",
    description="""
A simple machine learning API for predicting sentiment of a review made using scikit-learn (logistic regression).  
Built by Reggie Bain.
""",
    version="1.0.0",
    contact={
        "name": "Reggie Bain",
        "url": "https://reggiebain.github.io/",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

model, vectorizer = load_model()
DB_PATH = "data/predictions.db"


class ReviewRequest(BaseModel):
    text: str


class BatchReviewRequest(BaseModel):
    texts: List[str]


class PredictionResponse(BaseModel):
    prediction: int


class BatchPredictionResponse(BaseModel):
    predictions: List[int]


class PredictionLog(BaseModel):
    timestamp: str
    review: str
    prediction: int


@app.post("/predict", response_model=PredictionResponse)
def predict_review(req: ReviewRequest) -> PredictionResponse:
    try:
        result = predict(req.text, model, vectorizer)
        if isinstance(result, list):
            prediction = int(result[0])
        else:
            prediction = int(result)
        log_prediction(req.text, prediction)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchReviewRequest) -> BatchPredictionResponse:
    try:
        result = predict(req.texts, model, vectorizer)
        if isinstance(result, int):
            predictions = [result]
        else:
            predictions = result
        for text, label in zip(req.texts, predictions):
            log_prediction(text, int(label))
        return BatchPredictionResponse(predictions=[int(p) for p in predictions])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs", response_model=List[PredictionLog])
def get_logs() -> List[PredictionLog]:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT timestamp, review, prediction FROM predictions ORDER BY timestamp DESC LIMIT 100"
        )
        rows = c.fetchall()
        conn.close()
        return [
            PredictionLog(timestamp=r[0], review=r[1], prediction=r[2]) for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain() -> dict:
    try:
        os.system("python model/train.py")
        global model, vectorizer
        model, vectorizer = load_model()
        return {"status": "Model retrained and reloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def log_prediction(text: str, prediction: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        review TEXT,
        prediction INTEGER
    )"""
    )
    c.execute(
        "INSERT INTO predictions (timestamp, review, prediction) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), text, prediction),
    )
    conn.commit()
    conn.close()
