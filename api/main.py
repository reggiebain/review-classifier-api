# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.model import load_model, predict
import sqlite3
from typing import List
from datetime import datetime
import os

app = FastAPI()
model, vectorizer = load_model()
DB_PATH = "data/predictions.db"


class ReviewRequest(BaseModel):
    text: str


class BatchReviewRequest(BaseModel):
    texts: List[str]


@app.post("/predict")
def predict_review(req: ReviewRequest) -> dict:
    try:
        prediction = int(predict(req.text, model, vectorizer))
        log_prediction(req.text, prediction)
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
def predict_batch(req: BatchReviewRequest) -> dict:
    try:
        predictions = predict(req.texts, model, vectorizer)
        for text, label in zip(req.texts, predictions):
            log_prediction(text, int(label))
        return {"predictions": [int(p) for p in predictions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
def get_logs():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT timestamp, review, prediction FROM predictions ORDER BY timestamp DESC LIMIT 100"
        )
        rows = c.fetchall()
        conn.close()
        return [{"timestamp": r[0], "review": r[1], "prediction": r[2]} for r in rows]
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
