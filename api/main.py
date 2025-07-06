# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.model import load_model, predict

app = FastAPI()
model, vectorizer = load_model()

class ReviewRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_review(req: ReviewRequest):
    try:
        prediction = predict(req.text, model, vectorizer)
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}