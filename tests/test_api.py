# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post("/predict", json={"text": "This product is great!"})
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_retrain():
    response = client.post("/retrain")
    assert response.status_code == 200
    assert "status" in response.json()
