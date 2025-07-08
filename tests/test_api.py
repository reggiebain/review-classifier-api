# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict() -> None:
    response = client.post("/predict", json={"text": "This product is great!"})
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_retrain() -> None:
    response = client.post("/retrain")
    assert response.status_code == 200
    assert "status" in response.json()
