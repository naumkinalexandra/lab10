from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_upload_model_rejects_wrong_extension():
    response = client.post("/upload-model", files={"file": ("model.txt", b"bad", "text/plain")})
    assert response.status_code == 400


def test_predict_rejects_invalid_payload():
    response = client.post("/predict", json="not a record")
    assert response.status_code in (422, 503)
