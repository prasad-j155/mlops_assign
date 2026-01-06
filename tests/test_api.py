from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
