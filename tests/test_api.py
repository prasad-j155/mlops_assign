from fastapi.testclient import TestClient
from api.main import app
import pytest
import numpy as np

client = TestClient(app)


class DummyModel:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """
    Automatically mock the ML model for all API tests
    """
    from api import main
    monkeypatch.setattr(main, "get_model", lambda: DummyModel())


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
