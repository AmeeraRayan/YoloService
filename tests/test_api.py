import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import sys
import json
from fastapi.testclient import TestClient
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

client = TestClient(app)

def test_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!"}

def test_predict_success():
    response = client.post("/predict", json={
        "image_name": "file_71.jpg"
    })
    assert response.status_code == 200
    assert "message" not in response.json()
    assert "prediction_uid" in response.json()

def test_predict_missing_image_name():
    response = client.post("/predict", json={})
    assert response.status_code == 400

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_response_structure():
    response = client.post("/predict", json={"image_name": "file_71.jpg"})
    assert response.status_code == 200

    data = response.json()
    assert "prediction_uid" in data
    assert isinstance(data["prediction_uid"], str)

    assert "detection_count" in data
    assert isinstance(data["detection_count"], int)

    assert "labels" in data
    assert isinstance(data["labels"], list)

    assert "predicted_s3_key" in data
    assert data["predicted_s3_key"].endswith("_predicted.jpg")