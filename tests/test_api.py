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

def test_predict_missing_image_name():
    response = client.post("/predict", json={})
    assert response.status_code == 400

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

