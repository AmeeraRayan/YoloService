from fastapi.testclient import TestClient
from app import app  # ודאי שקובץ app.py מכיל את האובייקט FastAPI בשם app
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
client = TestClient(app)

def test_homepage():
    response = client.get("/")
    assert response.status_code == 200