import requests

BASE_URL = "http://localhost:8000"  # נעדכן את זה בהמשך לכתובת EC2

def test_home():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200