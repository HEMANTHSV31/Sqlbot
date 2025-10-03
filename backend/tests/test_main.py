import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_transcribe_endpoint():
    # Test with mock audio file
    files = {"file": ("test.wav", b"fake_audio_content", "audio/wav")}
    data = {"session_id": "test_session", "role": "user"}
    
    response = client.post("/transcribe", files=files, data=data)
    assert response.status_code in [200, 401]  # 401 if no auth

def test_query_endpoint():
    data = {
        "session_id": "test_session",
        "role": "user",
        "text": "Show me all users"
    }
    
    response = client.post("/query", data=data)
    assert response.status_code in [200, 401]  # 401 if no auth

def test_health_check():
    response = client.get("/")
    assert response.status_code == 404  # No root endpoint