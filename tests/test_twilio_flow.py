import pytest
from fastapi.testclient import TestClient
from voice_rag_server_azure import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_twilio_webhook_text_join():
    data = {
        "MessageSid": "SM123",
        "From": "+1234567890",
        "To": "+0987654321",
        "Body": "join sandbox"
    }
    response = client.post("/webhook", data=data)
    assert response.status_code == 200
    assert "xml" in response.headers["content-type"]
    assert "<Message>" in response.text

def test_twilio_webhook_voice_note():
    data = {
        "MessageSid": "MM123",
        "From": "+1234567890",
        "To": "+0987654321",
        "MediaUrl0": "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/MM123/Media/ME123"
    }
    response = client.post("/webhook", data=data)
    assert response.status_code == 200
    assert "xml" in response.headers["content-type"]
    assert "Got your voice note" in response.text
