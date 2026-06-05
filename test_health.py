from fastapi.testclient import TestClient
from voice_rag_server_azure import app

client = TestClient(app)

response = client.get("/health")
print("Response Status:", response.status_code)
print("Response JSON:", response.json())
