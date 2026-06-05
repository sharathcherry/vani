"""session.py — Manages conversation context persistence in Azure Blob Storage."""

import json
import time

from config import (
    SESSION_TTL_SECONDS,
    blob_service_client,
    CONTAINER_SESSIONS,
)

def _get_blob_name(phone_number: str) -> str:
    # Ensure phone number is alphanumeric for safe keys
    clean_phone = "".join(c for c in phone_number if c.isalnum())
    return f"{clean_phone}.json"

def get_session(phone_number: str) -> dict:
    """Retrieve session from Azure Blob Storage if it exists and hasn't expired."""
    blob_name = _get_blob_name(phone_number)
    
    if not blob_service_client:
        return _default_session()
        
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_SESSIONS, blob=blob_name)
        if blob_client.exists():
            data = json.loads(blob_client.download_blob().readall().decode("utf-8"))
            
            # Check TTL
            if time.time() - data.get("timestamp", 0) > SESSION_TTL_SECONDS:
                return _default_session()
            return data
        else:
            return _default_session()
    except Exception as e:
        print(f"[session] Error retrieving session: {e}")
        return _default_session()

def update_session(phone_number: str, role: str, content: str, lang: str = "en-IN") -> None:
    """Append a message to the session history in Azure Blob Storage."""
    session = get_session(phone_number)
    session["history"].append({"role": role, "content": content})
    session["timestamp"] = time.time()
    session["lang"] = lang

    # Keep only the last 6 turns (12 messages) to prevent context bloat
    if len(session["history"]) > 12:
        session["history"] = session["history"][-12:]

    blob_name = _get_blob_name(phone_number)
    if not blob_service_client:
        return
        
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_SESSIONS, blob=blob_name)
        blob_client.upload_blob(json.dumps(session).encode("utf-8"), overwrite=True)
    except Exception as e:
        print(f"[session] Error saving session: {e}")

def _default_session() -> dict:
    return {
        "timestamp": time.time(),
        "lang": "en-IN",
        "history": [],
    }
