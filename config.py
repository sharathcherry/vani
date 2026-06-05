"""config.py — Centralised configuration for the Gov Schemes Voice Bot.

All environment variables, Azure clients, and shared constants live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Azure Storage Configuration
# ---------------------------------------------------------------------------
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")

if AZURE_STORAGE_CONNECTION_STRING:
    from azure.storage.blob import BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
else:
    blob_service_client = None

CONTAINER_VOICE_INPUT = "voice-input"
CONTAINER_VOICE_OUTPUT = "voice-output"
CONTAINER_TRANSLATE_CACHE = "translate-cache"
CONTAINER_TTS_CACHE = "wa-tts-cache"
CONTAINER_SESSIONS = "sessions"

# ---------------------------------------------------------------------------
# Twilio
# ---------------------------------------------------------------------------
TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN",  "")

# ---------------------------------------------------------------------------
# Azure Speech (STT + TTS)
# ---------------------------------------------------------------------------
AZURE_SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY",    "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "centralindia")

# Azure Neural TTS voice for each supported Indian language
AZURE_VOICE_MAP: dict[str, tuple[str, str]] = {
    "en-IN": ("en-IN", "en-IN-NeerjaNeural"),
    "hi-IN": ("hi-IN", "hi-IN-SwaraNeural"),
    "te-IN": ("te-IN", "te-IN-ShrutiNeural"),
    "ta-IN": ("ta-IN", "ta-IN-PallaveNeural"),
    "kn-IN": ("kn-IN", "kn-IN-SapnaNeural"),
    "ml-IN": ("ml-IN", "ml-IN-SobhanaNeural"),
    "mr-IN": ("mr-IN", "mr-IN-AarohiNeural"),
    "bn-IN": ("bn-IN", "bn-IN-TanishaaNeural"),
    "gu-IN": ("gu-IN", "gu-IN-DhwaniNeural"),
    "pa-IN": ("pa-IN", "pa-IN-VaaniNeural"),
    "or-IN": ("or-IN", "or-IN-SubhasiniNeural"),
    "as-IN": ("as-IN", "as-IN-YashicaNeural"),
}

# ---------------------------------------------------------------------------
# TTL settings
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS = 3  * 60 * 60   # 3 hours  — per-user conversation context
CACHE_TTL_SECONDS   = 24 * 60 * 60   # 24 hours — RAG answer cache

# ---------------------------------------------------------------------------
# Language display names (used in translation prompts)
# ---------------------------------------------------------------------------
LANG_NAMES: dict[str, str] = {
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "or-IN": "Odia",
    "as-IN": "Assamese",
    "en-IN": "English",
}

# ---------------------------------------------------------------------------
# Azure AI Search Configuration
# ---------------------------------------------------------------------------
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "gov-schemes")
