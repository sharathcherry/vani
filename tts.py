"""tts.py — Text-to-Speech via Azure Neural TTS.

Converts the translated answer text into an MP3 voice note:
  1. Build an SSML payload with the appropriate Neural voice for the language
  2. POST to Azure Cognitive Services TTS endpoint
  3. Upload the MP3 to Azure Blob
  4. Return a 1-hour presigned SAS URL that Twilio can attach to a WhatsApp message

Caching: synthesized MP3s are stored in Azure Blob under wa-tts-cache/ keyed by
SHA-256(sanitised_text|lang_code). Identical text+language combinations
skip the Azure call (~1-2 s) and return a fresh SAS URL for the
already-stored audio.
"""

import hashlib
import time
from datetime import datetime, timedelta, timezone

import httpx
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

from config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_VOICE_MAP,
    CACHE_TTL_SECONDS,
    CONTAINER_TTS_CACHE,
    blob_service_client,
)

def _generate_sas_url(blob_name: str) -> str:
    """Generate a 1-hour SAS URL for the given blob in the TTS cache container."""
    if not blob_service_client: return ""
    account_name = blob_service_client.account_name
    account_key = blob_service_client.credential.account_name if hasattr(blob_service_client.credential, "account_name") else blob_service_client.credential.account_key
    
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=CONTAINER_TTS_CACHE,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    return f"https://{account_name}.blob.core.windows.net/{CONTAINER_TTS_CACHE}/{blob_name}?{sas_token}"

def synthesize_speech(text: str, message_sid: str, lang_code: str = "en-IN") -> str:
    """Azure Neural TTS → Azure Blob → presigned SAS URL for Twilio media attachment.

    Args:
        text:        The answer text to synthesise (URLs already stripped).
        message_sid: Used as the S3 object key to keep audio files unique.
        lang_code:   BCP-47 language tag, e.g. "hi-IN".

    Returns:
        A 1-hour presigned Azure Blob SAS URL that Twilio can serve as a WhatsApp audio message.
    """
    if not AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY environment variable is not set.")

    xml_lang, voice_name = AZURE_VOICE_MAP.get(lang_code, ("en-IN", "en-IN-NeerjaNeural"))
    print(f"[TTS] Azure Speech: lang={lang_code}, voice={voice_name}")

    # Sanitise text for SSML — done first so the cache key matches actual content
    safe_text = text[:2000].replace("&", "and").replace("<", "").replace(">", "")

    # Check Azure Blob TTS cache (same text + voice = identical audio)
    cache_key = hashlib.sha256(f"{safe_text}|{lang_code}".encode()).hexdigest()
    blob_name = f"{cache_key}.mp3"
    
    if blob_service_client:
        try:
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_TTS_CACHE, blob=blob_name)
            if blob_client.exists():
                props = blob_client.get_blob_properties()
                age = time.time() - props.last_modified.timestamp()
                if age <= CACHE_TTL_SECONDS:
                    print(f"[TTS] Azure Blob cache hit — skipping synthesis")
                    return _generate_sas_url(blob_name)
        except Exception:
            pass  # cache miss — fall through to Azure

    ssml = (
        f"<speak version='1.0' xml:lang='{xml_lang}'>"
        f"<voice name='{voice_name}'>{safe_text}</voice>"
        f"</speak>"
    )

    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3",
        "User-Agent": "GovSchemes-VoiceBot/1.0",
    }

    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, headers=headers, content=ssml.encode("utf-8"))
    resp.raise_for_status()

    audio_bytes = resp.content
    print(f"[TTS] Azure returned {len(audio_bytes):,} bytes of MP3")

    # Upload to Azure Blob and generate a SAS URL (valid 1 hour)
    if blob_service_client:
        try:
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_TTS_CACHE, blob=blob_name)
            blob_client.upload_blob(audio_bytes, overwrite=True)
            return _generate_sas_url(blob_name)
        except Exception as e:
            print(f"[TTS] Failed to upload to Azure Blob: {e}")
            
    return ""
