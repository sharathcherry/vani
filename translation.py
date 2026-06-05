"""translation.py — Language translation via Azure OpenAI.

Translates between any two supported Indian languages and English.
Used twice per voice query: native → English (before RAG) and
English → native (before TTS).

Caching: translated pairs are persisted to Azure Blob (translate-cache).
On a cache hit the LLM call is skipped entirely.
"""

import hashlib
import time

from config import blob_service_client, CONTAINER_TRANSLATE_CACHE, CACHE_TTL_SECONDS, LANG_NAMES
from rag.llm_client import LLMClient

def _translate_cache_get(cache_key: str) -> str | None:
    """Return cached translation from Azure Blob, or None on miss/expiry/error."""
    if not blob_service_client: return None
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_TRANSLATE_CACHE, blob=f"{cache_key}.txt")
        if blob_client.exists():
            props = blob_client.get_blob_properties()
            age = time.time() - props.last_modified.timestamp()
            if age > CACHE_TTL_SECONDS:
                return None
            return blob_client.download_blob().readall().decode("utf-8")
    except Exception:
        return None
    return None


def _translate_cache_set(cache_key: str, text: str) -> None:
    """Write translation to Azure Blob cache (best-effort; errors are non-fatal)."""
    if not blob_service_client: return
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_TRANSLATE_CACHE, blob=f"{cache_key}.txt")
        blob_client.upload_blob(text.encode("utf-8"), overwrite=True)
    except Exception as e:
        print(f"[translate] Blob cache write failed: {e}")


def translate_text(text: str, source_lang_code: str, target_lang_code: str) -> str:
    """Translate text between languages using Azure OpenAI.

    Returns the original text unchanged if:
    - source and target languages are the same, or
    - the input text is empty, or
    - the LLM call fails (graceful degradation).

    Checks Azure Blob cache before calling LLM and writes through on a miss.
    """
    src = LANG_NAMES.get(source_lang_code, "English")
    tgt = LANG_NAMES.get(target_lang_code, "English")

    if src == tgt or not text.strip():
        return text

    cache_key = hashlib.sha256(f"{text}|{source_lang_code}|{target_lang_code}".encode()).hexdigest()
    cached = _translate_cache_get(cache_key)
    if cached is not None:
        print(f"[translate] Azure Blob cache hit: {src} -> {tgt}")
        return cached

    prompt = (
        f"Translate the following {src} text to {tgt}. "
        f"Output ONLY the translation, nothing else.\n\n{text}"
    )
    
    try:
        # LLMClient already uses LLM_PROVIDER=azure and correctly loads AZURE_OPENAI_ endpoints
        llm = LLMClient()
        translated = llm.complete(
            system="You are a professional language translator. Only return the translated text.",
            user=prompt
        ).strip()
        
        print(f"[translate] {src} -> {tgt}: '{translated[:80]}'")
        if translated:
            _translate_cache_set(cache_key, translated)
            return translated
        return text
    except Exception as e:
        print(f"[translate] failed ({e}), returning original")
        return text
