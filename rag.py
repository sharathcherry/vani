"""rag.py — RAG answer retrieval with a 3-layer cache.

Retrieves a government scheme answer for an English query using three
cache layers to minimise latency and cost:

  L1 — In-memory LRU  (~0 ms)   Survives across requests.
  L2 — Azure Blob     (~30 ms)  Survives process restarts; 24-hour TTL.
  L3 — Local RAG      (~1–8 s)  Full hybrid search + rerank + Bedrock.

Also provides intent detection and URL extraction utilities used by the
voice pipeline to guard against off-topic queries.
"""

import hashlib
import time

from config import blob_service_client, CONTAINER_VOICE_INPUT, CACHE_TTL_SECONDS
from cache import answer_cache

_RAG_PIPELINE = None

# ---------------------------------------------------------------------------
# Intent guard
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS = {
    "scheme", "schemes", "yojana", "scholarship", "loan", "subsidy",
    "benefit", "benefits", "apply", "application", "eligible", "eligibility",
    "government", "pension", "insurance", "grant", "allowance", "ration",
    "card", "certificate", "income", "caste", "student", "farmer", "woman",
    "disability", "welfare", "help", "support", "fee", "free", "money",
    "what", "how", "when", "where", "which", "is there", "are there",
    "can i", "do i", "tell me", "list", "?",
}

import re
_URL_RE = re.compile(r'https?://[^\s,)>"\']+')


def has_scheme_intent(english_text: str) -> bool:
    """Return True if the query looks like a government scheme question."""
    lower = english_text.lower()
    return any(k in lower for k in _INTENT_KEYWORDS)


def extract_urls(text: str) -> list[str]:
    """Return all URLs found in text, deduplicated, order preserved."""
    seen: set[str] = set()
    result: list[str] = []
    for url in _URL_RE.findall(text):
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def strip_urls(text: str) -> str:
    """Remove URLs from text so TTS does not read them aloud."""
    return _URL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _normalize_query(q: str) -> str:
    """Lowercase, collapse whitespace, and strip session context prefix."""
    if "New question:" in q:
        q = q.split("New question:")[-1]
    return " ".join(q.lower().split())


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------
def get_rag_answer(english_query: str) -> str:
    """Retrieve a government scheme answer with 3-layer caching.

    Cache key is a SHA-256 hash of the normalised query so semantically
    identical questions (different whitespace / capitalisation) share a
    cache entry.
    """
    global _RAG_PIPELINE
    normalized = _normalize_query(english_query)
    cache_key  = hashlib.sha256(normalized.encode()).hexdigest() + ".txt"

    # -- L1: in-memory LRU (fastest) --------------------------------------
    mem_hit = answer_cache.get(normalized)
    if mem_hit is not None:
        print(f"[cache-L1] HIT (in-memory) key={cache_key[-12:]}")
        return mem_hit

    # -- L2: Azure Blob persistent cache (survives cold starts) -----------
    if blob_service_client:
        try:
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_VOICE_INPUT, blob=f"wa-cache/{cache_key}")
            if blob_client.exists():
                props = blob_client.get_blob_properties()
                age = time.time() - props.last_modified.timestamp()
                if age < CACHE_TTL_SECONDS:
                    cached = blob_client.download_blob().readall().decode("utf-8")
                    print(f"[cache-L2] HIT (Azure Blob, age={age/3600:.1f}h) key={cache_key[-12:]}")
                    answer_cache.set(normalized, cached)   # promote to L1
                    return cached
                print(f"[cache-L2] EXPIRED (age={age/3600:.1f}h)")
        except Exception:
            pass

    # -- L3: Local RAG Pipeline -------------------------------------------
    print("[cache] MISS — invoking RAG Pipeline")
    if _RAG_PIPELINE is None:
        from rag.pipeline import RAGPipeline
        _RAG_PIPELINE = RAGPipeline()
        
    result = _RAG_PIPELINE.answer(english_query)
    answer = result.answer if result and result.answer else "No information found."

    # Write-through to both cache layers
    answer_cache.set(normalized, answer)
    if blob_service_client:
        try:
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_VOICE_INPUT, blob=f"wa-cache/{cache_key}")
            blob_client.upload_blob(answer.encode("utf-8"), overwrite=True)
            print(f"[cache-L2] STORED key={cache_key[-12:]}")
        except Exception:
            pass

    return answer
