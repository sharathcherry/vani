import time
from typing import List

from openai import OpenAI

from .config import NVIDIA_API_KEY, NVIDIA_BASE_URL, EMBED_MODEL, EMBED_BATCH_SIZE

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    return _client


def embed(texts: List[str], sleep_s: float = 0.4) -> List[List[float]]:
    """Embed a list of texts in batches. Returns list of 1024-dim vectors."""
    client = _get_client()
    results: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(
            model=EMBED_MODEL, input=batch, encoding_format="float"
        )
        results.extend(item.embedding for item in resp.data)
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(sleep_s)
    return results


def embed_query(text: str) -> List[float]:
    return embed([text])[0]
