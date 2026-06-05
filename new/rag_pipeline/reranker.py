from typing import Any, Dict, List

import requests

from .config import NVIDIA_API_KEY, NVIDIA_BASE_URL, RERANK_MODEL, FINAL_TOP_K


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = FINAL_TOP_K,
) -> List[Dict[str, Any]]:
    """Rerank chunks using the NVIDIA NIM ranking endpoint."""
    if not chunks:
        return chunks

    texts = [c.get("text", "") for c in chunks]

    url = f"{NVIDIA_BASE_URL}/ranking"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "query": {"role": "user", "content": query},
        "passages": [{"role": "user", "content": t} for t in texts],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    rankings = resp.json().get("rankings", [])
    ranked_by_score = sorted(rankings, key=lambda x: x.get("logit", 0.0), reverse=True)

    reranked = []
    for item in ranked_by_score[:top_k]:
        idx = item["index"]
        chunk = dict(chunks[idx])
        chunk["rerank_score"] = item.get("logit", 0.0)
        reranked.append(chunk)
    return reranked
