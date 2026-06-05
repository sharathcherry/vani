"""Hybrid retriever backed by Qdrant.

Queries the `schemes_hybrid` collection using QdrantClient.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.config import (
    INDEX_DIR,
    MAX_CHUNKS_PER_SCHEME,
    RERANK_TOP_K,
    CHUNK_TYPE_BOOST,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_LOCAL_PATH,
)

PARENT_DOCS_PATH = str(INDEX_DIR / "parent_docs.pkl")


@dataclass
class RetrievedChunk:
    id:          str
    scheme_id:   str
    scheme_name: str
    chunk_type:  str
    text:        str
    score:       float
    payload:     dict = field(default_factory=dict)

    def get_parent_doc(self, parent_docs: dict[str, dict]) -> dict:
        doc = parent_docs.get(self.scheme_id)
        if doc:
            return doc
        p = self.payload
        return {
            "scheme_id":       self.scheme_id,
            "scheme_name":     self.scheme_name,
            "state_or_ut":     p.get("state_or_ut", ""),
            "scheme_category": p.get("scheme_category", ""),
            "ministry":        p.get("ministry", ""),
            "summary":         p.get("summary", ""),
            "eligibility":     p.get("eligibility", ""),
            "benefits":        p.get("benefits", ""),
            "application":     p.get("application", ""),
            "link":            p.get("link", ""),
        }


class Retriever:
    """Production-grade hybrid retriever backed by Qdrant."""

    def __init__(self) -> None:
        if QDRANT_URL:
            self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            self._client = QdrantClient(path=QDRANT_LOCAL_PATH)
            
        self._parent_docs = _load_parent_docs()

    def retrieve(
        self,
        query_vector: np.ndarray | list[float],
        query_text:   str,
        metadata_filter: dict | None = None,
        top_k: int = RERANK_TOP_K,
    ) -> list[RetrievedChunk]:
        """Single-query dense retrieval via Qdrant."""
        
        vector = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        qdrant_filter = _build_qdrant_filter(metadata_filter)

        results = self._client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            query_filter=qdrant_filter,
            limit=top_k * 2,
        )

        chunks: list[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            chunk_type = payload.get("chunk_type", "full") or "full"
            boost = CHUNK_TYPE_BOOST.get(chunk_type, 1.0)
            
            chunks.append(RetrievedChunk(
                id          = str(hit.id),
                scheme_id   = payload.get("scheme_id", ""),
                scheme_name = payload.get("scheme_name", ""),
                chunk_type  = chunk_type,
                text        = payload.get("content", ""),
                score       = hit.score * boost,
                payload     = payload,
            ))

        chunks.sort(key=lambda c: c.score, reverse=True)
        return _deduplicate(chunks, MAX_CHUNKS_PER_SCHEME)[:top_k]

    def retrieve_multi(
        self,
        query_vectors: np.ndarray | Iterable[Iterable[float]],
        query_texts:   list[str],
        metadata_filter: dict | None = None,
        top_k: int = RERANK_TOP_K,
    ) -> list[RetrievedChunk]:
        """Multi-query retrieval — best score wins per chunk_id."""
        merged: dict[str, RetrievedChunk] = {}
        for vec, text in zip(query_vectors, query_texts):
            for chunk in self.retrieve(vec, text, metadata_filter=metadata_filter, top_k=top_k):
                prev = merged.get(chunk.id)
                if prev is None or chunk.score > prev.score:
                    merged[chunk.id] = chunk

        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return _deduplicate(ranked, MAX_CHUNKS_PER_SCHEME)[:top_k]

    def hybrid_retrieve(self, query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
        embedder = _get_default_embedder()
        vec = embedder.embed_query(query)
        chunks = self.retrieve(vec, query, metadata_filter=None, top_k=top_k)
        return [_chunk_to_dict(c) for c in chunks]


def _build_qdrant_filter(meta: dict | None) -> models.Filter | None:
    if not meta:
        return None

    must = []
    if state := meta.get("state"):
        must.append(models.FieldCondition(key="state_or_ut", match=models.MatchValue(value=state)))
    if category := meta.get("category"):
        must.append(models.FieldCondition(key="scheme_category", match=models.MatchValue(value=category)))
    
    bool_keys = {
        "for_women": "is_for_women",
        "for_farmers": "is_for_farmers",
        "for_disabled": "is_for_disabled",
        "for_sc_st": "is_for_sc_st",
        "for_students": "is_for_students",
    }
    for meta_key, field_name in bool_keys.items():
        if meta.get(meta_key):
            must.append(models.FieldCondition(key=field_name, match=models.MatchValue(value=True)))

    if not must:
        return None
    return models.Filter(must=must)


def _deduplicate(chunks: list[RetrievedChunk], max_per_scheme: int) -> list[RetrievedChunk]:
    counts: dict[str, int] = defaultdict(int)
    out: list[RetrievedChunk] = []
    for chunk in chunks:
        if counts[chunk.scheme_id] < max_per_scheme:
            out.append(chunk)
            counts[chunk.scheme_id] += 1
    return out


def _chunk_to_dict(c: RetrievedChunk) -> dict:
    return {
        "content": c.text,
        "score": c.score,
        "metadata": {
            "id":              c.id,
            "scheme_id":       c.scheme_id,
            "scheme_name":     c.scheme_name,
            "chunk_type":      c.chunk_type,
            "state_or_ut":     c.payload.get("state_or_ut", ""),
            "scheme_category": c.payload.get("scheme_category", ""),
            "ministry":        c.payload.get("ministry", ""),
            "link":            c.payload.get("link", ""),
            "eligibility":     c.payload.get("eligibility", ""),
        },
    }


def _load_parent_docs() -> dict[str, dict]:
    p = Path(PARENT_DOCS_PATH)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


_DEFAULT_EMBEDDER = None

def _get_default_embedder():
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        from rag.embedder import Embedder
        _DEFAULT_EMBEDDER = Embedder()
    return _DEFAULT_EMBEDDER
