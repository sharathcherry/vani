"""Hybrid retriever backed by Azure AI Search.

Queries the `AZURE_SEARCH_INDEX` collection using azure-search-documents.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType

from rag.config import (
    INDEX_DIR,
    MAX_CHUNKS_PER_SCHEME,
    RERANK_TOP_K,
    CHUNK_TYPE_BOOST,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX,
    AZURE_SEARCH_SEMANTIC_CONFIG,
    AZURE_SEARCH_USE_SEMANTIC,
    AZURE_SEARCH_VECTOR_K,
    AZURE_SEARCH_TOP,
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
    """Production-grade hybrid retriever backed by Azure AI Search."""

    def __init__(self) -> None:
        if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
            print("WARNING: AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_API_KEY is not set. Retrieval will fail.")
            self._client = None
        else:
            self._client = SearchClient(
                endpoint=AZURE_SEARCH_ENDPOINT,
                index_name=AZURE_SEARCH_INDEX,
                credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
            )
        self._parent_docs = _load_parent_docs()

    def retrieve(
        self,
        query_vector: np.ndarray | list[float],
        query_text:   str,
        metadata_filter: dict | None = None,
        top_k: int = RERANK_TOP_K,
    ) -> list[RetrievedChunk]:
        """Single-query hybrid retrieval via Azure AI Search."""
        
        vector = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        odata_filter = _build_odata_filter(metadata_filter)

        vector_query = VectorizedQuery(
            vector=vector, 
            k_nearest_neighbors=AZURE_SEARCH_VECTOR_K, 
            fields="embedding"
        )
        
        query_type = QueryType.SEMANTIC if AZURE_SEARCH_USE_SEMANTIC else QueryType.SIMPLE

        results = self._client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            filter=odata_filter,
            top=AZURE_SEARCH_TOP,
            query_type=query_type,
            semantic_configuration_name=AZURE_SEARCH_SEMANTIC_CONFIG if AZURE_SEARCH_USE_SEMANTIC else None,
        )

        chunks: list[RetrievedChunk] = []
        for hit in results:
            payload = hit
            chunk_type = payload.get("chunk_type", "full") or "full"
            boost = CHUNK_TYPE_BOOST.get(chunk_type, 1.0)
            
            # Azure Search provides @search.score for hybrid/semantic queries
            # semantic queries might have @search.reranker_score 
            score = hit.get("@search.reranker_score") or hit.get("@search.score", 1.0)
            
            chunks.append(RetrievedChunk(
                id          = str(payload.get("id", "")),
                scheme_id   = payload.get("scheme_id", ""),
                scheme_name = payload.get("scheme_name", ""),
                chunk_type  = chunk_type,
                text        = payload.get("content", ""),
                score       = score * boost,
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


def _build_odata_filter(meta: dict | None) -> str | None:
    if not meta:
        return None

    clauses = []
    if state := meta.get("state"):
        # Azure Search string filter using eq
        # escape single quotes just in case
        state = state.replace("'", "''")
        clauses.append(f"state_or_ut eq '{state}'")
        
    if category := meta.get("category"):
        category = category.replace("'", "''")
        clauses.append(f"scheme_category eq '{category}'")
    
    bool_keys = {
        "is_for_women": "is_for_women",
        "is_for_farmers": "is_for_farmers",
        "is_for_disabled": "is_for_disabled",
        "is_for_sc_st": "is_for_sc_st",
        "is_for_students": "is_for_students",
    }
    for meta_key, field_name in bool_keys.items():
        if meta.get(meta_key):
            clauses.append(f"{field_name} eq true")

    if not clauses:
        return None
    
    return " and ".join(clauses)


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
