"""Cross-encoder reranker using Cohere or local BGE.

Takes a user query + candidate chunks → returns chunks re-ordered by
relevance score.

Falls back to local BGE if Cohere API key is not available.
"""
from __future__ import annotations

import os
import httpx
from rag.config import RERANKER_MODEL, FINAL_TOP_K
from rag.retriever import RetrievedChunk


class Reranker:
    """Reranker client using Cohere API or Local HuggingFace."""

    def __init__(self) -> None:
        self.cohere_api_key = os.getenv("COHERE_API_KEY", "")
        self._local_model = None
        
        if not self.cohere_api_key:
            from sentence_transformers import CrossEncoder
            self._local_model = CrossEncoder(
                RERANKER_MODEL,
                max_length=512,
                automodel_args={"torch_dtype": "auto"},
            )

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = FINAL_TOP_K,
    ) -> list[RetrievedChunk]:
        """Re-score all chunks and return top_k sorted by reranker score."""
        if not chunks:
            return []

        if self.cohere_api_key:
            return self._cohere_rerank(query, chunks, top_k)
        else:
            return self._local_rerank(query, chunks, top_k)

    def _cohere_rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        url = "https://api.cohere.ai/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Cohere expects a list of text strings
        documents = [chunk.text for chunk in chunks]
        
        payload = {
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": documents,
            "top_n": top_k
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            
            # Map scores back to original chunks
            reranked_chunks = []
            for res in results:
                idx = res["index"]
                score = res["relevance_score"]
                chunk = chunks[idx]
                chunk.score = float(score)
                reranked_chunks.append(chunk)
                
            return reranked_chunks
        except Exception as e:
            print(f"[Reranker] Cohere API failed: {e}. Falling back to default scores.")
            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks[:top_k]

    def _local_rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._local_model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]
