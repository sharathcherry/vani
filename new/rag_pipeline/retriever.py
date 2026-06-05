from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient

from .bm25_store import BM25Store
from .config import (
    BM25_TOP_K, BOOST_WEIGHTS, COLLECTION_NAME, FINAL_TOP_K,
    QDRANT_HOST, QDRANT_LOCAL_PATH, QDRANT_MODE, QDRANT_PORT,
    RERANK_TOP_K, RRF_K, VECTOR_TOP_K,
)
from .embedder import embed_query


def _make_qdrant_client() -> QdrantClient:
    if QDRANT_MODE == "local":
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ── RRF & scoring helpers ─────────────────────────────────────────────────────

def _rrf(ranks: List[int], k: int = RRF_K) -> float:
    return sum(1.0 / (k + r) for r in ranks)


def _rrf_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    rank_map: Dict[str, List[int]] = {}
    for rank, (doc_id, _) in enumerate(vector_results):
        rank_map.setdefault(doc_id, []).append(rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        rank_map.setdefault(doc_id, []).append(rank + 1)
    fused = [(doc_id, _rrf(ranks)) for doc_id, ranks in rank_map.items()]
    return sorted(fused, key=lambda x: x[1], reverse=True)


def _apply_boost(
    fused: List[Tuple[str, float]],
    payloads: Dict[str, Dict],
) -> List[Tuple[str, float]]:
    boosted = []
    for doc_id, score in fused:
        chunk_type = payloads.get(doc_id, {}).get("chunk_type", "qa")
        boost = BOOST_WEIGHTS.get(chunk_type, 1.0)
        boosted.append((doc_id, score * boost))
    return sorted(boosted, key=lambda x: x[1], reverse=True)


def _deduplicate_by_scheme(
    results: List[Tuple[str, float]],
    payloads: Dict[str, Dict],
) -> List[Tuple[str, float]]:
    """Keep the top-scoring chunk per scheme_id."""
    seen: set = set()
    deduped = []
    for doc_id, score in results:
        sid = payloads.get(doc_id, {}).get("scheme_id", doc_id)
        if sid not in seen:
            seen.add(sid)
            deduped.append((doc_id, score))
    return deduped


def _apply_metadata_filter(
    results: List[Tuple[str, float]],
    payloads: Dict[str, Dict],
    filters: Optional[Dict[str, Any]],
) -> List[Tuple[str, float]]:
    if not filters:
        return results
    filtered = []
    for doc_id, score in results:
        p = payloads.get(doc_id, {})
        match = True
        for key, value in filters.items():
            if key == "state_or_ut":
                # Central schemes apply to everyone
                if p.get("state_or_ut") not in (value, "Central"):
                    match = False
            elif key in ("is_for_women", "is_for_disabled", "is_for_sc_st",
                         "is_for_students", "is_for_farmers"):
                if value and not p.get(key, False):
                    match = False
            elif key == "scheme_category":
                if p.get("scheme_category") != value:
                    match = False
            if not match:
                break
        if match:
            filtered.append((doc_id, score))
    return filtered


# ── Main retriever class ──────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self, bm25_store: BM25Store):
        self.qdrant = _make_qdrant_client()
        self.bm25 = bm25_store

    def retrieve(
        self,
        primary_query: str,           # expanded primary query (for BM25)
        all_queries: List[str],       # ALL variants (for multi-query vector search)
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = RERANK_TOP_K,
    ) -> List[Dict[str, Any]]:
        # ── Step 1: Multi-query vector search, merge via inter-query RRF ──────
        multi_ranks: Dict[str, List[int]] = {}
        for q in all_queries:
            vec = embed_query(q)
            hits = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=vec,
                limit=VECTOR_TOP_K,
                with_payload=True,
            )
            for rank, hit in enumerate(hits):
                multi_ranks.setdefault(str(hit.id), []).append(rank + 1)

        vector_results = sorted(
            [(doc_id, _rrf(ranks)) for doc_id, ranks in multi_ranks.items()],
            key=lambda x: x[1], reverse=True,
        )[:VECTOR_TOP_K]

        # ── Step 2: BM25 search ───────────────────────────────────────────────
        bm25_results = self.bm25.search(primary_query, top_k=BM25_TOP_K)

        # ── Step 3: RRF fusion ────────────────────────────────────────────────
        fused = _rrf_fusion(vector_results, bm25_results)

        # ── Step 4: Fetch payloads for top 200 candidates ────────────────────
        candidate_ids = [doc_id for doc_id, _ in fused[:200]]
        payloads = self._fetch_payloads(candidate_ids)

        # ── Step 5: Metadata filter ───────────────────────────────────────────
        fused = _apply_metadata_filter(fused, payloads, filters)

        # ── Step 6: Document boosting (qa=1.5x, summary=1.2x, etc.) ──────────
        fused = _apply_boost(fused, payloads)

        # ── Step 7: Scheme-level deduplication ───────────────────────────────
        fused = _deduplicate_by_scheme(fused, payloads)

        # ── Step 8: Return top_k with full payload ────────────────────────────
        return [
            {"id": doc_id, "score": score, **payloads.get(doc_id, {})}
            for doc_id, score in fused[:top_k]
        ]

    def _fetch_payloads(self, ids: List[str]) -> Dict[str, Dict]:
        if not ids:
            return {}
        int_ids = []
        for i in ids:
            try:
                int_ids.append(int(i))
            except ValueError:
                pass
        if not int_ids:
            return {}
        points = self.qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=int_ids,
            with_payload=True,
        )
        return {str(p.id): p.payload for p in points}
