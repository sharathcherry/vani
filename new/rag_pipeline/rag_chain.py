import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import (
    FINAL_TOP_K, LLM_MODEL, MAX_CONTEXT_CHARS,
    NVIDIA_API_KEY, NVIDIA_BASE_URL, RERANK_TOP_K,
)
from .query_processor import expand_query, generate_query_variants, rewrite_query
from .reranker import rerank
from .retriever import HybridRetriever

_SYSTEM_PROMPT = """You are a helpful assistant specialising in Indian government welfare schemes.
Answer the user's question based ONLY on the provided scheme information.
Be specific and concise. If multiple schemes are relevant, name each one.
If the answer cannot be found in the provided context, say so clearly.
Never invent eligibility criteria, amounts, or portal links."""


def _compress(chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Keep the most query-relevant sentences from each chunk (semantic compression)."""
    query_words = set(re.sub(r"[^a-z0-9\s]", "", query.lower()).split())
    compressed, total = [], 0
    for chunk in chunks:
        sentences = re.split(r"(?<=[.?!])\s+", chunk.get("text", ""))
        scored = sorted(
            sentences,
            key=lambda s: sum(1 for w in s.lower().split() if w in query_words),
            reverse=True,
        )
        excerpt = " ".join(scored[:5])[:900]
        if total + len(excerpt) > MAX_CONTEXT_CHARS:
            break
        compressed.append({**chunk, "text": excerpt})
        total += len(excerpt)
    return compressed


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        name  = c.get("scheme_name", "Unknown")
        cat   = c.get("scheme_category", "")
        state = c.get("state_or_ut", "")
        parts.append(f"[{i}] {name} ({cat}, {state})\n{c.get('text', '')}")
    return "\n\n".join(parts)


def answer(
    query: str,
    retriever: HybridRetriever,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = FINAL_TOP_K,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    # 1. Query rewriting
    rewritten = rewrite_query(query)

    # 2. Query expansion (synonym dict)
    expanded = expand_query(rewritten)

    # 3. Multi-query variant generation
    variants = generate_query_variants(rewritten)
    all_queries = [expanded] + variants

    # 4. Hybrid retrieval — returns RERANK_TOP_K deduplicated, boosted candidates
    candidates = retriever.retrieve(
        primary_query=expanded,
        all_queries=all_queries,
        filters=filters,
        top_k=RERANK_TOP_K,
    )

    # 5. Reranking
    if use_reranker and candidates:
        top_chunks = rerank(query, candidates, top_k=top_k)
    else:
        top_chunks = candidates[:top_k]

    # 6. Semantic chunk compression
    compressed = _compress(top_chunks, query)

    # 7. LLM answer
    context = _format_context(compressed)
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=800,
        temperature=0.1,
    )

    return {
        "query": query,
        "rewritten_query": rewritten,
        "query_variants": variants,
        "filters_applied": filters,
        "candidates_retrieved": len(candidates),
        "top_schemes": [c.get("scheme_name") for c in top_chunks],
        "answer": resp.choices[0].message.content.strip(),
    }
