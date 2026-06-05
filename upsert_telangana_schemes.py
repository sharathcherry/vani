"""Upsert the Telangana scheme corpus into Azure AI Search.

Phase 2 replacement for the old Qdrant upsert path. Reads structured scheme
JSON files matching ``Telangana*.json`` from ``structured_schemes/``,
re-uses the chunk loader / parent-doc loader from ``rag.indexer``, embeds
each chunk with the BGE-m3 embedder, and uploads to Azure AI Search via
``SearchClient.upload_documents`` in batched calls.

Usage:
    python upsert_telangana_schemes.py
    python upsert_telangana_schemes.py --force   # delete + recreate index first

Required env vars (see ``.env.example``):
    AZURE_SEARCH_ENDPOINT
    AZURE_SEARCH_API_KEY       # NEVER commit
    AZURE_SEARCH_INDEX
"""
from __future__ import annotations

import sys
from pathlib import Path

from rag.config import AZURE_SEARCH_INDEX
from rag.embedder import Embedder
from rag.indexer import (
    _load_all_chunks,
    _load_parent_docs,
    chunk_to_search_document,
    delete_index_if_exists,
    ensure_index,
    make_index_client,
    make_search_client,
    upload_documents,
)

TELANGANA_NEEDLE = "telangana"


def _is_telangana(chunk: dict) -> bool:
    state = (chunk.get("state_or_ut") or "").lower()
    name = (chunk.get("scheme_name") or "").lower()
    sid = (chunk.get("scheme_id") or "").lower()
    return (
        TELANGANA_NEEDLE in state
        or TELANGANA_NEEDLE in name
        or TELANGANA_NEEDLE in sid
    )


def run(force: bool = False) -> int:
    print("[upsert] Loading parent documents ...")
    parent_docs = _load_parent_docs()
    telangana_schemes = [d for d in parent_docs.values()
                         if TELANGANA_NEEDLE in (d.get("state_or_ut", "") or "").lower()
                         or TELANGANA_NEEDLE in (d.get("scheme_name", "") or "").lower()]
    print(f"[upsert] {len(telangana_schemes)} Telangana parent schemes found.")

    print("[upsert] Loading chunks (bedrock_chunks/) ...")
    all_chunks = _load_all_chunks()
    chunks = [c for c in all_chunks if _is_telangana(c)]
    print(f"[upsert] {len(chunks)} Telangana chunks selected from {len(all_chunks)} total.")

    if not chunks:
        print("[upsert] Nothing to upsert. Confirm bedrock_chunks/ contains Telangana data.")
        return 0

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1
    for ctype, cnt in sorted(by_type.items()):
        print(f"  {ctype:15}: {cnt:5} chunks")

    index_client = make_index_client()
    if force:
        delete_index_if_exists(index_client, AZURE_SEARCH_INDEX)
    ensure_index(index_client, AZURE_SEARCH_INDEX)

    print("[upsert] Embedding Telangana chunks ...")
    embedder = Embedder()
    vectors = embedder.embed_passages([c["text"] for c in chunks])

    print("[upsert] Uploading to Azure AI Search ...")
    documents = [chunk_to_search_document(c, v.tolist()) for c, v in zip(chunks, vectors)]
    search_client = make_search_client(AZURE_SEARCH_INDEX)
    success, failures = upload_documents(search_client, documents)
    print(f"[upsert] Done. uploaded={success} failed={failures} index={AZURE_SEARCH_INDEX}")
    return failures


if __name__ == "__main__":
    sys.exit(run(force="--force" in sys.argv))
