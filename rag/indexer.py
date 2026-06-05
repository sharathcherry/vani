"""Build Azure AI Search index + BM25 sidecar from bedrock_chunks/.

Phase 2 migration: Qdrant replaced by Azure AI Search for primary hybrid
retrieval. The BM25 sidecar (rank_bm25 pickle) is still produced because
several offline scripts (golden-set evaluators, debug probes) lean on it.

Index schema (also see ``_build_index_schema``):
  id              Edm.String        key
  content         Edm.String        searchable, analyzer=standard.lucene
  embedding       Collection(Single) searchable=true, vector_search_dimensions=EMBED_DIM
  scheme_name     Edm.String        searchable + filterable + facetable
  scheme_id       Edm.String        filterable
  scheme_category Edm.String        filterable + facetable
  scheme_level    Edm.String        filterable
  ministry        Edm.String        filterable
  chunk_type      Edm.String        filterable + facetable
  state_or_ut     Edm.String        filterable + facetable
  eligibility     Edm.String        searchable (semantic config keyword)
  link            Edm.String        retrievable
  is_for_women    Edm.Boolean       filterable
  is_for_disabled Edm.Boolean       filterable
  is_for_sc_st    Edm.Boolean       filterable
  is_for_students Edm.Boolean       filterable
  is_for_farmers  Edm.Boolean       filterable
  max_amount_inr  Edm.Double        filterable + sortable
  qa_index        Edm.Int32         filterable

A ``semanticConfiguration`` named AZURE_SEARCH_SEMANTIC_CONFIG (default:
``default``) is attached. ``content`` is the prioritized field; ``scheme_name``
provides the title slot.

Run once:      python -m rag.indexer
Force rebuild: python -m rag.indexer --force
"""
from __future__ import annotations

import json
import pickle
import sys
import uuid
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

from rag.config import (
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX,
    AZURE_SEARCH_SEMANTIC_CONFIG,
    BM25_INDEX_PATH,
    CORPUS_PATH,
    DATA_DIR,
    EMBED_DIM,
    INDEX_DIR,
)
from rag.embedder import Embedder

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_DIR = BASE_DIR / "bedrock_chunks"
PARENT_DOCS_PATH = str(INDEX_DIR / "parent_docs.pkl")

CHUNK_SUBDIRS = ["summary", "eligibility", "benefits", "application", "qa", "new"]

# Azure AI Search caps upload_documents at 1000 docs OR 16 MB per batch.
# 500 keeps payload well under both limits for our chunk sizes.
_UPLOAD_BATCH_SIZE = 500
_HNSW_PROFILE = "hnsw-default"
_HNSW_ALGO = "hnsw-config"


# ── Client factories ─────────────────────────────────────────────────────────

def _credential() -> AzureKeyCredential:
    if not AZURE_SEARCH_API_KEY:
        raise ValueError(
            "AZURE_SEARCH_API_KEY is not set. Add it to .env (never commit)."
        )
    return AzureKeyCredential(AZURE_SEARCH_API_KEY)


def make_index_client() -> SearchIndexClient:
    if not AZURE_SEARCH_ENDPOINT:
        raise ValueError("AZURE_SEARCH_ENDPOINT is not set in .env.")
    return SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=_credential())


def make_search_client(index_name: str = AZURE_SEARCH_INDEX) -> SearchClient:
    if not AZURE_SEARCH_ENDPOINT:
        raise ValueError("AZURE_SEARCH_ENDPOINT is not set in .env.")
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=_credential(),
    )


# ── Schema definition ────────────────────────────────────────────────────────

def _build_index_schema(index_name: str) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),

        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="standard.lucene",
        ),

        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name=_HNSW_PROFILE,
            hidden=True,  # Do not return vector bytes in normal responses.
        ),

        # Filterable / faceted metadata
        SearchableField(name="scheme_name", type=SearchFieldDataType.String,
                        filterable=True, facetable=True),
        SimpleField(name="scheme_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="scheme_category", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="scheme_level", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="ministry", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_type", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="state_or_ut", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),

        SearchableField(name="eligibility", type=SearchFieldDataType.String,
                        filterable=True),
        SimpleField(name="link", type=SearchFieldDataType.String, retrievable=True),

        # Beneficiary boolean flags
        SimpleField(name="is_for_women", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="is_for_disabled", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="is_for_sc_st", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="is_for_students", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="is_for_farmers", type=SearchFieldDataType.Boolean, filterable=True),

        # Numerics
        SimpleField(name="max_amount_inr", type=SearchFieldDataType.Double,
                    filterable=True, sortable=True),
        SimpleField(name="qa_index", type=SearchFieldDataType.Int32, filterable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=_HNSW_ALGO,
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=16,
                    ef_construction=200,
                    ef_search=200,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name=_HNSW_PROFILE,
                algorithm_configuration_name=_HNSW_ALGO,
            ),
        ],
    )

    semantic_search = SemanticSearch(configurations=[
        SemanticConfiguration(
            name=AZURE_SEARCH_SEMANTIC_CONFIG,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="scheme_name"),
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[
                    SemanticField(field_name="eligibility"),
                    SemanticField(field_name="scheme_category"),
                    SemanticField(field_name="state_or_ut"),
                ],
            ),
        ),
    ])

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def ensure_index(index_client: SearchIndexClient | None = None,
                 index_name: str = AZURE_SEARCH_INDEX) -> None:
    """Idempotently create or update the Azure AI Search index.

    Uses ``create_or_update_index`` so re-runs do not crash. Existing
    documents are preserved if the schema is compatible.
    """
    index_client = index_client or make_index_client()
    schema = _build_index_schema(index_name)
    index_client.create_or_update_index(schema)
    print(f"[indexer] Azure AI Search index ensured: {index_name}")


def delete_index_if_exists(index_client: SearchIndexClient | None = None,
                           index_name: str = AZURE_SEARCH_INDEX) -> None:
    """Used only when --force flag is passed."""
    index_client = index_client or make_index_client()
    try:
        index_client.delete_index(index_name)
        print(f"[indexer] Deleted existing index: {index_name}")
    except ResourceNotFoundError:
        pass


# ── Parent docs + chunk loaders (unchanged shape) ────────────────────────────

def _load_parent_docs() -> dict[str, dict]:
    """Build scheme_id → full parent doc dict from structured_schemes/."""
    parent_docs: dict[str, dict] = {}
    for path in DATA_DIR.glob("*.json"):
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = s.get("scheme_id", "")
        if not sid:
            continue
        parent_docs[sid] = {
            "scheme_id":       sid,
            "scheme_name":     s.get("scheme_name", ""),
            "state_or_ut":     s.get("state_or_ut", ""),
            "scheme_category": s.get("scheme_category", ""),
            "scheme_level":    s.get("scheme_level", ""),
            "ministry":        s.get("ministry_or_department", ""),
            "summary":         s.get("text_summary", ""),
            "eligibility":     s.get("text_eligibility", ""),
            "benefits":        s.get("text_benefits", ""),
            "application":     s.get("text_application", ""),
            "objective":       s.get("objective", ""),
            "link":             s.get("link", ""),
        }
    return parent_docs


def _load_all_chunks() -> list[dict]:
    seen: set[str] = set()
    chunks: list[dict] = []

    for subdir in CHUNK_SUBDIRS:
        subdir_path = CHUNKS_DIR / subdir
        if not subdir_path.exists():
            continue

        for txt_path in sorted(f for f in subdir_path.iterdir() if f.suffix == ".txt"):
            meta_path = Path(str(txt_path) + ".metadata.json")
            if not meta_path.exists():
                continue
            try:
                text     = txt_path.read_text(encoding="utf-8").strip()
                meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not text:
                continue

            attrs       = meta_raw.get("metadataAttributes", {})
            scheme_id   = attrs.get("scheme_id", "")
            scheme_name = attrs.get("scheme_name", "")
            chunk_type  = attrs.get("chunk_type", subdir)
            qa_index    = attrs.get("qa_index", -1)
            try:
                qa_index = int(qa_index)
            except (TypeError, ValueError):
                qa_index = -1

            if not scheme_id:
                continue

            dedup_key = f"{scheme_id}::{chunk_type}::{qa_index}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{scheme_id}::{chunk_type}::{qa_index}",
            ))

            def _b(key: str) -> bool:
                v = attrs.get(key, False)
                return v if isinstance(v, bool) else str(v).lower() == "true"

            def _f(key: str):
                v = attrs.get(key)
                if v is None or v in ("", "null", "None"):
                    return None
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None

            chunks.append({
                "id":              point_id,
                "text":            text,
                "chunk_type":      chunk_type,
                "scheme_id":       scheme_id,
                "scheme_name":     scheme_name,
                "qa_index":        qa_index,
                "state_or_ut":     attrs.get("state_or_ut", ""),
                "scheme_category": attrs.get("scheme_category", ""),
                "scheme_level":    attrs.get("scheme_level", ""),
                "ministry":        attrs.get("ministry", ""),
                "link":            attrs.get("link", ""),
                "eligibility":     attrs.get("eligibility_text", ""),
                "is_for_women":    _b("is_for_women"),
                "is_for_disabled": _b("is_for_disabled"),
                "is_for_sc_st":    _b("is_for_sc_st"),
                "is_for_students": _b("is_for_students"),
                "is_for_farmers":  _b("is_for_farmers"),
                "max_amount_inr":     _f("meta_max_amount_inr"),
            })

    return chunks


# ── Chunk → Azure AI Search document mapping ────────────────────────────────

def chunk_to_search_document(chunk: dict, vector: list[float] | None) -> dict:
    """Map a chunk dict (+ its dense embedding) to an Azure AI Search document.

    Azure AI Search rejects null for numeric / boolean fields; coerce or drop.
    """
    doc: dict = {
        "id":              chunk["id"],
        "content":         chunk["text"],
        "chunk_type":      chunk.get("chunk_type", ""),
        "scheme_id":       chunk.get("scheme_id", ""),
        "scheme_name":     chunk.get("scheme_name", ""),
        "scheme_category": chunk.get("scheme_category", ""),
        "scheme_level":    chunk.get("scheme_level", ""),
        "ministry":        chunk.get("ministry", ""),
        "state_or_ut":     chunk.get("state_or_ut", ""),
        "eligibility":     chunk.get("eligibility", ""),
        "link":            chunk.get("link", ""),
        "is_for_women":    bool(chunk.get("is_for_women", False)),
        "is_for_disabled": bool(chunk.get("is_for_disabled", False)),
        "is_for_sc_st":    bool(chunk.get("is_for_sc_st", False)),
        "is_for_students": bool(chunk.get("is_for_students", False)),
        "is_for_farmers":  bool(chunk.get("is_for_farmers", False)),
        "qa_index":        int(chunk.get("qa_index") or -1),
    }
    max_amt = chunk.get("max_amount_inr")
    if max_amt is not None:
        doc["max_amount_inr"] = float(max_amt)
    if vector is not None:
        doc["embedding"] = list(vector)
    return doc


def upload_documents(
    search_client: SearchClient,
    documents: list[dict],
    batch_size: int = _UPLOAD_BATCH_SIZE,
) -> tuple[int, int]:
    """Batch upload documents. Returns (success_count, failure_count)."""
    success = 0
    failures = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        try:
            results = search_client.upload_documents(documents=batch)
            for r in results:
                if r.succeeded:
                    success += 1
                else:
                    failures += 1
                    print(f"[indexer] Upload failed key={r.key} reason={r.error_message}")
        except HttpResponseError as exc:
            failures += len(batch)
            print(f"[indexer] Batch upload error: {exc}")
    return success, failures


# ── Public orchestration ─────────────────────────────────────────────────────

def build_index(force: bool = False) -> None:
    index_client = make_index_client()

    # ── 1. Parent docs ────────────────────────────────────────────────────
    print("[indexer] Loading parent documents from structured_schemes/ ...")
    parent_docs = _load_parent_docs()
    print(f"[indexer] {len(parent_docs)} parent docs loaded.")
    with open(PARENT_DOCS_PATH, "wb") as f:
        pickle.dump(parent_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── 2. Load all pre-made chunks ───────────────────────────────────────
    print("[indexer] Scanning bedrock_chunks/ ...")
    chunks = _load_all_chunks()
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1
    for ctype, cnt in sorted(by_type.items()):
        print(f"  {ctype:15}: {cnt:5} chunks")
    print(f"  {'TOTAL':15}: {len(chunks):5} chunks")

    # ── 3. (Re)create Azure AI Search index ───────────────────────────────
    if force:
        delete_index_if_exists(index_client)
    ensure_index(index_client)

    # ── 4. Embed + upsert ─────────────────────────────────────────────────
    print("[indexer] Embedding all chunks ...")
    embedder = Embedder()
    vectors = embedder.embed_passages([c["text"] for c in chunks])

    print("[indexer] Uploading documents to Azure AI Search ...")
    docs = [chunk_to_search_document(c, v.tolist()) for c, v in zip(chunks, vectors)]
    search_client = make_search_client()
    success, failures = upload_documents(search_client, docs)
    print(f"[indexer] Azure upload: {success} ok, {failures} failed.")

    # ── 5. BM25 sidecar (offline evaluators still need it) ────────────────
    print("[indexer] Building BM25Okapi sidecar index ...")
    from rank_bm25 import BM25Okapi

    corpus = [
        {
            "id":              c["id"],
            "scheme_id":       c["scheme_id"],
            "scheme_name":     c["scheme_name"],
            "chunk_type":      c["chunk_type"],
            "text":            c["text"],
            "state_or_ut":     c["state_or_ut"],
            "scheme_category": c["scheme_category"],
            "is_for_women":    c["is_for_women"],
            "is_for_disabled": c["is_for_disabled"],
            "is_for_sc_st":    c["is_for_sc_st"],
            "is_for_students": c["is_for_students"],
            "is_for_farmers":  c["is_for_farmers"],
            "max_amount_inr":  c["max_amount_inr"],
        }
        for c in chunks
    ]
    tokenized = [item["text"].lower().split() for item in corpus]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"\n[indexer] Done.\n"
        f"  Chunks  : {len(chunks)}\n"
        f"  Schemes : {len(parent_docs)}\n"
        f"  AISearch: {AZURE_SEARCH_ENDPOINT} :: index={AZURE_SEARCH_INDEX}\n"
        f"  BM25    : {BM25_INDEX_PATH}\n"
        f"  Parents : {PARENT_DOCS_PATH}"
    )


if __name__ == "__main__":
    build_index(force="--force" in sys.argv)
