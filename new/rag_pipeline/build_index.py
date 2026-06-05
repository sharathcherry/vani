"""
One-time index builder.
Usage:  python -m rag_pipeline.build_index

Prerequisites:
  - Qdrant running: docker compose up -d
  - NVIDIA_API_KEY set in environment
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, OptimizersConfigDiff, PayloadSchemaType,
    PointStruct, VectorParams,
)
from tqdm import tqdm

from .bm25_store import BM25Store
from .config import (
    BM25_INDEX_PATH, CHUNK_TYPES, CHUNKS_DIR,
    COLLECTION_NAME, EMBED_BATCH_SIZE, EMBED_DIM,
    QDRANT_HOST, QDRANT_LOCAL_PATH, QDRANT_MODE, QDRANT_PORT, STRUCTURED_DIR,
)
from .embedder import embed


# ── 1. Load structured metadata ───────────────────────────────────────────────

def _load_structured_meta() -> Dict[str, Dict]:
    """Load all structured_schemes/*.json, keyed by scheme_id."""
    meta: Dict[str, Dict] = {}
    for p in Path(STRUCTURED_DIR).glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            sid = d.get("scheme_id")
            if sid:
                meta[sid] = d
        except Exception as e:
            print(f"  [warn] {p.name}: {e}")
    return meta


# ── 2. Load all chunks ────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _load_chunks(structured_meta: Dict[str, Dict]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for chunk_type in CHUNK_TYPES:
        type_dir = Path(CHUNKS_DIR) / chunk_type
        if not type_dir.exists():
            print(f"  [skip] {chunk_type}/ not found")
            continue
        txt_files = [p for p in sorted(type_dir.iterdir())
                     if p.suffix == ".txt" and not p.name.endswith(".metadata.json")]
        print(f"  {chunk_type}: {len(txt_files)} files")
        for txt_path in txt_files:
            meta_path = Path(str(txt_path) + ".metadata.json")
            if not meta_path.exists():
                continue
            try:
                text = txt_path.read_text(encoding="utf-8").strip()
                with open(meta_path, encoding="utf-8") as f:
                    raw = json.load(f)
                # Handle both flat and nested metadata formats
                attrs = raw.get("metadataAttributes", raw)
                sid = attrs.get("scheme_id", "")

                # Enrich from structured scheme if available
                extra: Dict[str, Any] = {}
                if sid and sid in structured_meta:
                    sm = structured_meta[sid]
                    extra = {
                        "meta_max_income_inr":  _safe_float(sm.get("meta_max_income_inr")),
                        "meta_min_age":         _safe_float(sm.get("meta_min_age")),
                        "meta_max_age":         _safe_float(sm.get("meta_max_age")),
                        "beneficiary_tags":     sm.get("beneficiary_tags", []),
                        "caste_criteria":       sm.get("caste_criteria", []),
                        "application_mode":     sm.get("application_mode", []),
                        "objective":            sm.get("objective", ""),
                    }

                chunks.append({
                    "text":              text,
                    "scheme_id":         sid,
                    "scheme_name":       attrs.get("scheme_name", ""),
                    "scheme_category":   attrs.get("scheme_category", ""),
                    "scheme_level":      attrs.get("scheme_level", ""),
                    "state_or_ut":       attrs.get("state_or_ut", ""),
                    "ministry":          attrs.get("ministry", ""),
                    "chunk_type":        attrs.get("chunk_type", chunk_type),
                    "qa_index":          attrs.get("qa_index"),
                    "is_for_women":      bool(attrs.get("is_for_women", False)),
                    "is_for_disabled":   bool(attrs.get("is_for_disabled", False)),
                    "is_for_sc_st":      bool(attrs.get("is_for_sc_st", False)),
                    "is_for_students":   bool(attrs.get("is_for_students", False)),
                    "is_for_farmers":    bool(attrs.get("is_for_farmers", False)),
                    "meta_max_amount_inr": _safe_float(attrs.get("meta_max_amount_inr")),
                    "meta_min_amount_inr": _safe_float(attrs.get("meta_min_amount_inr")),
                    "meta_max_age":      _safe_float(attrs.get("meta_max_age")),
                    **extra,
                })
            except Exception as e:
                print(f"  [warn] {txt_path.name}: {e}")

    return chunks


# ── 3. Build Qdrant index ─────────────────────────────────────────────────────

def _build_qdrant(chunks: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    if QDRANT_MODE == "local":
        client = QdrantClient(path=QDRANT_LOCAL_PATH)
        print(f"  Qdrant embedded mode → {QDRANT_LOCAL_PATH}")
    else:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"  Qdrant server mode → {QDRANT_HOST}:{QDRANT_PORT}")

    # Recreate collection
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        print(f"  Deleting existing collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
    )

    # Create payload indexes for fast metadata filtering
    for field in ["scheme_id", "state_or_ut", "scheme_category",
                  "chunk_type", "scheme_level"]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # Embed in batches
    texts = [c["text"] for c in chunks]
    all_vecs: List[List[float]] = []
    print(f"  Embedding {len(texts)} chunks with BGE-M3 (batches of {EMBED_BATCH_SIZE})...")
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE)):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        all_vecs.extend(embed(batch, sleep_s=0.3))

    # Upsert
    print(f"  Upserting {len(chunks)} points to Qdrant...")
    points = []
    for idx, (chunk, vec) in enumerate(zip(chunks, all_vecs)):
        payload = dict(chunk)   # text stored in payload for retrieval
        points.append(PointStruct(id=idx, vector=vec, payload=payload))

    for i in tqdm(range(0, len(points), 500)):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + 500])

    print(f"  ✓ Qdrant: {len(points)} points in '{COLLECTION_NAME}'")
    return [(str(idx), chunks[idx]["text"]) for idx in range(len(chunks))]


# ── 4. Build BM25 index ───────────────────────────────────────────────────────

def _build_bm25(id_text_pairs: List[Tuple[str, str]]) -> None:
    store = BM25Store()
    store.build([p[0] for p in id_text_pairs], [p[1] for p in id_text_pairs])
    store.save(BM25_INDEX_PATH)
    print(f"  ✓ BM25 index saved → {BM25_INDEX_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("━━━ [1/4] Loading structured metadata...")
    structured_meta = _load_structured_meta()
    print(f"  {len(structured_meta)} structured schemes loaded")

    print("\n━━━ [2/4] Loading chunks...")
    chunks = _load_chunks(structured_meta)
    print(f"  {len(chunks)} total chunks loaded")

    print("\n━━━ [3/4] Building Qdrant index...")
    id_text_pairs = _build_qdrant(chunks)

    print("\n━━━ [4/4] Building BM25 index...")
    _build_bm25(id_text_pairs)

    print("\n✓ All indexes built successfully!")


if __name__ == "__main__":
    main()
