import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# ── NVIDIA API ────────────────────────────────────────────────────────────────
NVIDIA_API_KEY   = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL  = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL      = "baai/bge-m3"
EMBED_DIM        = 1024
EMBED_BATCH_SIZE = 32          # stay under NVIDIA rate limits
RERANK_MODEL     = "nv-rerank-qa-mistral-4b:1"
LLM_MODEL        = "meta/llama-3.1-70b-instruct"

# ── Qdrant ────────────────────────────────────────────────────────────────────
# QDRANT_MODE:
#   "local"  → embedded mode, data saved to QDRANT_LOCAL_PATH (no Docker needed)
#   "server" → connects to a running Qdrant server (Docker / AWS EC2)
QDRANT_MODE       = os.environ.get("QDRANT_MODE", "local")
QDRANT_LOCAL_PATH = str(BASE_DIR / "qdrant_storage")   # used when mode=local
QDRANT_HOST       = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME   = "gov_schemes"

# ── Data paths ────────────────────────────────────────────────────────────────
CHUNKS_DIR       = BASE_DIR / "bedrock_chunks"
STRUCTURED_DIR   = BASE_DIR / "structured_schemes"
INDEX_DIR        = BASE_DIR / "index"
BM25_INDEX_PATH  = INDEX_DIR / "bm25_index.pkl"

# ── Retrieval parameters ──────────────────────────────────────────────────────
CHUNK_TYPES   = ["qa", "summary", "eligibility", "benefits", "application"]
BOOST_WEIGHTS = {"qa": 1.5, "summary": 1.2, "eligibility": 1.0,
                 "benefits": 1.0, "application": 1.0}
RRF_K          = 60
VECTOR_TOP_K   = 50
BM25_TOP_K     = 50
RERANK_TOP_K   = 20   # candidates sent to reranker
FINAL_TOP_K    = 5
N_QUERY_VARIANTS = 3
MAX_CONTEXT_CHARS = 3000
