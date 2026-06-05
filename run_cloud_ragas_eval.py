"""
run_cloud_ragas_eval.py
=======================
Fresh RAGAS evaluation against the cloud Qdrant DB.

Steps:
  1. Load QA samples from qdrant_eval_30sample.json (or 200 with --samples 200)
  2. Connect to cloud Qdrant, encode each question with fastembed
  3. Hybrid search (dense bge-large + sparse bm25, RRF fusion)
  4. Generate answer via Azure OpenAI gpt-4o
  5. Run RAGAS 0.4.x with Azure OpenAI as judge
  6. Save results to cloud_ragas_results_<timestamp>.json

Usage:
  python run_cloud_ragas_eval.py                   # 30 samples
  python run_cloud_ragas_eval.py --samples 200     # full 200-sample run
  python run_cloud_ragas_eval.py --samples 10      # quick sanity check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL    = os.getenv("QDRANT_CLOUD_URL",    "https://126e5307-b4ed-441a-b955-bcec6c7f7d7b.eu-central-1-0.aws.cloud.qdrant.io:6333")
QDRANT_APIKEY = os.getenv("QDRANT_CLOUD_API_KEY", "")
COLLECTION    = os.getenv("QDRANT_COLLECTION",   "schemes_hybrid")

DENSE_MODEL  = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_K      = 50
SPARSE_K     = 50
TOP_K        = 8

AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT",       "")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY",        "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION",    "2024-08-01-preview")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT","gpt-4o")

NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_LLM_MODEL  = "meta/llama-3.3-70b-instruct"
NVIDIA_EMB_MODEL  = "nvidia/nv-embedqa-e5-v5"

HERE = Path(__file__).parent
EVAL_30  = HERE / "qdrant_eval_30sample.json"
EVAL_200 = HERE / "qdrant_eval_200sample.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_samples(n: int) -> list[dict]:
    src = EVAL_200 if n > 30 else EVAL_30
    data = json.loads(src.read_text(encoding="utf-8"))
    per_sample = data.get("per_sample", [])
    return [
        {
            "user_input": s["user_input"],
            "reference":  s.get("reference", ""),
        }
        for s in per_sample[:n]
    ]


def build_clients():
    from qdrant_client import QdrantClient
    from fastembed import TextEmbedding, SparseTextEmbedding

    print(f"  Connecting to Qdrant cloud: {QDRANT_URL}")
    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_APIKEY, timeout=30)
    info = qclient.get_collection(COLLECTION)
    print(f"  Collection '{COLLECTION}' — {info.points_count} points")

    print(f"  Loading dense embedder: {DENSE_MODEL}")
    dense_enc = TextEmbedding(DENSE_MODEL)
    print(f"  Loading sparse embedder: {SPARSE_MODEL}")
    sparse_enc = SparseTextEmbedding(SPARSE_MODEL)

    return qclient, dense_enc, sparse_enc


def retrieve(question: str, qclient, dense_enc, sparse_enc) -> list[str]:
    from qdrant_client import models

    dvec = list(dense_enc.embed([question]))[0].tolist()
    svec_raw = list(sparse_enc.embed([question]))[0]
    svec = models.SparseVector(indices=svec_raw.indices.tolist(),
                               values=svec_raw.values.tolist())

    result = qclient.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dvec, using="dense",  limit=DENSE_K),
            models.Prefetch(query=svec, using="sparse", limit=SPARSE_K),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=TOP_K,
        with_payload=True,
    )
    return [p.payload.get("text", "") for p in result.points if p.payload.get("text")]


def generate_answer(question: str, contexts: list[str]) -> str:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )
    ctx_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts[:5]))
    system = (
        "You are a helpful government scheme advisor for Indian citizens. "
        "Answer the question using ONLY the provided context chunks. "
        "Be concise (2-4 sentences)."
    )
    user = f"Context:\n{ctx_block}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.0,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def run_ragas(samples_data: list[dict], judge: str = "nvidia") -> dict:
    from ragas import evaluate as ragas_evaluate, EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

    if judge == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
        llm_lc = ChatNVIDIA(
            model=NVIDIA_LLM_MODEL,
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        emb_lc = NVIDIAEmbeddings(
            model=NVIDIA_EMB_MODEL,
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        print(f"  RAGAS judge LLM : NVIDIA {NVIDIA_LLM_MODEL}")
        print(f"  RAGAS judge emb : NVIDIA {NVIDIA_EMB_MODEL}")
    else:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        llm_lc = AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            azure_deployment=AZURE_DEPLOYMENT,
            api_version=AZURE_API_VERSION,
            temperature=0.0,
            max_tokens=1024,
        )
        emb_lc = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            azure_deployment="text-embedding-ada-002",
            api_version=AZURE_API_VERSION,
        )
        print(f"  RAGAS judge LLM : Azure {AZURE_DEPLOYMENT}")
        print(f"  RAGAS judge emb : Azure text-embedding-ada-002")

    ragas_llm = LangchainLLMWrapper(llm_lc)
    ragas_emb = LangchainEmbeddingsWrapper(emb_lc)

    faithfulness.llm        = ragas_llm
    answer_relevancy.llm    = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm   = ragas_llm
    context_recall.llm      = ragas_llm

    ragas_samples = [
        SingleTurnSample(
            user_input         = s["user_input"],
            response           = s["response"],
            retrieved_contexts = s["retrieved_contexts"],
            reference          = s["reference"],
        )
        for s in samples_data
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    from ragas.run_config import RunConfig
    run_cfg = RunConfig(timeout=600, max_retries=5, max_wait=60, max_workers=2)

    result = ragas_evaluate(
        dataset,
        metrics          = [faithfulness, answer_relevancy, context_precision, context_recall],
        llm              = ragas_llm,
        embeddings       = ragas_emb,
        run_config       = run_cfg,
        raise_exceptions = False,
    )
    df = result.to_pandas()
    agg = df.mean(numeric_only=True).to_dict()
    scores = {k: round(float(v), 4) for k, v in agg.items() if not math.isnan(v)}
    per = df.to_dict("records")
    return scores, per


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30,
                        help="Number of QA samples to evaluate (default: 30)")
    parser.add_argument("--judge", choices=["nvidia", "azure"], default="nvidia",
                        help="RAGAS judge provider (default: nvidia)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Cloud Qdrant RAGAS Evaluation")
    print(f"  Collection: {COLLECTION}")
    print(f"  Samples:    {args.samples}")
    print("="*60 + "\n")

    # 1. Load QA pairs
    samples = load_samples(args.samples)
    print(f"Loaded {len(samples)} QA pairs from eval dataset\n")

    # 2. Build clients
    qclient, dense_enc, sparse_enc = build_clients()
    print()

    # 3. Retrieve + generate
    print("Running retrieval + answer generation...")
    enriched = []
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        q = s["user_input"]
        print(f"  [{i:3}/{len(samples)}] {q[:70]}...", end=" ", flush=True)
        try:
            ctxs = retrieve(q, qclient, dense_enc, sparse_enc)
            ans  = generate_answer(q, ctxs) if ctxs else "No relevant context found."
            enriched.append({
                "user_input":         q,
                "reference":          s["reference"],
                "retrieved_contexts": ctxs,
                "response":           ans,
            })
            print(f"OK ({len(ctxs)} chunks)")
        except Exception as exc:
            print(f"ERROR: {exc}")
            enriched.append({
                "user_input":         q,
                "reference":          s["reference"],
                "retrieved_contexts": [],
                "response":           "",
            })
    retrieve_time = time.time() - t0
    print(f"\nRetrieval + generation done in {retrieve_time:.1f}s\n")

    # Filter out empty-context samples before RAGAS
    valid = [s for s in enriched if s["retrieved_contexts"] and s["response"]]
    print(f"Valid samples for RAGAS: {len(valid)}/{len(enriched)}\n")

    # 4. RAGAS evaluation
    print("Running RAGAS evaluation (faithfulness / answer_relevancy / context_precision / context_recall)...")
    t1 = time.time()
    scores, per_sample = run_ragas(valid, judge=args.judge)
    ragas_time = time.time() - t1

    # 5. Print results
    print("\n" + "="*60)
    print("  RAGAS EVALUATION RESULTS")
    print("="*60)
    metric_labels = {
        "faithfulness"     : "Faithfulness       (0-1)",
        "answer_relevancy" : "Answer Relevancy   (0-1)",
        "context_precision": "Context Precision  (0-1)",
        "context_recall"   : "Context Recall     (0-1)",
    }
    for k, label in metric_labels.items():
        v = scores.get(k)
        if v is not None:
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"  {label}: {v:.4f}  {bar}")
    if scores:
        composite = sum(scores.values()) / len(scores)
        print(f"\n  Composite Average: {composite:.4f}")
    print(f"\n  RAGAS evaluation time: {ragas_time:.1f}s")
    print(f"  Total elapsed:         {time.time() - t0:.1f}s")

    # 6. Save
    out = {
        "timestamp"       : datetime.now().isoformat(),
        "config": {
            "collection"  : COLLECTION,
            "qdrant_url"  : QDRANT_URL,
            "dense_model" : DENSE_MODEL,
            "sparse_model": SPARSE_MODEL,
            "top_k"       : TOP_K,
            "judge_llm"   : f"{args.judge}:{NVIDIA_LLM_MODEL if args.judge == 'nvidia' else AZURE_DEPLOYMENT}",
            "num_samples" : len(valid),
        },
        "aggregate_scores": scores,
        "elapsed_seconds" : round(time.time() - t0, 2),
        "per_sample"      : per_sample,
    }
    out_path = HERE / f"cloud_ragas_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved: {out_path}\n")


if __name__ == "__main__":
    main()
