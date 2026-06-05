"""
evaluate_bedrock_pipeline.py
------------------------------
RAGAS evaluation that benchmarks 4 pipeline configurations against
the golden dataset, so you can see the measurable improvement from
each upgrade.

4 configurations tested:
  baseline  — retrieve_and_generate (plain HYBRID, no reranking)
  +expand   — baseline + rule-based query expansion (BM25 boost)
  +rewrite  — baseline + LLM query rewriting (vector alignment boost)
  +rerank   — baseline + Bedrock Rerank API (top-20 → top-5 precision boost)

Baseline RAGAS scores (previous eval, raw JSON retrieval):
  context_precision : 0.307
  context_recall    : 0.718
  faithfulness      : 0.447
  answer_relevancy  : 0.525

Scores written to: bedrock_ragas_results.json

Usage:
  python evaluate_bedrock_pipeline.py                  # full 14-sample eval
  python evaluate_bedrock_pipeline.py --limit 5        # quick sanity check
  python evaluate_bedrock_pipeline.py --config rerank  # single config only
  python evaluate_bedrock_pipeline.py --no-llm-eval    # skip RAGAS, retrieval only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── colour helpers ──────────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def _col(val: Optional[float], width: int = 7) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f"{'N/A':>{width}}"
    s = f"{val:.3f}"
    if val >= 0.7:  return f"{GREEN}{s:>{width}}{RESET}"
    if val >= 0.5:  return f"{YELLOW}{s:>{width}}{RESET}"
    return f"{RED}{s:>{width}}{RESET}"

def _bar(val: Optional[float], width: int = 16) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "░" * width
    filled = int((val or 0) * width)
    return "█" * filled + "░" * (width - filled)

def _delta(new: Optional[float], base: Optional[float]) -> str:
    if new is None or base is None:
        return f"{'':>7}"
    d = new - base
    s = f"{d:+.3f}"
    if d > 0.02:   return f"{GREEN}{s:>7}{RESET}"
    if d < -0.02:  return f"{RED}{s:>7}{RESET}"
    return f"{YELLOW}{s:>7}{RESET}"


# ── Known baseline from previous raw-JSON eval ──────────────────────────────
BASELINE_SCORES = {
    "context_precision": 0.307,
    "context_recall"   : 0.718,
    "faithfulness"     : 0.447,
    "answer_relevancy" : 0.525,
}

METRIC_LABELS = {
    "context_precision": "Context Precision",
    "context_recall"   : "Context Recall",
    "faithfulness"     : "Faithfulness",
    "answer_relevancy" : "Answer Relevancy",
}

GOLDEN_DATASET_PATH = Path("gov_rag_pipeline/tests/golden_dataset.json")
OUTPUT_PATH         = Path("bedrock_ragas_results.json")


# ---------------------------------------------------------------------------
# Load golden dataset
# ---------------------------------------------------------------------------

def load_golden_dataset(limit: Optional[int] = None, dataset_path: Optional[str] = None) -> list[dict]:
    path = Path(dataset_path) if dataset_path else GOLDEN_DATASET_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    if limit:
        data = data[:limit]
    return data


# ---------------------------------------------------------------------------
# Pipeline runners — one function per configuration
# ---------------------------------------------------------------------------

def run_baseline(sample: dict, top_k: int, kb_id: str) -> dict:
    """Plain HYBRID retrieve-and-generate."""
    from gov_bedrock_pipeline.query import retrieve_and_generate, retrieve
    q = sample["question"]
    rag = retrieve_and_generate(q, kb_id=kb_id, top_k=top_k)
    ctx = [c["text"] for c in rag["citations"]] or _fallback_contexts(q, kb_id, top_k)
    return {"answer": rag["answer"], "contexts": ctx}


def run_expand(sample: dict, top_k: int, kb_id: str) -> dict:
    """HYBRID + rule-based query expansion."""
    from gov_bedrock_pipeline.query import retrieve_and_generate, expand_query
    q = expand_query(sample["question"])
    rag = retrieve_and_generate(q, kb_id=kb_id, top_k=top_k)
    ctx = [c["text"] for c in rag["citations"]] or _fallback_contexts(q, kb_id, top_k)
    return {"answer": rag["answer"], "contexts": ctx}


def run_rewrite(sample: dict, top_k: int, kb_id: str) -> dict:
    """HYBRID + LLM query rewriting."""
    from gov_bedrock_pipeline.query import rag_with_rewrite
    rag = rag_with_rewrite(sample["question"], kb_id=kb_id, top_k=top_k)
    ctx = [c["text"] for c in rag["citations"]] or _fallback_contexts(
        rag.get("rewritten_query", sample["question"]), kb_id, top_k
    )
    return {"answer": rag["answer"], "contexts": ctx, "rewritten": rag.get("rewritten_query")}


def run_rerank(sample: dict, top_k: int, kb_id: str) -> dict:
    """Two-stage: retrieve 20, vector-filter 15, dedup, rerank 10, answer top-5."""
    from gov_bedrock_pipeline.query import retrieve_rerank_generate
    rag = retrieve_rerank_generate(
        sample["question"],
        kb_id        = kb_id,
        top_k        = top_k,    # 5
        candidate_k  = 20,
        pre_rerank_k = 15,
        rerank_k     = 10,
        deduplicate  = True,
    )
    ctx = [c["text"] for c in rag["citations"]]
    return {"answer": rag["answer"], "contexts": ctx}



def run_rewrite_rerank(sample: dict, top_k: int, kb_id: str) -> dict:
    """LLM query rewrite → two-stage retrieve + dedup + rerank."""
    from gov_bedrock_pipeline.query import rag_with_rewrite, retrieve_rerank_generate
    rewrite_result = rag_with_rewrite(sample["question"], kb_id=kb_id, top_k=top_k)
    rewritten_q = rewrite_result.get("rewritten_query", sample["question"])
    rag = retrieve_rerank_generate(
        rewritten_q,
        kb_id        = kb_id,
        top_k        = top_k,
        candidate_k  = 20,
        pre_rerank_k = 15,
        rerank_k     = 10,
        deduplicate  = True,
    )
    ctx = [c["text"] for c in rag["citations"]]
    return {"answer": rag["answer"], "contexts": ctx, "rewritten": rewritten_q}


def run_optimized(sample: dict, top_k: int, kb_id: str) -> dict:
    """Full pipeline: expand → rewrite → retrieve 20 → prefilter 15 → dedup → rerank 10 → top 5."""
    from gov_bedrock_pipeline.query import expand_query, rewrite_query, retrieve_rerank_generate
    q = sample["question"]
    # rewrite first (vocabulary alignment), then expand (recall boost)
    try:
        q = rewrite_query(q)
    except Exception:
        pass
    q = expand_query(q)
    rag = retrieve_rerank_generate(
        q,
        kb_id        = kb_id,
        top_k        = top_k,
        candidate_k  = 20,
        pre_rerank_k = 15,
        rerank_k     = 10,
        deduplicate  = True,
    )
    ctx = [c["text"] for c in rag["citations"]]
    return {"answer": rag["answer"], "contexts": ctx, "rewritten": q}

def _fallback_contexts(q: str, kb_id: str, top_k: int) -> list[str]:
    """If RAG citations are empty, call retrieve() directly for context."""
    from gov_bedrock_pipeline.query import retrieve
    chunks = retrieve(q, kb_id=kb_id, top_k=top_k)
    return [c["text"] for c in chunks]


CONFIGS: dict[str, callable] = {
    "baseline"       : run_baseline,
    "+expand"        : run_expand,
    "+rewrite"       : run_rewrite,
    "+rerank"        : run_rerank,
    "+rewrite+rerank": run_rewrite_rerank,
    "+optimized"     : run_optimized,
}


# ---------------------------------------------------------------------------
# RAGAS evaluation helpers
# ---------------------------------------------------------------------------

def _build_ragas_dataset(samples: list[dict], run_results: list[dict]):
    """Build a RAGAS 0.4.x EvaluationDataset (SingleTurnSample)."""
    from ragas import EvaluationDataset, SingleTurnSample
    ragas_samples = []
    for sample, result in zip(samples, run_results):
        ragas_samples.append(SingleTurnSample(
            user_input         = sample["question"],
            response           = result.get("answer", ""),
            retrieved_contexts = result.get("contexts", []),
            reference          = sample.get("reference", ""),
        ))
    return EvaluationDataset(samples=ragas_samples)


def _run_ragas_eval(
    dataset,
    judge_provider: str = "bedrock",
    judge_model: Optional[str] = None,
) -> dict[str, float]:
    """Run RAGAS 0.4.x evaluation with selectable judge provider.

    Supported providers:
    - bedrock: ChatBedrockConverse + BedrockEmbeddings
    - nvidia:  ChatNVIDIA + NVIDIAEmbeddings
    """
    import os, warnings
    from dotenv import load_dotenv
    load_dotenv(override=True)

    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    if judge_provider == "bedrock":
        from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
    elif judge_provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
    else:
        raise ValueError(f"Unsupported judge provider: {judge_provider}")

    # Suppress the ragas.metrics import deprecation warnings — we know they
    # are deprecated but they are the only ones evaluate() accepts in 0.4.x.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if judge_provider == "bedrock":
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            llm_model = (
                judge_model
                or os.getenv("BEDROCK_RAGAS_LLM_MODEL")
                or os.getenv("BEDROCK_LLM_MODEL")
                or "amazon.nova-lite-v1:0"
            )
            emb_model = os.getenv("BEDROCK_RAGAS_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
            ragas_llm = LangchainLLMWrapper(ChatBedrockConverse(
                model=llm_model,
                region_name=aws_region,
                temperature=0.0,
                max_tokens=1024,
            ))
            ragas_emb = LangchainEmbeddingsWrapper(BedrockEmbeddings(
                model_id=emb_model,
                region_name=aws_region,
            ))
        else:
            nvidia_key = os.getenv("NVIDIA_API_KEY", "")
            llm_model = judge_model or os.getenv("NVIDIA_RAGAS_LLM_MODEL", "meta/llama-3.3-70b-instruct")
            emb_model = os.getenv("NVIDIA_RAGAS_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
            ragas_llm = LangchainLLMWrapper(ChatNVIDIA(
                model    = llm_model,
                api_key  = nvidia_key,
                base_url = "https://integrate.api.nvidia.com/v1",
            ))
            ragas_emb = LangchainEmbeddingsWrapper(NVIDIAEmbeddings(
                model    = emb_model,
                api_key  = nvidia_key,
                base_url = "https://integrate.api.nvidia.com/v1",
            ))

    # Inject LLM/embeddings onto the shared metric singletons
    faithfulness.llm       = ragas_llm
    answer_relevancy.llm   = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm  = ragas_llm
    context_recall.llm     = ragas_llm

    from ragas.run_config import RunConfig
    run_cfg = RunConfig(
        timeout     = 600,   # 10 min per LLM call — covers slow NVIDIA inference
        max_retries = 5,
        max_wait    = 60,
        max_workers = 2,     # 2 workers avoids concurrent embedding rate-limits
    )

    result = ragas_evaluate(
        dataset,
        metrics          = [faithfulness, answer_relevancy, context_precision, context_recall],
        llm              = ragas_llm,
        embeddings       = ragas_emb,
        run_config       = run_cfg,
        raise_exceptions = False,
    )
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    return {k: round(float(v), 4) for k, v in scores.items() if not math.isnan(v)}


# ---------------------------------------------------------------------------
# Retrieval-only metrics (no RAGAS LLM calls)
# ---------------------------------------------------------------------------

def _recall_at_k(expected_scheme: str, contexts: list[str]) -> float:
    """1.0 if the expected scheme name appears in any retrieved context."""
    if not expected_scheme:
        return float("nan")
    normalized = expected_scheme.lower().replace(" ", "")
    for ctx in contexts:
        if normalized in ctx.lower().replace(" ", ""):
            return 1.0
    return 0.0


def _mrr(expected_scheme: str, contexts: list[str]) -> float:
    """Mean Reciprocal Rank based on scheme name mention in contexts."""
    if not expected_scheme:
        return float("nan")
    normalized = expected_scheme.lower().replace(" ", "")
    for i, ctx in enumerate(contexts):
        if normalized in ctx.lower().replace(" ", ""):
            return 1.0 / (i + 1)
    return 0.0


def compute_retrieval_metrics(samples: list[dict], run_results: list[dict]) -> dict:
    recalls, mrrs = [], []
    for sample, result in zip(samples, run_results):
        expected = sample.get("expected_scheme", "")
        contexts = result.get("contexts", [])
        r   = _recall_at_k(expected, contexts)
        mrr = _mrr(expected, contexts)
        if not math.isnan(r):   recalls.append(r)
        if not math.isnan(mrr): mrrs.append(mrr)
    return {
        "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "mrr"        : round(sum(mrrs)   / len(mrrs  ), 4) if mrrs    else 0.0,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    limit        : Optional[int] = None,
    configs      : Optional[list[str]] = None,
    top_k        : int  = 7,
    run_llm_eval : bool = True,
    kb_id        : Optional[str] = None,
    dataset_path : Optional[str] = None,
    judge_provider: str = "bedrock",
    judge_model  : Optional[str] = None,
) -> dict:
    from gov_bedrock_pipeline.config import BEDROCK_KB_ID
    kb_id = kb_id or BEDROCK_KB_ID
    if not kb_id:
        raise ValueError(
            "BEDROCK_KB_ID not set. Run setup-kb first, or pass --kb-id."
        )

    samples  = load_golden_dataset(limit=limit, dataset_path=dataset_path)
    selected = {k: v for k, v in CONFIGS.items() if not configs or k in configs}

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Bedrock Pipeline RAGAS Evaluation{RESET}")
    print(f"  {len(samples)} samples  |  {len(selected)} config(s)  |  top_k={top_k}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    all_results: dict[str, dict] = {}

    for cfg_name, runner in selected.items():
        print(f"{CYAN}[{cfg_name}]{RESET} Running …")
        t0           = time.time()
        run_results  : list[dict] = []
        errors       = 0

        for i, sample in enumerate(samples, 1):
            try:
                res = runner(sample, top_k=top_k, kb_id=kb_id)
            except Exception as exc:
                print(f"  [WARN] sample {i} failed: {exc}")
                res = {"answer": "", "contexts": []}
                errors += 1
            run_results.append(res)
            print(f"  {i}/{len(samples)} done", end="\r")
            if i < len(samples):
                time.sleep(3)   # 3 s gap between questions avoids burst throttling

        elapsed = time.time() - t0
        print(f"  {len(samples)}/{len(samples)} done  ({elapsed:.1f}s){' '*10}")

        ret_metrics = compute_retrieval_metrics(samples, run_results)
        print(f"  Recall@{top_k}: {ret_metrics['recall_at_k']:.3f}  MRR: {ret_metrics['mrr']:.3f}")

        ragas_scores: dict[str, float] = {}
        if run_llm_eval:
            try:
                print(f"  Running RAGAS evaluation …")
                ds = _build_ragas_dataset(samples, run_results)
                ragas_scores = _run_ragas_eval(
                    ds,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                )
                for k, v in ragas_scores.items():
                    if k in METRIC_LABELS:
                        print(f"  {METRIC_LABELS.get(k, k)}: {_col(v)}")
            except Exception as exc:
                print(f"  [WARN] RAGAS eval failed: {exc}")

        all_results[cfg_name] = {
            **ret_metrics,
            **ragas_scores,
            "elapsed_s": round(elapsed, 1),
            "errors"   : errors,
            "samples"  : [
                {
                    "question"  : s["question"],
                    "answer"    : r.get("answer", ""),
                    "contexts"  : r.get("contexts", []),
                    "rewritten" : r.get("rewritten"),
                    "reference" : s.get("reference", ""),
                }
                for s, r in zip(samples, run_results)
            ],
        }
        print()

    # ── Print comparison table ──────────────────────────────────────────────
    _print_comparison_table(all_results, run_llm_eval=run_llm_eval)

    # ── Save results ────────────────────────────────────────────────────────
    save_data = {k: {mk: mv for mk, mv in v.items() if mk != "samples"}
                 for k, v in all_results.items()}
    save_data["_baseline_previous_eval"] = BASELINE_SCORES
    save_data["_timestamp"]  = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_data["_num_samples"] = len(samples)
    OUTPUT_PATH.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUTPUT_PATH}")

    return all_results


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def _print_comparison_table(results: dict, run_llm_eval: bool = True) -> None:
    configs    = list(results.keys())
    has_ragas  = run_llm_eval and any(
        "faithfulness" in v for v in results.values()
    )

    bar_w  = 16
    col_w  = 8
    name_w = 12

    print(f"\n{BOLD}{'='*78}{RESET}")
    print(f"{BOLD}  PIPELINE COMPARISON{RESET}")
    print(f"{BOLD}{'='*78}{RESET}")

    # Header row
    metrics = ["recall_at_k", "mrr"]
    if has_ragas:
        metrics += ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    short_labels = {
        "recall_at_k"      : "Recall@K",
        "mrr"              : "MRR",
        "context_precision": "Ctx Prec",
        "context_recall"   : "Ctx Rec",
        "faithfulness"     : "Faith",
        "answer_relevancy" : "AnsRel",
    }

    header_cells = f"{'Config':<{name_w}}"
    for m in metrics:
        header_cells += f"  {short_labels[m]:>{col_w}}"
    if has_ragas:
        header_cells += f"  {'vs Baseline':>{col_w+4}}"
    print(f"\n{BOLD}{header_cells}{RESET}")
    print("─" * (name_w + (col_w + 2) * len(metrics) + 20))

    # Previous baseline row (if RAGAS)
    if has_ragas:
        row = f"{'prev-baseline':<{name_w}}"
        for m in metrics:
            v = BASELINE_SCORES.get(m)
            row += f"  {_col(v, col_w)}"
        print(f"{row}  (raw JSON eval)")
        print("─" * (name_w + (col_w + 2) * len(metrics) + 20))

    # One row per config
    baseline_ref = results.get("baseline", {})
    for cfg, data in results.items():
        row = f"{cfg:<{name_w}}"
        for m in metrics:
            row += f"  {_col(data.get(m), col_w)}"
        if has_ragas:
            # Show delta vs previous baseline for faithfulness
            f_new  = data.get("faithfulness")
            f_base = BASELINE_SCORES.get("faithfulness")
            row += f"  {_delta(f_new, f_base):>12}"
        errors = data.get("errors", 0)
        if errors:
            row += f"  ({errors} errors)"
        print(row)

    print("─" * (name_w + (col_w + 2) * len(metrics) + 20))
    print(f"\n  {GREEN}Green{RESET}=≥0.70  {YELLOW}Yellow{RESET}=≥0.50  {RED}Red{RESET}=<0.50")
    print(f"  Delta = difference vs previous raw-JSON baseline (faithfulness)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Bedrock RAG pipeline across 4 configurations using RAGAS"
    )
    parser.add_argument("--limit",    type=int, default=None,
                        help="Limit number of golden samples (default: all 14)")
    parser.add_argument("--config",   nargs="+", default=None,
                        choices=list(CONFIGS.keys()),
                        help="Run specific config(s) only")
    parser.add_argument("--top-k",    type=int, default=5,
                        help="Number of chunks for final answer (default: 5)")
    parser.add_argument("--kb-id",    default=None,
                        help="Override BEDROCK_KB_ID from .env")
    parser.add_argument("--no-llm-eval", action="store_true",
                        help="Skip RAGAS LLM evaluation — only compute Recall@K and MRR")
    parser.add_argument("--dataset", default=None,
                        help="Path to a custom JSON dataset (default: golden_dataset.json)")
    parser.add_argument("--judge-provider", choices=["bedrock", "nvidia"], default="bedrock",
                        help="RAGAS judge provider (default: bedrock)")
    parser.add_argument("--judge-model", default=None,
                        help="Override model for selected --judge-provider")
    parser.add_argument("--generate", type=int, default=None, metavar="N",
                        help="Generate a fresh N-question dataset before running eval")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for --generate (default: random each run)")
    args = parser.parse_args()

    dataset_path = args.dataset
    if args.generate:
        from gen_eval_dataset import generate_dataset
        import json, tempfile, os
        generated = generate_dataset(n=args.generate, seed=args.seed)
        tmp = Path(tempfile.mktemp(suffix=".json", prefix="eval_"))
        tmp.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[gen] {args.generate} fresh questions written to {tmp.name}")
        for i, d in enumerate(generated, 1):
            print(f"  {i:2}. {d['question'][:85]}...")
        print()
        dataset_path = str(tmp)

    evaluate(
        limit        = args.limit,
        configs      = args.config,
        top_k        = args.top_k,
        run_llm_eval = not args.no_llm_eval,
        kb_id        = args.kb_id,
        dataset_path = dataset_path,
        judge_provider = args.judge_provider,
        judge_model    = args.judge_model,
    )


if __name__ == "__main__":
    main()

