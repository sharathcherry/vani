"""
gen_200_dataset.py
------------------
Generates a 200-sample RAGAS golden dataset from the structured scheme JSONs
in all_sources/structured_schemes/.

Strategy:
  - Samples 200 unique schemes, diverse across category, state, and beneficiary type.
  - Generates one realistic question + reference answer per scheme using GPT-4o-mini.
  - Question types are rotated across: benefit, eligibility, application, overview.
  - Output: gov_rag_pipeline/tests/golden_dataset.json  (same format used by
    evaluate_qdrant_bedrock.py — each item has 'question' and 'reference').

Usage:
  python gen_200_dataset.py
  python gen_200_dataset.py --output my_dataset.json --workers 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────
SCHEMES_DIR   = Path("all_sources/structured_schemes")
OUTPUT_PATH   = Path("gov_rag_pipeline/tests/golden_dataset.json")
NUM_SAMPLES   = 200
SEED          = 42

QUESTION_TYPES = [
    "benefit",       # How much money / what benefit does it provide?
    "eligibility",   # Who is eligible / what are the criteria?
    "application",   # How to apply / what documents are needed?
    "overview",      # What is this scheme? What is its objective?
    "comparison",    # specific detail about amount, age limit, income limit
]

SYSTEM_PROMPT = """\
You are creating evaluation questions for a RAG system that answers queries about \
Indian government welfare schemes. Generate exactly ONE realistic user question and \
ONE concise factual reference answer based on the scheme information provided.

Rules:
- Question must be what a real citizen would ask (natural, conversational).
- Reference answer must be factually grounded ONLY in the provided scheme data.
- Reference answer: 1-4 sentences, include specific figures (amounts, ages, %).
- Output ONLY valid JSON: {"question": "...", "reference": "..."}
- No extra text, no markdown fences.
"""


def load_all_schemes() -> list[dict]:
    """Load all structured scheme JSON files."""
    files = list(SCHEMES_DIR.glob("*.json"))
    print(f"[load] Found {len(files)} scheme files in {SCHEMES_DIR}")
    schemes = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("scheme_name") and data.get("text_for_embedding"):
                schemes.append(data)
        except Exception:
            pass
    print(f"[load] Parsed {len(schemes)} valid schemes")
    return schemes


def sample_diverse(schemes: list[dict], n: int, seed: int) -> list[dict]:
    """
    Sample n schemes with diversity across category and state/UT.
    Ensures no two selected schemes share the same scheme_id.
    """
    rng = random.Random(seed)

    # Build category buckets
    buckets: dict[str, list[dict]] = {}
    for s in schemes:
        cat = s.get("scheme_category", "Other") or "Other"
        buckets.setdefault(cat, []).append(s)

    categories = list(buckets.keys())
    rng.shuffle(categories)

    selected: list[dict] = []
    seen_ids: set[str] = set()

    # Round-robin across categories until we have n
    cat_idx = 0
    shuffled: dict[str, list[dict]] = {c: rng.sample(v, len(v)) for c, v in buckets.items()}
    pointers: dict[str, int] = {c: 0 for c in categories}

    while len(selected) < n:
        cat = categories[cat_idx % len(categories)]
        ptr = pointers[cat]
        pool = shuffled[cat]
        if ptr < len(pool):
            s = pool[ptr]
            pointers[cat] += 1
            sid = s.get("scheme_id", s["scheme_name"])
            if sid not in seen_ids:
                seen_ids.add(sid)
                selected.append(s)
        cat_idx += 1
        # safety exit if all pools exhausted
        if cat_idx > len(categories) * 10000:
            break

    print(f"[sample] Selected {len(selected)} schemes across {len(set(s.get('scheme_category','Other') for s in selected))} categories")
    return selected[:n]


def build_scheme_context(s: dict) -> str:
    """Compact scheme description for the prompt — keeps token cost low."""
    lines = [f"Scheme: {s.get('scheme_name', 'Unknown')}"]
    if s.get("scheme_category"):
        lines.append(f"Category: {s['scheme_category']}")
    if s.get("state_or_ut"):
        lines.append(f"State/Level: {s['state_or_ut']}")
    if s.get("ministry_or_department"):
        lines.append(f"Administered by: {s['ministry_or_department']}")
    if s.get("objective"):
        lines.append(f"Objective: {s['objective']}")
    if s.get("target_beneficiaries"):
        tgt = s["target_beneficiaries"]
        lines.append(f"Target beneficiaries: {', '.join(tgt) if isinstance(tgt, list) else tgt}")

    # Eligibility
    age = s.get("age_criteria", {})
    if age and (age.get("min_age") or age.get("max_age")):
        lines.append(f"Age criteria: {age.get('min_age','?')} – {age.get('max_age','?')} years")
    inc = s.get("income_criteria", {})
    if inc and inc.get("max_income_inr"):
        lines.append(f"Max income: ₹{inc['max_income_inr']:,}")
    if s.get("occupation_criteria"):
        lines.append(f"Eligible occupations: {', '.join(s['occupation_criteria'][:5])}")
    if s.get("caste_criteria"):
        lines.append(f"Caste criteria: {', '.join(s['caste_criteria'])}")
    if s.get("gender_criteria"):
        lines.append(f"Gender criteria: {', '.join(s['gender_criteria'])}")

    # Benefits
    bd = s.get("benefit_details", {})
    if bd:
        if bd.get("benefit_type"):
            lines.append(f"Benefit type: {bd['benefit_type']}")
        if bd.get("max_amount_inr") or bd.get("min_amount_inr"):
            lo = bd.get("min_amount_inr")
            hi = bd.get("max_amount_inr")
            amt = f"₹{lo:,} – ₹{hi:,}" if lo and hi else f"₹{lo or hi:,}"
            lines.append(f"Benefit amount: {amt}")
        if bd.get("subsidy_percentage"):
            lines.append(f"Subsidy: {bd['subsidy_percentage']}%")
        if bd.get("frequency"):
            lines.append(f"Payment frequency: {bd['frequency']}")

    if s.get("exclusions"):
        lines.append(f"Not eligible: {'; '.join(s['exclusions'][:3])}")
    if s.get("documents_required"):
        lines.append(f"Documents: {', '.join(s['documents_required'][:4])}")
    if s.get("application_mode"):
        mode = s["application_mode"]
        lines.append(f"How to apply: {', '.join(mode) if isinstance(mode, list) else mode}")

    return "\n".join(lines)


def generate_qa(
    scheme: dict,
    question_type: str,
    client: OpenAI,
    retries: int = 3,
) -> dict | None:
    """Call GPT-4o-mini to generate one Q&A pair for the scheme."""
    context = build_scheme_context(scheme)

    type_instructions = {
        "benefit":      "Focus on: financial benefit amounts, subsidy percentages, payment frequency, or duration.",
        "eligibility":  "Focus on: who is eligible, age limits, income limits, caste/gender criteria, or exclusions.",
        "application":  "Focus on: how to apply, documents required, online/offline process, or deadlines.",
        "overview":     "Focus on: what the scheme is, its objective, who administers it, and which state/level.",
        "comparison":   "Focus on: a specific numeric detail (exact amount, age limit, income cap, or subsidy %).",
    }
    instruction = type_instructions.get(question_type, "")

    user_prompt = f"""{context}

Question type hint: {instruction}

Generate a question and reference answer about this scheme."""

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            if "question" in parsed and "reference" in parsed:
                return {
                    "question":    parsed["question"],
                    "reference":   parsed["reference"],
                    "scheme_name": scheme.get("scheme_name", ""),
                    "scheme_id":   scheme.get("scheme_id", ""),
                    "category":    scheme.get("scheme_category", ""),
                    "state":       scheme.get("state_or_ut", ""),
                    "question_type": question_type,
                }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"  [FAIL] {scheme.get('scheme_name','?')}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default=str(OUTPUT_PATH))
    parser.add_argument("--workers", type=int, default=8, help="Parallel API calls")
    parser.add_argument("--seed",    type=int, default=SEED)
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise SystemExit("ERROR: OPENAI_API_KEY not set in .env")

    client = OpenAI(api_key=openai_key)

    # Load + sample
    schemes = load_all_schemes()
    selected = sample_diverse(schemes, NUM_SAMPLES, args.seed)

    # Assign question types in rotation
    qt_cycle = QUESTION_TYPES * ((NUM_SAMPLES // len(QUESTION_TYPES)) + 1)
    random.Random(args.seed).shuffle(qt_cycle)
    tasks = [(s, qt_cycle[i]) for i, s in enumerate(selected)]

    # Generate in parallel
    print(f"\n[generate] Generating {len(tasks)} Q&A pairs with {args.workers} workers...")
    results: list[dict] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_qa, s, qt, client): (i, s) for i, (s, qt) in enumerate(tasks)}
        for i, fut in enumerate(as_completed(futures), 1):
            idx, scheme = futures[fut]
            result = fut.result()
            if result:
                results.append(result)
                print(f"  [{i}/{len(tasks)}] ✓ {scheme.get('scheme_name','')[:60]}  [{result['question_type']}]")
            else:
                failed += 1
                print(f"  [{i}/{len(tasks)}] ✗ FAILED — {scheme.get('scheme_name','')[:60]}")

    print(f"\n[done] Generated {len(results)} samples ({failed} failed)")

    # Sort by scheme name for reproducibility
    results.sort(key=lambda x: x["scheme_name"])

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out}  ({len(results)} samples)")

    # Summary stats
    from collections import Counter
    cats = Counter(r["category"] for r in results)
    qtypes = Counter(r["question_type"] for r in results)
    print("\nCategory spread:")
    for cat, cnt in cats.most_common(10):
        print(f"  {cat:40s}: {cnt}")
    print("\nQuestion type spread:")
    for qt, cnt in qtypes.most_common():
        print(f"  {qt:15s}: {cnt}")


if __name__ == "__main__":
    main()
