from typing import List

from openai import OpenAI

from .config import NVIDIA_API_KEY, NVIDIA_BASE_URL, LLM_MODEL, N_QUERY_VARIANTS

# Domain-specific synonym map for Indian government scheme terminology
SYNONYM_MAP = {
    "weaver":        ["handloom", "bunkar", "textile worker", "artisan"],
    "farmer":        ["kisan", "agriculturalist", "cultivator", "krishi"],
    "sc":            ["scheduled caste", "dalit", "sc/st"],
    "st":            ["scheduled tribe", "adivasi", "tribal", "sc/st"],
    "obc":           ["other backward class", "backward community"],
    "disabled":      ["divyangjan", "handicapped", "person with disability", "pwd"],
    "widow":         ["widowed", "vidhwa"],
    "student":       ["scholar", "pupil", "learner"],
    "scholarship":   ["stipend", "education grant", "vidya", "financial assistance education"],
    "loan":          ["credit", "microfinance", "mudra", "kcc", "kisan credit card"],
    "subsidy":       ["grant", "financial assistance", "financial support"],
    "pension":       ["monthly assistance", "social security", "old age benefit"],
    "insurance":     ["bima", "coverage", "protection scheme"],
    "self employed": ["entrepreneur", "shg", "self help group", "msme"],
    "women":         ["mahila", "stree", "female", "lady"],
    "minority":      ["muslim", "christian", "sikh", "buddhist", "jain", "parsi"],
    "bpl":           ["below poverty line", "ews", "economically weaker section"],
    "housing":       ["awas", "shelter", "home", "pradhan mantri awas"],
    "health":        ["medical", "hospital", "treatment", "ayushman"],
    "skill":         ["training", "vocational", "pmkvy", "pradhan mantri kaushal"],
}


def expand_query(query: str) -> str:
    """Append synonym expansions to the query for domain coverage."""
    q_lower = query.lower()
    expansions = []
    for term, synonyms in SYNONYM_MAP.items():
        if term in q_lower:
            expansions.extend(synonyms)
    if expansions:
        unique = list(dict.fromkeys(expansions))[:8]
        return query + " " + " ".join(unique)
    return query


def rewrite_query(query: str) -> str:
    """Rewrite a vague or informal query into a precise formal search query."""
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    prompt = (
        "You are a query rewriter for an Indian government welfare scheme search engine. "
        "Rewrite the user's query to be clearer, more specific, and use formal terminology "
        "used in Indian government schemes. Include relevant categories like scholarship, loan, "
        "subsidy, insurance, pension, housing, skill development, etc. when applicable. "
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"Original: {query}\nRewritten:"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def generate_query_variants(query: str) -> List[str]:
    """Generate N query variants that approach the topic from different angles."""
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    prompt = (
        f"Generate {N_QUERY_VARIANTS} different search queries for Indian government schemes "
        "based on this query. Each should approach the topic from a different angle "
        "(e.g., beneficiary-focused, benefit-focused, eligibility-focused). "
        "Return ONLY the queries, one per line, no numbering or extra text.\n\n"
        f"Query: {query}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
    )
    lines = [l.strip() for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]
    return lines[:N_QUERY_VARIANTS]
