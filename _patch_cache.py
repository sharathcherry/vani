import re

with open("main.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1) Add 'import hashlib' after 'import threading'
code = code.replace(
    "import threading\n",
    "import threading\nimport hashlib\n",
    1
)

# 2) Add CACHE_TTL constant after SESSION_TTL_SECONDS line
code = code.replace(
    "SESSION_TTL_SECONDS = 3 * 60 * 60  # 3 hours\n",
    "SESSION_TTL_SECONDS = 3 * 60 * 60  # 3 hours\n"
    "CACHE_TTL_SECONDS   = 24 * 60 * 60  # 24-hour answer cache\n",
    1
)

# 3) Replace get_rag_answer with cached version
old_rag = '''def get_rag_answer(english_query: str) -> str:
    response = lambda_client.invoke(
        FunctionName=RAG_LAMBDA_NAME,
        InvocationType="RequestResponse",
        Payload=json.dumps({
            "rawPath": "/debug/query",
            "body": json.dumps({"query": english_query}),
            "headers": {"content-type": "application/json"},
        }),
    )
    result = json.loads(response["Payload"].read())
    body   = json.loads(result.get("body", "{}"))
    return body.get("answer", "No information found.")'''

new_rag = '''def _normalize_query(q: str) -> str:
    """Lowercase, collapse whitespace, strip session context prefix."""
    # Strip session context if present (everything before "New question:")
    if "New question:" in q:
        q = q.split("New question:")[-1]
    return " ".join(q.lower().split())


def get_rag_answer(english_query: str) -> str:
    """Fetch answer from RAG Lambda, with 24-hour S3 cache on the raw query."""
    normalized = _normalize_query(english_query)
    cache_key  = "wa-cache/" + hashlib.sha256(normalized.encode()).hexdigest() + ".txt"

    # -- Try cache first ---------------------------------------------------
    try:
        obj = s3.get_object(Bucket=S3_BUCKET_IN, Key=cache_key)
        age = time.time() - obj["LastModified"].timestamp()
        if age < CACHE_TTL_SECONDS:
            cached = obj["Body"].read().decode("utf-8")
            print(f"[cache] HIT (age={age/3600:.1f}h) key={cache_key[-12:]}")
            return cached
        print(f"[cache] EXPIRED (age={age/3600:.1f}h)")
    except s3.exceptions.NoSuchKey:
        pass
    except Exception:
        pass   # cache miss — non-critical

    # -- Cache miss: call RAG Lambda ---------------------------------------
    print("[cache] MISS — invoking RAG Lambda")
    response = lambda_client.invoke(
        FunctionName=RAG_LAMBDA_NAME,
        InvocationType="RequestResponse",
        Payload=json.dumps({
            "rawPath": "/debug/query",
            "body": json.dumps({"query": english_query}),
            "headers": {"content-type": "application/json"},
        }),
    )
    result = json.loads(response["Payload"].read())
    body   = json.loads(result.get("body", "{}"))
    answer = body.get("answer", "No information found.")

    # -- Write to cache (best-effort, non-blocking) ------------------------
    try:
        s3.put_object(Bucket=S3_BUCKET_IN, Key=cache_key, Body=answer.encode("utf-8"))
        print(f"[cache] STORED key={cache_key[-12:]}")
    except Exception:
        pass

    return answer'''

code = code.replace(old_rag, new_rag)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(code)

print("OK: caching patched into main.py")
