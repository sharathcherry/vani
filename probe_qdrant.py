import boto3, json

lambda_client = boto3.client('lambda', region_name='eu-north-1')

# Test multiple queries to see what Telangana data exists in Qdrant
queries = [
    "Telangana scholarship scheme",
    "Telangana student BC SC OBC scheme",
    "Telangana engineering BTech scholarship SC caste",
    "Dalit student scholarship Telangana",
    "Post matric scholarship Telangana SC ST",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print('='*60)

    response = lambda_client.invoke(
        FunctionName='gov-schemes-voice-rag',
        InvocationType='RequestResponse',
        Payload=json.dumps({
            "rawPath": "/debug/query",
            "body": json.dumps({"query": q}),
            "headers": {"content-type": "application/json"}
        })
    )
    result = json.loads(response['Payload'].read())
    body = json.loads(result.get('body', '{}'))

    for i, ctx in enumerate(body.get('contexts', [])):
        lines = ctx.split('\n')
        scheme = next((l for l in lines if 'SCHEME:' in l), '')
        state = next((l for l in lines if 'APPLICABLE IN' in l), '')
        caste = next((l for l in lines if 'CASTE' in l), '')
        print(f"  [{i+1}] {scheme} | {state} | {caste}")

    with open('rag_probe.txt', 'a', encoding='utf-8') as f:
        f.write(f"\nQUERY: {q}\n")
        for i, ctx in enumerate(body.get('contexts', [])):
            lines = ctx.split('\n')
            scheme = next((l for l in lines if 'SCHEME:' in l), '')
            state = next((l for l in lines if 'APPLICABLE IN' in l), '')
            caste = next((l for l in lines if 'CASTE' in l), '')
            f.write(f"  [{i+1}] {scheme} | {state} | {caste}\n")

print("\nDone. Results also written to rag_probe.txt")
