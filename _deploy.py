"""Deploy vani_webhook.zip to AWS Lambda and increase memory."""
import subprocess, json, sys

LOG = "_deploy_output.txt"

def run(cmd, label):
    print(f"\n=== {label} ===")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout.strip() or r.stderr.strip() or "(no output)"
    print(out)
    with open(LOG, "a") as f:
        f.write(f"\n=== {label} ===\n{out}\n")
    return r.returncode

# Clear log
open(LOG, "w").close()

# 1. Test AWS CLI
rc = run(["aws", "sts", "get-caller-identity", "--output", "json"], "AWS Identity Check")
if rc != 0:
    print("ERROR: AWS CLI not working. Check credentials/memory.")
    sys.exit(1)

# 2. Deploy code
rc = run([
    "aws", "lambda", "update-function-code",
    "--function-name", "vani-jan-webhook",
    "--region", "eu-north-1",
    "--zip-file", "fileb://vani_webhook.zip",
    "--query", "{LastModified:LastModified,CodeSize:CodeSize}",
    "--output", "json"
], "Deploy Lambda Code")

if rc != 0:
    print("ERROR: Deploy failed.")
    sys.exit(2)

# 3. Increase memory to 1024MB
rc = run([
    "aws", "lambda", "update-function-configuration",
    "--function-name", "vani-jan-webhook",
    "--region", "eu-north-1",
    "--memory-size", "1024",
    "--query", "{LastModified:LastModified,MemorySize:MemorySize}",
    "--output", "json"
], "Update Memory to 1024MB")

if rc != 0:
    print("ERROR: Memory update failed.")
    sys.exit(3)

print("\n=== ALL DONE ===")
