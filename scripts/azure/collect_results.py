#!/usr/bin/env python3
"""Reassemble the RESULT blocks that infra/trainer/train.py prints into the job log.

Usage: scripts/azure/collect_results.py <execution-name> <out-dir>
Reads `az containerapp job logs show` for that execution (the log lines are JSON objects with a "Log"
field), decodes every RESULT_BEGIN ... RESULT_END block (base64 of gzip) and writes the files to out-dir.
No storage account is needed; results are a few hundred KB. Also prints the DRAFTDB_* marker lines.
"""
import sys, os, json, subprocess, base64, gzip
exe, out = sys.argv[1], sys.argv[2]
rg, job = os.environ.get("AZURE_RESOURCE_GROUP", "ml-training-rg"), os.environ.get("AZURE_TRAINING_JOB", "a100-training")
raw = subprocess.check_output(["az", "containerapp", "job", "logs", "show", "--name", job, "--resource-group", rg,
                               "--execution", exe, "--container", "trainer", "--tail", "100000"], text=True, stderr=subprocess.DEVNULL)
lines = []
for l in raw.splitlines():
    try:
        lines.append(json.loads(l)["Log"])
    except Exception:
        pass
os.makedirs(out, exist_ok=True)
cur, buf, n = None, [], 0
for l in lines:
    l = l[2:] if l.startswith("F ") else l
    if l.startswith("DRAFTDB_"):
        print(l)
    elif l.startswith("RESULT_BEGIN "):
        cur, buf = l.split()[1], []
    elif l.startswith("R ") and cur:
        buf.append(l[2:])
    elif l.startswith("RESULT_END ") and cur:
        data = gzip.decompress(base64.b64decode("".join(buf)))
        open(os.path.join(out, cur), "wb").write(data); n += 1
        print("wrote", os.path.join(out, cur), len(data), "bytes"); cur = None
print(f"{n} files")
