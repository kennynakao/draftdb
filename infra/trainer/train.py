"""A100 job entrypoint: run the GPU variant queue of the DraftDB lab and emit every result through the job log.
Outputs (exp_*.json, board_*.csv, experiments.jsonl) are written to /workspace/out and then printed as
RESULT blocks (base64 of gzip) so they can be reassembled from `az containerapp job logs show` without
any storage account. Queue name comes from QUEUE (default G)."""
import os, sys, glob, gzip, base64, io, time, subprocess, json
os.environ.setdefault("TABICL_DEVICE", "cuda")
os.environ.setdefault("DRAFTDB_LAB_OUT", "/workspace/out")
os.makedirs("/workspace/out", exist_ok=True)
import torch
print("DRAFTDB_TRAIN_START", json.dumps(dict(cuda=torch.cuda.is_available(), device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, count=torch.cuda.device_count())), flush=True)
sys.path.insert(0, "/workspace/draftdb/tools/lab")
t0 = time.time()
rc = subprocess.call([sys.executable, "/workspace/draftdb/tools/lab/run_queue.py", os.environ.get("QUEUE", "G")], cwd="/workspace/draftdb")
print(f"DRAFTDB_QUEUE_DONE rc={rc} secs={time.time()-t0:.0f}", flush=True)
for p in sorted(glob.glob("/workspace/out/*")):
    raw = open(p, "rb").read()
    b = base64.b64encode(gzip.compress(raw)).decode()
    print(f"RESULT_BEGIN {os.path.basename(p)} {len(b)}", flush=True)
    for i in range(0, len(b), 3000):
        print("R " + b[i:i+3000], flush=True)
    print(f"RESULT_END {os.path.basename(p)}", flush=True)
print("DRAFTDB_TRAIN_END", flush=True)
