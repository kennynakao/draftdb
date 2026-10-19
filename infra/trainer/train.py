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
# primary channel: Blob Storage (results/<execution>/...), when the job carries the connection string
conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING"); cont = os.environ.get("AZURE_STORAGE_CONTAINER", "ml-artifacts")
exe = os.environ.get("CONTAINER_APP_JOB_EXECUTION_NAME", time.strftime("run-%Y%m%d%H%M%S"))
if conn:
    try:
        from azure.storage.blob import BlobServiceClient
        cc = BlobServiceClient.from_connection_string(conn).get_container_client(cont)
        n = 0
        for p in sorted(glob.glob("/workspace/out/*")):
            cc.upload_blob(f"results/{exe}/{os.path.basename(p)}", open(p, "rb"), overwrite=True); n += 1
        print(f"DRAFTDB_UPLOADED {n} files to {cont}/results/{exe}", flush=True)
    except Exception as e:
        print("DRAFTDB_UPLOAD_FAILED", repr(e)[:200], flush=True)
# fallback channel: the log (Log Analytics drops lines under load, so this is best effort)
for p in sorted(glob.glob("/workspace/out/*")):
    raw = open(p, "rb").read()
    b = base64.b64encode(gzip.compress(raw)).decode()
    print(f"RESULT_BEGIN {os.path.basename(p)} {len(b)}", flush=True)
    for i in range(0, len(b), 3000):
        print("R " + b[i:i+3000], flush=True)
    print(f"RESULT_END {os.path.basename(p)}", flush=True)
print("DRAFTDB_TRAIN_END", flush=True)
