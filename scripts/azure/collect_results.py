#!/usr/bin/env python3
"""Collect a job execution's outputs.

Usage: scripts/azure/collect_results.py <out-dir> [hours-back] [execution-name]
Primary channel: Blob Storage, results/<execution>/ in the ml-artifacts container of the account named
AZURE_STORAGE_ACCOUNT (.azure-training.env); downloaded with the account's connection string. Fallback: the
RESULT blocks in Log Analytics (unreliable under load; lines get dropped).
The live log stream (`az containerapp job logs show`) only works while a replica is running, so this
reads the environment's Log Analytics workspace (ContainerAppConsoleLogs_CL), which keeps the console
output of finished executions (ingestion lags a few minutes). Every RESULT_BEGIN ... RESULT_END block
(base64 of gzip) is decoded into out-dir; DRAFTDB_* marker lines are printed. Needs the log-analytics
CLI extension (az extension add --name log-analytics).
"""
import sys, os, json, subprocess, base64, gzip
out = sys.argv[1]; hours = sys.argv[2] if len(sys.argv) > 2 else "6"; exe = sys.argv[3] if len(sys.argv) > 3 else None
acct = os.environ.get("AZURE_STORAGE_ACCOUNT")
if acct:
    rgx = os.environ.get("AZURE_RESOURCE_GROUP", "ml-training-rg")
    conn = subprocess.check_output(["az", "storage", "account", "show-connection-string", "--name", acct, "--resource-group", rgx,
                                    "--query", "connectionString", "-o", "tsv"], text=True, stderr=subprocess.DEVNULL).strip()
    if not exe:
        names = json.loads(subprocess.check_output(["az", "storage", "blob", "list", "--container-name", "ml-artifacts", "--prefix", "results/",
                                                    "--connection-string", conn, "--query", "[].name", "-o", "json"], text=True, stderr=subprocess.DEVNULL))
        runs = sorted({n.split("/")[1] for n in names if n.count("/") >= 2})
        exe = runs[-1] if runs else None
    if exe:
        os.makedirs(out, exist_ok=True)
        subprocess.check_call(["az", "storage", "blob", "download-batch", "--destination", out, "--source", "ml-artifacts", "--pattern",
                               f"results/{exe}/*", "--connection-string", conn, "-o", "none"], stderr=subprocess.DEVNULL)
        # download-batch keeps the blob path; flatten results/<exe>/* into out-dir
        import shutil
        src = os.path.join(out, "results", exe)
        if os.path.isdir(src):
            for f in os.listdir(src):
                shutil.move(os.path.join(src, f), os.path.join(out, f))
            shutil.rmtree(os.path.join(out, "results"), ignore_errors=True)
        print(f"downloaded results/{exe} into {out} ({len(os.listdir(out))} files)"); sys.exit(0)
    print("no results in blob storage yet; falling back to Log Analytics")
rg = os.environ.get("AZURE_RESOURCE_GROUP", "ml-training-rg"); job = os.environ.get("AZURE_TRAINING_JOB", "a100-training")
env = os.environ.get("AZURE_CONTAINERAPPS_ENV", "ml-training-env")
ws = subprocess.check_output(["az", "containerapp", "env", "show", "--name", env, "--resource-group", rg, "--query",
                              "properties.appLogsConfiguration.logAnalyticsConfiguration.customerId", "-o", "tsv"], text=True, stderr=subprocess.DEVNULL).strip()
q = (f"ContainerAppConsoleLogs_CL | where TimeGenerated > ago({hours}h) and ContainerJobName_s == '{job}' "
     "| project TimeGenerated, Log_s | order by TimeGenerated asc")
rows = json.loads(subprocess.check_output(["az", "monitor", "log-analytics", "query", "--workspace", ws, "--analytics-query", q, "-o", "json"],
                                          text=True, stderr=subprocess.DEVNULL))
lines = [r["Log_s"] for r in rows]
os.makedirs(out, exist_ok=True)
cur, buf, n = None, [], 0
for l in lines:
    if l.startswith("DRAFTDB_") or l.startswith("== ") or l.startswith("   folds:") or l.startswith("   strict:") or l.startswith("   expanding:"):
        print(l[:200])
    elif l.startswith("RESULT_BEGIN "):
        cur, buf = l.split()[1], []
    elif l.startswith("R ") and cur:
        buf.append(l[2:])
    elif l.startswith("RESULT_END ") and cur:
        try:
            data = gzip.decompress(base64.b64decode("".join(buf)))
            open(os.path.join(out, cur), "wb").write(data); n += 1; print("wrote", os.path.join(out, cur), len(data), "bytes")
        except Exception as e:
            print("could not decode", cur, repr(e)[:80], "(ingestion may still be incomplete)")
        cur = None
print(f"{len(rows)} log rows, {n} files written to {out}")
