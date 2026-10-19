#!/usr/bin/env bash
# Fail closed unless the a100-training job still sits on the A100 workload profile with the a100-test resource shape.
# On this environment the GPU comes from the profile (NC24-A100); the container resources carry no gpu field,
# exactly like the known-good a100-test app (cpu 24, memory 220Gi).
set -euo pipefail
RG="${AZURE_RESOURCE_GROUP:-ml-training-rg}"; JOB="${AZURE_TRAINING_JOB:-a100-training}"
J="$(az containerapp job show --name "$JOB" --resource-group "$RG" -o json)"
python3 - "$J" <<'PY'
import json, sys
d = json.loads(sys.argv[1]); p = d["properties"]
prof = p.get("workloadProfileName"); c = p["template"]["containers"][0]; r = c["resources"]
print("image:", c["image"]); print("workloadProfileName:", prof); print("resources:", json.dumps(r))
ok = prof == "NC24-A100" and float(r.get("cpu", 0)) >= 24 and str(r.get("memory", "")).startswith("220")
print("GPU CONFIG OK" if ok else "GPU CONFIG CHANGED - refusing to start")
sys.exit(0 if ok else 1)
PY
