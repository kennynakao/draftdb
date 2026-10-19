#!/usr/bin/env bash
set -euo pipefail
EXECUTION="${1:?Usage: $0 <execution-name>}"
az containerapp job stop --name "${AZURE_TRAINING_JOB:-a100-training}" --resource-group "${AZURE_RESOURCE_GROUP:-ml-training-rg}" --job-execution-name "$EXECUTION"
