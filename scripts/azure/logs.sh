#!/usr/bin/env bash
# Tail the trainer container's log for one execution (default: the newest). Add --follow to stream.
set -euo pipefail
RG="${AZURE_RESOURCE_GROUP:-ml-training-rg}"; JOB="${AZURE_TRAINING_JOB:-a100-training}"
EXECUTION="${1:-$(az containerapp job execution list --name "$JOB" --resource-group "$RG" --query "sort_by(@,&properties.startTime)[-1].name" -o tsv)}"
echo "Execution: $EXECUTION"
az containerapp job logs show --name "$JOB" --resource-group "$RG" --execution "$EXECUTION" --container trainer --tail 300 "${@:2}"
