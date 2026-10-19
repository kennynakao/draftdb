#!/usr/bin/env bash
set -euo pipefail
az containerapp job execution list --name "${AZURE_TRAINING_JOB:-a100-training}" --resource-group "${AZURE_RESOURCE_GROUP:-ml-training-rg}" \
  --query "[].{name:name,status:properties.status,start:properties.startTime,end:properties.endTime}" -o table
