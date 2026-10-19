#!/usr/bin/env bash
# Build the trainer image in ACR, point the a100-training job at it, verify the GPU profile is intact, start one execution.
# Usage: scripts/azure/train.sh [QUEUE]      (QUEUE defaults to G, see tools/lab/run_queue.py)
# Needs: az login (browser flow; the UC Davis tenant blocks device code), .azure-training.env at the repo root
# (non-secret names, gitignored), and infra/trainer/bundle assembled by infra/trainer/make_bundle.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .azure-training.env
./scripts/azure/check-auth.sh
QUEUE="${1:-G}"
TAG="$(git rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M%S)"
IMAGE="$AZURE_ACR_LOGIN_SERVER/trainer:$TAG"
echo "building $IMAGE"
az acr build --registry "$AZURE_ACR_NAME" --image "trainer:$TAG" --platform linux/amd64 infra/trainer
echo "updating job image and queue"
az containerapp job update --name "$AZURE_TRAINING_JOB" --resource-group "$AZURE_RESOURCE_GROUP" \
  --image "$IMAGE" --command python /workspace/train.py --set-env-vars "QUEUE=$QUEUE" "TABICL_DEVICE=cuda" -o none
./scripts/azure/check-gpu-config.sh
echo "starting"
EXEC="$(az containerapp job start --name "$AZURE_TRAINING_JOB" --resource-group "$AZURE_RESOURCE_GROUP" --query name -o tsv)"
echo "execution: $EXEC"
echo "$EXEC" > .azure-last-execution
