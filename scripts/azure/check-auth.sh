#!/usr/bin/env bash
set -euo pipefail
az account show >/dev/null
SUB_NAME="$(az account show --query name -o tsv)"
if [[ "$SUB_NAME" != "Azure subscription 1" ]]; then
  echo "Wrong subscription: $SUB_NAME (expected: Azure subscription 1)"; exit 1
fi
echo "Azure auth OK: $SUB_NAME ($(az account show --query user.name -o tsv))"
