# Training on the Azure A100 (set up 2026-10-19)

The model box is a serverless GPU job on Azure Container Apps, driven from a Mac with the Azure CLI. There is no SSH; a run is: build an image in the cloud registry, point the job at it, start one execution, read the log. The known-good GPU app `a100-test` is left as it was (scale 0 to 1, public quick-start image) and serves as the reference for the GPU resource shape.

## Resources

| what | name | notes |
|---|---|---|
| subscription | Azure subscription 1 (`5a166bfa-…`), tenant `7c6cae0e-…` (UC Davis) | sign in with `az login --tenant 7c6cae0e-643e-4fb4-b8e0-d54544311594`; the tenant blocks the device-code flow, the browser flow works |
| resource group | `ml-training-rg`, westus3 | |
| environment | `ml-training-env` | workload profiles `Consumption` and `NC24-A100`; serverless A100 quota 3 |
| GPU smoke-test app | `a100-test` | profile NC24-A100, cpu 24, memory 220Gi, no gpu field: the profile supplies the GPU |
| registry | `mltraininga00acf.azurecr.io` (Basic) | the job pulls with the registry admin credential stored as a job secret; managed-identity AcrPull needs a role assignment the signed-in account cannot write |
| job | `a100-training` | manual trigger, parallelism 1, retry 0, timeout 43200 s, profile NC24-A100, cpu 24, memory 220Gi, container `trainer`, system-assigned identity |
| logs | Log Analytics workspace of the environment, table `ContainerAppConsoleLogs_CL` | the live stream (`job logs show`) only works while a replica runs |

Names and the registry login server live in `.azure-training.env` at the repo root (gitignored, no secrets). No storage account yet: results come back through the log as encoded blocks (below), which is enough for the lab's few hundred KB.

## Verified on 2026-10-19

1. `a100-training` with the public GPU quick-start image and the command `nvidia-smi`: NVIDIA A100 80GB PCIe, driver 580.159.04, CUDA 13.0, execution `a100-training-ftu86et`, Succeeded.
2. The trainer image (`infra/trainer/Dockerfile`: pytorch 2.5.1 CUDA 12.4 runtime, xgboost, tabicl, the lab code and data bundle, non-root user, TabICL checkpoint pre-cached): `torch.cuda.is_available()` True, device NVIDIA A100 80GB PCIe, count 1. Execution `a100-training-z36en9t` ran queue G (three variants of the champion's TabICL recipe at 32 estimators × 3 permutations) in 18 minutes; folds take 5–11 s each on the GPU against 15 s per 8-estimator fit on the Mac.

## Run a training job

```bash
az login --tenant 7c6cae0e-643e-4fb4-b8e0-d54544311594      # browser flow
DRAFTDB_DATA=../DraftDB-Data infra/trainer/make_bundle.sh     # lab code + public answers + v4.22 inputs + names-stripped identity
scripts/azure/train.sh G                                      # az acr build, job update, GPU config check, start; prints the execution name
scripts/azure/status.sh
scripts/azure/collect_results.py experiments/lab_<date>/gpu   # decodes the RESULT blocks from Log Analytics (allow a few minutes of ingestion)
```

`scripts/azure/check-gpu-config.sh` fails closed if the job has left the NC24-A100 profile or lost the 24 cpu / 220Gi shape; `train.sh` runs it before every start. Queues are defined in `tools/lab/run_queue.py` (QUEUE env var on the job). Each variant's `exp_<name>.json`, boards and `experiments.jsonl` are written to `/workspace/out` in the container and printed by `infra/trainer/train.py` as `RESULT_BEGIN name` / `R <base64 gzip>` / `RESULT_END` lines.

## Known failure: a replica without a GPU (2026-10-20)

Execution `a100-training-jnssvkx` (queue S) started on the NC24-A100 profile with the GPU config check passing, yet inside the container `torch.cuda.is_available()` was False (`DRAFTDB_TRAIN_START {"cuda": false, "device": null, "count": 1}`) and every TabICL fit raised "No CUDA GPUs are available"; the job still reported Succeeded with zero result files. Starting the same job again, unchanged, ran normally (`a100-training-thv74rm`). `infra/trainer/train.py` now exits with code 3 and the marker `DRAFTDB_NO_GPU` in that state, so look for that line (or a zero-file download) and start the execution again. The polling loop in the pipeline scripts can also stall on a hung `az containerapp job execution show`; killing that one CLI process lets the loop continue.

## Rules kept from the handoff

Do not delete or repurpose `a100-test`; do not use `sleep infinity` as a job command; prefer immutable image tags (`<git sha>-<utc timestamp>`); `az acr build` instead of local Docker (the Mac is arm64, the job is linux/amd64); no credentials in git (the registry password exists only as a job secret); one GPU per replica, one replica per execution until a sweep is wanted (then up to three executions in parallel, within the quota).

## Costs

The A100 is billed only while an execution runs (the two runs above: about 4 and 18 minutes). The Basic registry is a fixed small monthly charge. `a100-test` stays at zero replicas when idle.
