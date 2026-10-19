#!/usr/bin/env bash
# Assemble infra/trainer/bundle (gitignored) for the A100 trainer image:
#   draftdb/tools/lab, draftdb/data/answers, draftdb/data/train_2000_2018.csv   (this repo)
#   DraftDB-Data/v4_build/staging_v422, a names-stripped identity file           (private data repo, DRAFTDB_DATA)
# The stripped identity keeps pid, draft_year, actual_pick only: no player name reaches the registry or the job.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../.. && pwd)"; DATA="${DRAFTDB_DATA:-$(cd "$REPO/.." && pwd)/DraftDB-Data}"
rm -rf bundle && mkdir -p bundle/draftdb/data bundle/DraftDB-Data/identity bundle/DraftDB-Data/v4_build
cp -r "$REPO/tools" bundle/draftdb/tools
cp -r "$REPO/data/answers" "$REPO/data/train_2000_2018.csv" bundle/draftdb/data/
cp -r "$DATA/v4_build/staging_v422" bundle/DraftDB-Data/v4_build/staging_v422
python3 - "$DATA/identity/tabular_names.csv" bundle/DraftDB-Data/identity/tabular_names.csv <<'PY'
import csv, sys
r = csv.DictReader(open(sys.argv[1])); w = csv.DictWriter(open(sys.argv[2], "w"), fieldnames=r.fieldnames); w.writeheader()
for row in r:
    row["player_name"] = ""; row["nba_id"] = ""; w.writerow(row)
PY
cp train.py bundle/train.py
du -sh bundle
