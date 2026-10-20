# DraftDB evaluation lab

A small bench for scoring draft boards and trying model variants. Outputs of the 2026-10-19 run are in
`experiments/lab_2026-10-19/`; the dashboard built from them is `lab.html` at the repo root.

## Scripts

- `common.py` - loaders (answers, names, consensus, v4.22 inputs), paths, and the pick-value curve.
- `metrics.py` - Spearman, top-weighted Spearman, value of order (linear or pick-value slot weights), WAR above slot,
  NDCG, top-N hits, paired bootstrap between two boards, and the sequential redraft.
- `evaluate_boards.py` - scores every board on the 2019-2025 classes (real draft, model strict/expanding, Colin,
  multi-site consensus, RSCI, college production, small ridge baselines) and writes `boards_eval.json`,
  `undrafted_eval.json` and `sequential_*.csv`. No model is selected here.
- `engine.py` - CPU re-implementation of the training pipeline on data v4.22: XGBoost + ridge + optional TabICL,
  season-consistent walk-forward folds 2012-2018, then strict and expanding blind looks at 2019-2025.
  Writes `exp_<name>.json`, `board_<name>_<mode>.csv` and a line in `experiments.jsonl`.
- `run_queue.py` - named variant queues run one after another through `engine.run`: A and C on the CPU recipe, B/G/H/S on the
  TabICL 32x3 recipe for the A100 job (S = the H6 top-of-board specialist, gene `star`).
- `summarize_experiments.py` - markdown table of every variant against base (folds first, blind looks last); optional
  arguments pick a result subfolder and the base variant (`summarize_experiments.py gpu_s s_star00`).
- `build_dashboard.py` - assembles the JSON outputs into `lab.html` using `dashboard_template.html`.
- `data/` - Colin's board (`colin_board.csv`, `colin_board_raw.csv`).

## Running (from the repo root)

    python tools/lab/evaluate_boards.py            # boards_eval.json, sequential_*.csv
    python tools/lab/engine.py base                # one variant; optional 2nd arg is a JSON genome override
    python tools/lab/engine.py hw_uniform '{"hw": "uniform"}'
    python tools/lab/run_queue.py A                # a named queue (A, C locally; B, G, H, S need TabICL, see docs/AZURE.md)
    python tools/lab/summarize_experiments.py      # markdown table to stdout
    python tools/lab/build_dashboard.py            # writes lab.html

Environment variables: `DRAFTDB_DATA` (private data repo, default `../DraftDB-Data` next to this repo) and
`DRAFTDB_LAB_OUT` (output folder, default `experiments/lab_2026-10-19`).

## What needs the private data repo

- Player names (`identity/tabular_names.csv`): display only; every computation is keyed by pid.
- Dated multi-site mock consensus, used by `evaluate_boards.py`.
- The v4.22 input matrix (`v4_build/staging_v422`), needed by `engine.py` and `run_queue.py`.

From the public repo alone, `build_dashboard.py` and `summarize_experiments.py` run on the committed outputs.
`evaluate_boards.py` needs the private consensus and names, and nothing that needs the engine inputs
(engine, run_queue) runs without them.

## Reading the results

Variant selection uses the walk-forward folds (2012-2018) only. The blind columns (strict and expanding, 2019-2025)
are one look each; do not pick or tune a variant on them.

## Environment

numpy, pandas, scipy, scikit-learn, xgboost (needs libomp on macOS: `brew install libomp`). Optional: torch and
tabicl for the TabICL member (CPU, 8 estimators).
