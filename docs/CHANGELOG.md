# Changelog

## 2026-10-20 — H7: the clean undrafted universe

| change | files | what and why |
|---|---|---|
| Early-entry collector | `DraftDB-Data/early_entry_collect.py`, `early_entry_raw/` (cached NBA releases with provenance), `early_entry_{2019..2025}.csv` | The league's initial early-entry lists, every withdrawal release and the post-deadline remaining lists (nba.com/news, pr.nba.com; Hoops Rumors' copy of the 2021 final list), parsed from HTML tables, bullet rows and padded columns, reconciled to the league's stated counts for all seven years. |
| Universe builder | `DraftDB-Data/universe_build.py`, `v4_build/universe_v422/` (inputs, outcomes, rebuilt-row checks, summary, README), `identity/universe_names.csv` | The pre-draft universe per class from pre-draft sources only (entrants + D-I seniors at 40% of minutes who did not play again in Y+1 + EuroLeague/EuroCup 22-year-olds); rows rebuilt from Torvik, RSCI and the Torvik-context builder where no v4.22 row exists, verified against real rows; 0-WAR outcomes for players who never played. |
| Clean-universe evaluation | `tools/lab/evaluate_undrafted.py` (new), `experiments/lab_2026-10-19/undrafted_eval.json` | Reproduces the earlier outcome-selected check from the saved base boards and adds `clean_strict` (the universe scored by the strict engine), `clean_strict_uniform` (every college player on a rebuilt row) and `consistency` (real vs rebuilt rows swapped into the pool). Result: on the full pool the strict board ties the real draft (Spearman 0.100 vs 0.106, pick-value 67.8 vs 68.8) and would have spent 35 of 60 picks per class on undrafted seniors who produced 10.3 WAR against 26.5 for the drafted players displaced; 9 gems found, 11 missed. |
| Dashboard | `tools/lab/build_dashboard.py`, `tools/lab/dashboard_template.html`, `lab.html` | Section 5b: the universe rule, the per-class table, gems found and missed, and every undrafted player the model's first 60 would have taken with what he produced. |
| Docs | `docs/METRICS.md` (clean-universe rule and results), `docs/EXPERIMENTS.md` (2026-10-20 H7 entry), `docs/MAKE_IT_BETTER.md` (H7 done, the undrafted line in section 1 replaced), `README.md`, `runs.html`, `tools/lab/README.md`, `DraftDB-Data/README.md` | The undrafted claim now rests on a universe built with no outcome information; the old "indicative only" numbers stay as the earlier check. |

Not changed: the champion model, its verified numbers, the vault, the ledger, the team pages.

## 2026-10-20 — H6 top-of-board specialist tested on the A100 and closed

| change | files | what and why |
|---|---|---|
| Gene `star` and top-5 hits in the lab engine | `tools/lab/engine.py`, `tools/lab/run_queue.py` (queue S), `tools/lab/summarize_experiments.py`, `tools/lab/README.md` | An XGBoost classifier for "top-3 discounted-WAR outcome in class" (class-balanced, same features and guards as the XGBoost member) whose probability rank is blended into the board for the consensus top 10 only, at weight `star`; fold scores and blind summaries now record top-5 hits. Queue S (star 0 / 0.25 / 0.5 / 0.75 / 1.0, and 0.5 with risk=mean) ran on the A100 (execution a100-training-thv74rm): every weight loses on the folds under all three judges and lowers blind top-5 hits, so H6 is closed. Entry in `docs/EXPERIMENTS.md`; H6 marked closed in `docs/MAKE_IT_BETTER.md`. |
| Results and dashboard | `experiments/lab_2026-10-19/gpu_s/` (new), `tools/lab/build_dashboard.py`, `tools/lab/dashboard_template.html`, `lab.html` | The six variants' JSON, boards and `experiments.jsonl`; the experiments table on lab.html gains a "Folds top-5 hits" column (– for runs that predate it) and lists the queue S rows as [A100]. |
| A/B screen of recipe changes, queues T and U | `tools/lab/engine.py` (genes `recency`, `xgbm`, `thin_q`, `wgrid`, `mono`, `xgb_lr`, `xgb_n`), `tools/lab/run_queue.py` (queues T, U), `tools/lab/run_queue_cpu.py` (new), `experiments/lab_2026-10-19/gpu_t/`, `gpu_u/`, `cpu_t/` (new), `lab.html` | Fifteen single changes with a stated mechanism (recency weights, both XGBoost objectives in one stack, more bagging, ridge strength, the international mask, the thin/rich split, a finer weight grid, monotone constraints, a slower XGBoost) run on the A100 and replicated on the CPU recipe, then seven combinations of the positive arms. Nothing clears 6 of 7 folds on Spearman; the positive arms are +0.01 and do not add. Entries in `docs/EXPERIMENTS.md`; `docs/MAKE_IT_BETTER.md` gains the closed knobs, a ceiling note (5b) and the lab result under H3. |
| Trainer fails fast without a GPU | `infra/trainer/train.py` | The first execution of queue S came up on the NC24-A100 profile with no usable CUDA device and every TabICL fit crashed; the entrypoint now exits with `DRAFTDB_NO_GPU` when `TABICL_DEVICE=cuda` and torch sees no GPU, so a bad replica costs seconds, not a run. |

Not changed: the champion model, its verified numbers, the vault, the ledger, the site pages.

## 2026-10-19 — response to the October review

An outside reviewer (a front-office analyst) made five points: the team pages summed WAR while counting the same player on several cards; rank swaps were weighted equally; no comparison to mock consensus or simpler baselines; only drafted players were ranked; early-career WAR is noisy. Each change below names the point it answers. Numbers come from `experiments/lab_2026-10-19/`; nothing in the model, the vault or the published headline changed.

| change | files | answers | what and why |
|---|---|---|---|
| Expanding-window board as a file | `model/board_expanding_2019_2025.csv` | 1 | The board the team pages use existed only on the training box; reconstructed from the ranks in `data/team_cards_audit.csv` (385 of 392 drafted players; 2019 equals the strict board). |
| Pick-value curve | `data/pick_value_curve.json` | 2 | Expected five-season WAR by overall pick, isotonic fit on drafted players 2000–2018, smoothed, floored at zero. Slot weight for the pick-value metric and "slot value" on the team pages. |
| Sequential team cards | `data/build_team_cards_sequential.py` (new), `data/team_cards.json`, `data/team_cards_audit.csv`, `tools/build_team_pages.py`, `teams/*.html` | 1 | The real draft replayed in pick order, each player taken once, replaces best-available-with-repeats. Golden State: 101 vs 18 becomes 52 vs 16 at slots worth 70. Cards whose slot has no board player left (28 late picks) are kept without a model pick and excluded from totals. The "Model agreed" rule is off (a better player taken earlier in the replay is not agreement). Slot value and surplus added; the caveats gain an item on early-career WAR. |
| Metrics definitions | `docs/METRICS.md` (new) | 2, 3, 5 | Spearman, top-weighted Spearman, value of order (linear and pick-value weights), WAR above slot, NDCG, top-N hits, the sequential rule, paired bootstrap, the expanded-pool rule, the horizon table, and the results. |
| Runs page | `runs.html`, `js/runs.js` | 2, 3, 4, 5 | New section "Scored the way a front office scores it" with the boards-and-metrics table, the sequential redraft by pick range, the horizon numbers, the consensus and undrafted paragraphs; the pick-value note under the headline; the training window corrected from 2010–2018 to 2007–2018 (the text was stale; the model has trained on 2007–2018 since 2026-09-06). |
| README | `README.md` | all | "Scored the way a front office scores it" subsection; repository layout rows for the new files. |
| Experiments log | `docs/EXPERIMENTS.md` | 2, 5 | 2026-10-19 entry: 22 engine variants on a local CPU re-implementation (folds first, one blind look each), none adopted; the quantile objective and the consensus anchors as the two readings. |
| Evaluation lab | `tools/lab/` (new), `experiments/lab_2026-10-19/` (new), `lab.html` (new) | all | Loaders, metrics, board scoring, the CPU engine, the variant queue, the dashboard builder and its outputs; `lab.html` is the test-bench page linked from the runs page. Needs the private data repo only for names, the multi-site consensus and the v4.22 engine inputs. |
| Next steps | `docs/MAKE_IT_BETTER.md` | all | Ranked hypotheses with evidence and tests; closed lines; data candidates. |

Not changed: the champion model, its verified numbers (41.8% strict / 50.6% expanding), the vault, the ledger, `index.html`.
