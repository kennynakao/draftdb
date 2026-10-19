# Changelog

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
