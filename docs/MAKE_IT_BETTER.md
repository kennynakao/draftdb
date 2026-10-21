# Make it better: what to do next (written 2026-10-18)

Status of the model after an outside review (Kirk, October 2026) and a one-day lab pass that re-scored every board under the metrics he asked for and ran 22 engine variants locally. This file is the queue for the next working session. Everything here was measured; nothing is proposed on taste alone. The lab code (metrics, sequential redraft, a CPU re-implementation of the engine, the dashboard builder) sits outside the repo for now and should be added under `tools/lab/` once the vault scoring function also returns the new metrics.

## 1. Where the model stands once the metrics change

Same seven classes, same players, four ways of scoring an order. "Teams" is the real draft on the same players.

| board | Spearman (the README headline) | teams | pick-value order (100 perfect, 50 random) | teams | WAR above slot | teams |
|---|---|---|---|---|---|---|
| current model, expanding | 0.496 | 0.254 | 72.3 | 69.6 | 676 | 667 |
| current model, strict | 0.418 | 0.263 | 73.2 | 69.9 | 731 | 683 |
| Colin baseline, strict | 0.522 | 0.262 | 73.8 | 70.0 | 732 | 683 |
| 14-input ridge + consensus (fit 2007–18) | 0.378 | 0.263 | 73.1 | 69.9 | 742 | 683 |
| mock consensus, 2–6 dated publishers | 0.231 | 0.263 | 70.2 | 69.9 | 682 | 683 |

- Pick-value order weights every slot by the expected 5-season WAR of that pick (isotonic fit on 2000–2018: pick 1 ≈ 21, pick 10 ≈ 8, pick 30 ≈ 5, pick 50 ≈ 2). WAR above slot is the same thing in WAR units relative to a random order.
- The 24-point Spearman lead over the teams becomes a 3-point lead on pick-value order. A ridge on 14 inputs matches the full model on the pick-weighted metrics. The stack is buying middle-and-late ordering, which the pick-value curve says is worth little.
- Paired bootstrap within class: the model's Spearman lead over the teams clears zero in 5 of 7 classes; its pick-value lead in 3 (2021, 2024, 2025).
- Sequential redraft (replay the draft in pick order, each player taken once; replaces the team pages' best-available-with-repeats): the model's pick beats the real pick at 173 of 385 slots and loses at 198. It gains at picks 6–10 (210 vs 135 WAR) and 31–45 (169 vs 97), the teams keep 1–5 (379 vs 342) and 46–60 (53 vs −36). Golden State: 101 vs 18 on the site becomes 52 vs 16 at the same twelve slots (slot value 70).
- Horizon: on the 2007–2018 classes the real draft order correlates 0.07 with first-season WAR and 0.32 with five-season WAR. A one-season ranking agrees with the five-season ranking at 0.60, two at 0.82, three at 0.91. The 2023–2025 numbers (one to three seasons) are therefore biased toward a front-weighted model, not merely noisy. In 2019 and 2020 the teams overtake the model on pick-value once four or five seasons are in; 2021 holds.
- Consensus: the model keeps a partial correlation of 0.18–0.66 with WAR given the consensus; the consensus given the model is about zero in five of seven classes. The model is not consensus plus noise.
- Undrafted (H7, done 2026-10-20): on a pre-draft universe built with no outcome information (early entrants, Division-I seniors, EuroLeague/EuroCup 22-year-olds; 334–823 scored players per class) the strict board ties the real draft, Spearman 0.100 vs 0.106 and pick-value 67.8 vs 68.8, and would have spent 35 of its 60 picks per class on players who went undrafted, almost all seniors, who produced 10.3 WAR per class against 26.5 for the drafted players displaced. Gems found: Konchar, Wade, Reaves, Alvarado, Goodwin, Huff, Justin Champagnie, Keon Ellis, Craig Porter; missed: Dort, Strus, Naz Reid, Mathews, Caleb Martin, Terence Davis, Hauser, Julian Champagnie, Pippen. The earlier numbers from the outcome-selected rows are superseded.

## 2. What the engine variants said

Local re-implementation on data v4.22 with the champion's input families (base 169 + `sc_` + `tc_`), international-line mask, shrinkage, shoot/prod features, Gaussian-rank label, ridge-on-top-100 + quantile XGBoost, fold-learned rank weights. It reproduces the box's own XGBoost+ridge ablation (strict 0.44 vs 0.395, expanding 0.47 vs 0.475). Folds = the seven season-consistent walk-forward folds 2012–2018, averaged over horizons 1–5. Strict and expanding are one blind look per variant, reported for all of them, used to select none of them.

| variant | folds Spearman (won of 7) | folds pick-value (won) | strict Sp. / PV | expanding Sp. / PV |
|---|---|---|---|---|
| base | 0.354 | 65.8 | 0.441 / 71.6 | 0.470 / 73.7 |
| risk=mean (squared error instead of quantile 0.25 in XGBoost) | 0.374 (3) | 65.2 (3) | 0.432 / 74.1 | 0.487 / 75.5 |
| combine participation flags (new) | 0.378 (5) | 65.8 (3) | 0.419 / 72.2 | 0.457 / 72.1 |
| consensus mapped through the pick-value curve (new) | 0.370 (3) | 65.7 (4) | 0.429 / 73.0 | 0.437 / 71.2 |
| top-weighted training rows, 0.5 | 0.372 (4) | 66.4 (4) | 0.387 / 72.9 | 0.416 / 73.7 |
| top-weighted training rows, 1.0 | 0.359 (4) | 65.8 (4) | 0.343 / 72.3 | 0.353 / 72.5 |
| xgb depth 4 | 0.356 (5) | 65.6 (3) | 0.439 / 72.0 | 0.474 / 74.1 |
| label weights uniform / disc70 / late / back | 0.341–0.353 | 64.5–64.7 | 0.387–0.444 | 0.427–0.451 |
| ridge only (also what a pick-value stack objective picks) | 0.350 (3) | 66.3 (4) | 0.418 / 72.1 | 0.456 / 72.7 |
| xgb only | 0.291 (1) | 58.7 (1) | 0.442 / 71.6 | 0.439 / 69.3 |
| consensus top-3 anchored into the model's top 5 | 0.341 (0) | 63.7 (0) | 0.444 / 73.2 | 0.469 / 75.8 |
| consensus top-5 anchored | 0.341 (0) | 64.3 (0) | 0.441 / 72.9 | 0.470 / 75.4 |
| consensus prior + learned residual, 0.5 / 1.0 | 0.293 / 0.221 | 58.6 / 57.4 | 0.462 / 68.6, 0.412 / 67.0 | 0.429 / 65.1, 0.399 / 64.6 |

Reading it the way the repo's own log reads such tables: fold gains of 0.02 on three to five folds have not transferred to the blind classes here either (top-weighting, combine flags, consensus value). Two things stand out and only two.

1. The quantile objective in XGBoost costs Spearman on the folds (0.374 vs 0.354) and the member is weak alone (0.291 vs ridge 0.350). On the box the XGBoost member carries weight zero on rich rows, so this never mattered there; it matters if XGBoost is to carry weight again.
2. Anchoring the consensus top three raises blind pick-value by two points while the folds fall on both metrics. One look, gate-negative; it says the top of the board is where the teams and the consensus still know something, which the sequential redraft at picks 1–5 also says.

## 3. Hypotheses, ranked

Each entry: the claim, the evidence, the change, how it is judged, what counts as success.

### H1. Put pick-value order in the vault and in the gate (highest value, lowest cost)
- Claim: Spearman rewards middle-and-late ordering; teams and reviewers price the top. The two metrics disagree about several variants above, so the metric decides the model.
- Change: `vault.score()` returns `{ic, draft_ic, vo_pick, war_above_slot, top5, top10, n}`; the ledger records all; the gate requires a candidate to win the recency-weighted fold mean on Spearman and not lose on pick-value order (or a declared joint objective, written down before the next search starts).
- Judge: no new data; a code change plus a re-run of the champion for the baseline numbers.
- Success: every future keep is reported on both; the README headline shows both.

### H2. Fix the horizon framing before adding anything (cost near zero)
- Claim: scoring 2023–2025 on one to three seasons flatters a front-weighted model against teams that draft for years three to five (real-draft correlation 0.07 at one season, 0.32 at five).
- Change: README and site show per-class results beside their horizon, mark 2023–2025 provisional, add the "teams at k" curve, and headline the three complete classes separately. Keep the discounted label (uniform, late and back-weighted labels all lost on the folds today, as they did on the box), but say why the discount is there.
- Success: a reader like Kirk cannot mistake a one-season class for a verdict.

### H3. Retest the XGBoost objective inside the full recipe (done 2026-10-21: same-sign gain under three seed sets, 4–5 of 7 folds, expanding +0.03 to +0.05 blind; below the 6-of-7 rule, recommended for the expanding recipe, see EXPERIMENTS)
- Claim: squared error beats quantile 0.25 for the member (+0.020 folds, +0.017 expanding Spearman, +1.8 expanding pick-value here).
- Change: `risk=mean` on the gen99 genome, three seed sets through the vault, with the stack weights re-learned so XGBoost can take weight.
- Judge: the existing gate. It won only 3 of 7 folds here, so it may fail the 6-of-7 rule; if it does, drop the XGBoost member rather than keep a zero-weight one.
- Success: a verified gain on either metric with no loss on the other; otherwise a simpler stack.
- Lab result (2026-10-20, queues T and U on the A100): risk=mean with nine bags, a slower XGBoost (0.01 × 1,600), the thin/rich split at the lower third, and both objectives in one stack each add +0.007 to +0.012 Spearman and +1 to +2 pick-value on the folds, win 3 to 4 of 7 Spearman folds, and do not add when combined (0.407–0.411 for every pair or triple). Still a vault candidate, not a fold winner; the simpler-stack branch of this item is the likelier outcome.

### H4. Rebuild the team pages on the sequential redraft (site, one evening)
- Claim: the current cards double count; the review called it out.
- Change: `data/build_team_cards.py` takes the sequential assignment (one player, one slot) and adds slot value and surplus per team; the summary becomes "WAR vs slot value" for both the team and the model. The Warriors example changes from 101 vs 18 to 52 vs 16, slot value 70.
- Success: no player appears as the model's pick at more than one slot in a class.

### H5. A value label that does not punish playing time (cheap fold test)
- Claim: the model's last-ranked fifteen produced −36 WAR against the teams' +53 at picks 46–60, and the Spearman gap narrows from 24 to 14 points when WAR is floored at zero. Negative WAR says "played a lot, badly, often on a bad team", which is partly team context.
- Change: label candidates `max(WAR, 0)` and `max(WAR, 0) + λ·minutes/1000` with λ small; evaluate on the folds under both metrics and on `war_pos` as a third.
- Risk: a floored label removes the bust signal that the quantile objective was meant to capture; test, do not assume.

### H6. A top-of-board specialist judged on pick-value — closed 2026-10-20
- Claim: at picks 1–5 the teams still beat the model (379 vs 342 WAR); the model's README already calls the top a dead heat.
- Change: a classifier for "top-3 outcome in class" (class-balanced), blended into the board only inside the consensus top 10; or the anchor above with a learned k. Both were rejected on Spearman before; neither has been judged on pick-value, which is where they would show.
- Judge: folds on pick-value order and top-5 hits, then the vault. The anchor fell on the folds today, so the classifier is the better bet.
- Result (A100 queue S, TabICL 32×3 recipe, `engine.py` gene `star`): the classifier blended at 0.25 / 0.5 / 0.75 / 1.0 loses on the folds under all three judges (Spearman 0.399 → 0.397–0.383, pick-value 65.7 → 64.9–61.9, top-5 hits 1.54 → 1.54–1.31; at most 3 of 7 folds won on any metric) and lowers blind top-5 hits (strict 2.43 → 2.14–2.29 against the teams' 2.43). On its own the classifier ranks the class at 0.36 fold Spearman against 0.43 for the ridge; with risk=mean it also falls short of risk=mean alone. Closed; the full table is in `EXPERIMENTS.md` (2026-10-20, queue S).

### H7. A clean undrafted universe — done 2026-10-20
- Claim: the test files' undrafted rows are outcome-selected, so the model's undrafted hit rate cannot be measured honestly, and the review asked for it.
- Done: the universe per class 2019–2025 from the league's final early-entry lists, Division-I seniors at 40% of team minutes who did not play college ball the next season, and EuroLeague/EuroCup 22-year-olds, with no knowledge of later NBA participation; inputs rebuilt from Torvik/RSCI/EuroLeague where no v4.22 row exists and verified against real rows; scored by the strict board. Rule, sources and tables in `METRICS.md` (Clean universe), `EXPERIMENTS.md` (2026-10-20 H7) and section 5b of `lab.html`; collector, builder and files in the private data repo.
- Result: Spearman 0.100 vs 0.106 for the teams, pick-value 67.8 vs 68.8; 35 undrafted players per class in the model's first 60 (10.3 WAR, 7.7 of them played) against 26.5 WAR for the drafted players displaced; 9 gems found, 11 missed. The model's drafted-only lead does not survive the open pool: it has no notion of who else is draftable and rates productive 23-year-old seniors like second-round picks.
- Partial: the international block is EuroLeague/EuroCup only (declared internationals in domestic leagues have no inputs); senior eligibility is proxied by the next season's roster; 2023–2025 have one to three seasons of outcomes. A follow-up with a mechanism: a "draftability" prior (age × class × consensus presence) learned on the folds, tested under the gate, since the open pool is now measurable.

### H7b. Draftability prior for the open pool (one look taken 2026-10-21)
- Claim: the open-pool failure is a missing draftability notion, not a talent-ranking failure.
- Evidence: a logistic P(drafted) on age, consensus, RSCI, production and height reaches AUC 0.97–0.996 on the clean universe; blended at λ = 0.25–0.75 with the strict board it lifts open-pool pick-value order from 67.8 to 70.5–71.0 (teams 68.8) and cuts undrafted top-60 picks from 35 to 17–29 per class; rank correlation on the open pool falls (0.10 to 0.05–0.09).
- Status: one blind look, no fold selection possible (the universe exists for 2019–2025 only). To use it, fix λ = 0.5 now and treat 2026 onward as the test; do not tune λ on these classes.

### H8. Variance discipline (cost near zero)
- Publish bootstrap intervals per class for model vs teams on both metrics (the lab code does this), stop reading two-point differences, and prefer seven-class means with their ±0.05 standard error.

### H9. Report-only checks worth one look each
- Rank-average of the current board and Colin's under pick-value (closed on Spearman at 48.6 vs 49.3 expanding; never judged on pick-value, where Colin's board is the best of the set).
- The three complete classes (2019–2021) at k = 5 only, as the number to quote until 2022 completes.

## 4. Closed by today's runs (do not retry without a new reason)
- (2026-10-20, A100) Value labels (WAR floored at zero, with or without a minutes term): fold losses and 5–12 point blind losses. Every collected block switched on alone under the TabICL 32×3 recipe: fold losses on both metrics; the pick-value metric does not change the Spearman-era verdicts. H5 is closed.
- (2026-10-20, A100) H6 top-of-board specialist: a class-balanced top-3 classifier blended into the consensus top 10 at any weight loses on the folds under Spearman, pick-value and top-5 hits, and lowers blind top-5 hits; the classifier alone ranks the class worse than the ridge. The top of the board stays a tie with the teams. Closed.
- (2026-10-20, A100 queues T and U) Recipe knobs: recency weights on the training rows (half-life 4 and 8), ridge strength 30 / 100 / 1000, five TabICL bags, nine XGBoost bags alone, a 0.125 stack-weight grid (identical to the 0.25 optimum), monotone constraints on impact, and every combination of the positive arms. Within ±0.01 Spearman of the base, none winning 6 of 7 folds; the recipe is at a flat optimum. Dropping the international mask is the largest fold loss (−0.042) and the best single blind look (0.477 expanding); recorded, not acted on.
- Label horizon weights (uniform, 0.70, late, back): lose on both fold metrics, as on the box.
- Top-weighted training rows: fold gains on both metrics, blind Spearman falls 5–12 points. The standard pattern of a fold-only gain.
- Consensus prior plus learned residual: folds collapse. Agrees with the reliability line in `EXPERIMENTS.md`: the consensus is worse than the model exactly where the model is unsure.
- Consensus anchors: gate-negative; keep as a diagnostic that the top of the board is the open problem.
- Two new features, consensus mapped through the pick-value curve and combine participation flags: fold gains, blind losses. Noise.
- Deeper XGBoost: nothing.

## 5. Data, honestly
Every block collected since September has failed the gate; today's two features did too. The data investments with a mechanism behind them:
- ~~More dated consensus publishers for 2019–2026~~ Done 2026-10-20: seven more publishers (6–13 per class) leave the consensus at 0.232 Spearman / 70.2 pick-value, unchanged. Closed; the mocks copy each other.
- ~~The undrafted universe (H7)~~ Done 2026-10-20; it changed what can be claimed: on the open pool the model ties the teams.
- Low expected value, listed so they stop resurfacing: CBBD recruiting composite (new information only in the rank 100–250 band), ESPN 100 grades (a disagreement feature at best), hometown density, role transitions.

## 5b. What the October lab says about the ceiling
Fifty-six engine variants on the A100 and forty on the CPU, all judged on the same seven folds under Spearman, pick-value and top-5 hits: no single change or combination clears 6 of 7 folds on Spearman, the positive ones are the size of fold noise (+0.01) and sub-additive, and fold gains do not transfer blind. The inputs are the ceiling, not the learner. What can still change the numbers is a data block with a mechanism the mocks and box scores lack, which nothing collected so far has had.

## 6. Order of work
1. H1 vault metrics and gate (half a day). 2. H4 site fix (an evening). 3. H2 README framing (an hour). 4. H3 one GPU run with the full recipe and three seed sets. 5. H5 label test on the folds. 6. ~~H6 top-of-board specialist under the new gate~~ (closed 2026-10-20). 7. ~~H7 undrafted universe~~ (done 2026-10-20). 8. Data, if anything above moved.
