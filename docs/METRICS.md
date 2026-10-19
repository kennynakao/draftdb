# Metrics: how an order is scored (2026-10-19)

Every board in this repo is a ranking of the players in a draft class. It is scored against what the players did: WAR over the NBA seasons completed so far (k seasons, 5 for 2019-2021, down to 1 for 2025). This file defines each score, how the pick-value curve is built, and the other rules used by the lab pass of 2026-10-19 (code in [`tools/lab/`](../tools/lab/README.md), outputs in `experiments/lab_2026-10-19/`, charts in [`lab.html`](../lab.html)). The plan that follows from the results is in [`MAKE_IT_BETTER.md`](MAKE_IT_BETTER.md).

Notation: a class has n players. A board puts them in an order, slot j = 1..n. `war_j` is the WAR of the player in slot j, `v_j` the pick value of slot j (below), `R_i` the rank of player i on the board and `Q_i` the rank by WAR (1 = best).

## 1. Definitions

**Order accuracy (Spearman).** Rank correlation between the board and the WAR ranking, averaged over classes. The site shows it as a percentage; the lab tables show the fraction. 1 is a perfect order, 0 is no relationship. Ties get average ranks.

    rho = 1 - 6 * sum_i (R_i - Q_i)^2 / (n * (n^2 - 1))

**Top-weighted Spearman** (Pinto da Costa and Soares, 2005). A disagreement near the top of either ranking costs more than one near the bottom.

    rho_w = 1 - 6 * sum_i (R_i - Q_i)^2 * ((n - R_i + 1) + (n - Q_i + 1)) / (n^4 + n^3 - n^2 - n)

**Value of the order, linear weights** (the site's existing second metric). With slot weights w_j = n - j + 1:

    V = sum_j w_j * war_j
    value = 100 * (V - V_worst) / (V_perfect - V_worst)

`V_perfect` sorts WAR descending into the slots, `V_worst` ascending. 100 is a perfect order, 0 the reverse, about 50 a random one. It is kept for continuity.

**Value of the order, pick-value weights** ("pick-value order"). Same formula with w_j = v_j, the expected five-season WAR of the j-th pick. A swap at slots 2 and 12 now costs much more than one at 40 and 45. This is the version to quote.

**WAR above slot.** The same weighted sum in WAR units, relative to a random order:

    sum_j v_j * war_j  -  mean(v) * sum_j war_j

Positive means good players sit where the expensive picks are. Reported as the mean per class.

**NDCG.** Gains `max(WAR, 0)`, discount 1 / log2(j + 1), divided by the same sum for the ideal order.

**Top-N hits.** How many of the board's first N players are among the N players with the highest WAR (N = 5, 10, 20). Random is N^2 / n.

## 2. The pick-value curve

`data/pick_value_curve.json`. Isotonic regression of five-season WAR on overall pick, drafted players of the 2000-2018 classes (non-increasing in pick), then lightly smoothed. Test classes are not used. Index 1-60; later indices are clamped to 60.

| pick | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expected 5-season WAR | 20.9 | 19.2 | 17.5 | 14.2 | 8.2 | 6.2 | 6.1 | 5.3 | 3.9 | 1.9 | 0.9 |

The curve is flat from about pick 15 to 29, which is why ordering that stretch well earns little under pick-value weights.

## 3. Rules

**Sequential redraft.** Replaces the team pages' earlier "best available at each slot, with repeats". Replay the real draft in overall-pick order. At each slot the team on the clock takes the model's highest-ranked player still available (ties broken by lower WAR first). A player is taken once, so he appears on one card only. Each slot then has a real player and a model player, and WAR is compared slot by slot and by pick range. `data/build_team_cards_sequential.py` builds the team cards from it; slot value is `v` at that pick.

**Paired bootstrap.** Within a class, resample the players with replacement (400 draws), score both boards on each resample and take the difference. A lead "clears zero" if the 95% interval of the difference excludes 0. Done per class for Spearman and for pick-value order, model against the real draft.

**Expanded pool.** The default pool is drafted players. For the undrafted check, undrafted players who reached the NBA are added and the real draft is given pick 61 for all of them, so the teams are ranked below every drafted player. The undrafted rows were collected knowing who reached the NBA, so this is indicative only and no score is quoted from it.

**Horizon.** Test classes carry one to five seasons of WAR. To see what that does, on the complete 2007-2018 classes a k-season WAR ranking is compared with the five-season one, and the real draft order is compared with each k-season ranking:

| seasons k | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| k-season ranking vs 5-season ranking | 0.60 | 0.82 | 0.91 | 0.97 | 1 |
| real draft order vs k-season WAR | 0.07 | 0.19 | 0.24 | 0.29 | 0.32 |

## 4. Results (2019-2025, seven classes, same players)

Spearman is the order accuracy as a fraction. "Teams" is the real draft on the same players (for the first row there is only one). Top-10 hits are out of 10.

| board | Spearman | teams | pick-value order | teams | WAR above slot | teams | top-10 hits |
|---|---|---|---|---|---|---|---|
| NBA teams (real draft) | 0.263 | | 69.9 | | 683 | | 4.4 |
| DraftDB model, expanding window | 0.496 | 0.254 | 72.3 | 69.6 | 676 | 667 | 4.7 |
| DraftDB model, strict | 0.418 | 0.263 | 73.2 | 69.9 | 731 | 683 | 4.9 |
| Colin baseline, strict | 0.522 | 0.262 | 73.8 | 70.0 | 732 | 683 | 5.3 |
| 14-input ridge + consensus (fit 2007-2018) | 0.378 | 0.263 | 73.1 | 69.9 | 742 | 683 | 4.9 |
| 12-input ridge, no consensus | 0.347 | 0.263 | 66.0 | 69.9 | 538 | 683 | |
| Mock consensus (mean of 2-6 publishers) | 0.231 | 0.263 | 70.2 | 69.9 | 682 | 683 | |
| CBS only | 0.298 | 0.300 | | | | | |
| Tankathon only | 0.240 | 0.285 | | | | | |
| ESPN best-available only | 0.197 | 0.251 | | | | | |
| RSCI recruiting rank | 0.248 | 0.443 | | | | | |
| College / intl. impact alone | 0.337 | 0.241 | | | | | |

The consensus is the mean over dated publishers (Tankathon, CBS, ESPN best-available, NBADraft.net, Sporting News, NBA Draft Room), the last Wayback capture before each draft. Single publishers and RSCI cover only the players and classes they cover; "teams" there is the real draft on that subset. The strict and expanding model rows are the published model's two protocols.

Per class (the 2019 expanding window equals strict):

| class | seasons | Spearman: teams | Spearman: model | pick-value: teams | pick-value: model |
|---|---|---|---|---|---|
| 2019 | 5 | 0.40 | 0.44 | 73.4 | 64.0 |
| 2020 | 5 | 0.35 | 0.33 | 64.5 | 58.1 |
| 2021 | 5 | 0.41 | 0.72 | 76.5 | 87.0 |
| 2022 | 4 | 0.26 | 0.53 | 69.8 | 70.5 |
| 2023 | 3 | 0.11 | 0.59 | 75.1 | 68.3 |
| 2024 | 2 | 0.13 | 0.55 | 57.7 | 75.0 |
| 2025 | 1 | 0.18 | 0.47 | 72.5 | 83.2 |

The expanding model beats the teams on pick-value in 4 of 7 classes. Under the paired bootstrap its Spearman lead clears zero in 5 of 7 and its pick-value lead in 3 (2021, 2024, 2025).

Sequential redraft, 385 slots with a model pick: the model's player is better at 173, the team's at 198, 14 tie. By pick range (real picks' WAR against the model's): 1-5, 379 vs 342; 6-10, 135 vs 210; 11-20, 284 vs 292; 21-30, 165 vs 153; 31-45, 97 vs 169; 46-60, 53 vs -36. Golden State: the earlier method gave 101 vs 18; sequential gives 52.1 vs 15.9 over 12 slots whose slot value is 70.3.

Alternate outcomes (mean Spearman): WAR floored at 0, model expanding 0.496, teams 0.359, Colin 0.528. Minutes played: teams 0.596, model 0.34 (minutes follow draft slot).

## 5. How to read it

The Spearman lead of about 24 points over the teams becomes about 3 points of pick-value order, because pick-value weights count the top of the board heavily and the middle and late slots little, and the model's Spearman gain is mostly in the middle and late ordering. A 14-input ridge with the consensus matches the full model on the pick-weighted metrics (73.1 against 73.2 strict; WAR above slot 742 against 731), so the complex stack is buying middle and late ordering. Three of the seven classes (2023-2025) have one to three seasons of WAR, and they lean toward the model: the real draft order correlates 0.07 with first-season WAR and 0.32 with five-season WAR, so early WAR is a poor yardstick for teams that draft for years three to five. On 2019 and 2020 the teams overtake the model on pick-value once four or five seasons are in; 2021 holds. Read the three complete classes before the others.

Given the mock consensus, the model's partial Spearman with WAR is 0.18-0.66 by class; the consensus given the model is about zero in five of seven classes (range -0.17 to 0.22). The model is not consensus plus noise.

See also: [`lab.html`](../lab.html), [`tools/lab/README.md`](../tools/lab/README.md), [`MAKE_IT_BETTER.md`](MAKE_IT_BETTER.md).
