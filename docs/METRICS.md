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

**Expanded pool (outcome-selected, the earlier check).** The default pool is drafted players. The first undrafted check added the test files' undrafted rows, who were collected knowing who reached the NBA, and gave the real draft pick 61 for all of them. That pool is enriched for success, so it is indicative only and no score is quoted from it.

**Clean universe (H7, 2026-10-20).** The pre-draft universe of class Y is built with no knowledge of later NBA participation. A player is in it if, before draft night, he was (a) on the league's final early-entry list for Y, college or international (the NBA's initial list minus every withdrawal release, reconciled to the league's stated remaining counts; `DraftDB-Data/early_entry_collect.py`); or (b) a Division-I senior in season Y with at least 40% of his team's minutes and 10 games (Torvik), minus the seniors who played Division-I basketball again in Y+1, who had eligibility left and so were not in that year's pool; or (c) a EuroLeague or EuroCup player listed at age 22 (automatic international eligibility) who was in no earlier class and had no NBA record. Every drafted player is in by definition, since the draft enumerates them; the few drafted outside (a)–(c) (G League Ignite, Overtime Elite, domestic leagues abroad) are counted. Undrafted players enter only through (a)–(c), and Torvik's post-draft `pick` column is never read. The test files' undrafted rows that the rule does not reach are dropped: on inspection they are players who had left college years earlier and were assigned to the class of their NBA debut, which is exactly the outcome selection the rule removes. Inputs: the v4.22 row where one exists, otherwise a row rebuilt from the same public sources (Torvik for the college line, verified column by column against the real rows; RSCI for the recruiting rank; the Torvik-context builder for the tc_ block; the EuroLeague exports for the intl_ block). Players with no Division-I or EuroLeague season (non-D-I colleges, domestic leagues abroad) are listed but not scored. The strict engine (train 2007–2018, base recipe, fold-learned weights) scores the pool; nothing was selected on 2019–2025. The real draft gets pick 61 for every undrafted player; WAR is summed over the class horizon k, so an undrafted player who never played scores 0. Two checks accompany the numbers: a consistency check (drafted college players scored on their real rows and again with rebuilt rows swapped into the pool) and a uniform-rows sensitivity (every college player scored on a rebuilt row). Results in section 4 and on `lab.html`; the builder and its outputs are in the private data repo (`universe_build.py`, `v4_build/universe_v422/`).

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

**Clean universe (H7, strict board, every undrafted player at pick 61 for the teams).** Rule in section 3; the per-class table and the gems are in [`EXPERIMENTS.md`](EXPERIMENTS.md) (2026-10-20) and on `lab.html`.

| class | k | scored (drafted / undrafted) | Spearman model | teams | pick-value model | teams | undrafted in model top 60 (played) | their WAR | WAR of the drafted players displaced |
|---|---|---|---|---|---|---|---|---|---|
| 2019 | 5 | 688 (58 / 630) | 0.093 | 0.125 | 61.2 | 67.4 | 38 (9) | 12.8 | 41.5 |
| 2020 | 5 | 674 (58 / 616) | 0.088 | 0.223 | 49.5 | 66.7 | 36 (5) | 0.3 | 107.4 |
| 2021 | 5 | 334 (56 / 278) | 0.305 | 0.360 | 78.1 | 72.7 | 33 (15) | 48.9 | 38.0 |
| 2022 | 4 | 620 (52 / 568) | 0.106 | 0.071 | 71.0 | 69.6 | 38 (6) | 8.9 | 27.4 |
| 2023 | 3 | 657 (56 / 601) | 0.047 | −0.063 | 59.9 | 74.3 | 34 (7) | 2.7 | −6.8 |
| 2024 | 2 | 648 (55 / 593) | 0.080 | 0.061 | 74.8 | 59.5 | 30 (6) | −0.5 | −7.3 |
| 2025 | 1 | 823 (57 / 766) | −0.016 | −0.033 | 80.3 | 71.4 | 37 (6) | −0.8 | −14.9 |
| mean | | | 0.100 | 0.106 | 67.8 | 68.8 | 35.1 (7.7) | 10.3 | 26.5 |

Consistency: drafted college players scored on their real rows and again with rows rebuilt from Torvik swapped into the pool keep the same count in the first 60 within three (Spearman between the two scorings 0.77–0.96); with every college player on a rebuilt row the means are 0.098 Spearman, 34.6 undrafted in the first 60, 9.3 WAR.

## 5. How to read it

The Spearman lead of about 24 points over the teams becomes about 3 points of pick-value order, because pick-value weights count the top of the board heavily and the middle and late slots little, and the model's Spearman gain is mostly in the middle and late ordering. A 14-input ridge with the consensus matches the full model on the pick-weighted metrics (73.1 against 73.2 strict; WAR above slot 742 against 731), so the complex stack is buying middle and late ordering. Three of the seven classes (2023-2025) have one to three seasons of WAR, and they lean toward the model: the real draft order correlates 0.07 with first-season WAR and 0.32 with five-season WAR, so early WAR is a poor yardstick for teams that draft for years three to five. On 2019 and 2020 the teams overtake the model on pick-value once four or five seasons are in; 2021 holds. Read the three complete classes before the others.

Given the mock consensus, the model's partial Spearman with WAR is 0.18-0.66 by class; the consensus given the model is about zero in five of seven classes (range -0.17 to 0.22). The model is not consensus plus noise.

See also: [`lab.html`](../lab.html), [`tools/lab/README.md`](../tools/lab/README.md), [`MAKE_IT_BETTER.md`](MAKE_IT_BETTER.md).


## Reading a team page: why the team wins a card (added 2026-10-21)

The league-wide sequential replay is zero-sum: a player goes to the first team on the clock whose board has him highest. A team that really took a player the model also rated above that slot therefore loses the card to the replay, not to the model's judgement. Each card the team wins is tagged with one cause (`loss_reason` in `data/team_cards.json`):

| cause | rule | league 2019-2025 | WAR lost |
|---|---|---|---|
| agreed, taken earlier | the model's rank of the real pick is at or above the slot | 113 of 198 | 552 |
| genuine miss | the model ranked the real pick below the slot; its own pick scored 0 or more | 35 | 218 |
| miss, negative-WAR pick | the model ranked the real pick below the slot and its own pick played and produced negative WAR | 50 | 166 (101 floored at zero) |

The teams with the largest raw deficits are mostly the first kind: Charlotte 8 of 14 cards, Oklahoma City 7 of 11, Memphis 6 of 11, the Clippers 5 of 7, Boston 6 of 9. The genuine misses are the known top-of-board pattern (Edwards, Wembanyama, Barnes, Cunningham, LaMelo, Garland, Maxey, Quickley): young, pedigreed players with ordinary college production, which the closed H6 and pedigree lines did not fix. Floored-at-zero totals (`war_actual_total_pos`, `war_model_total_pos`) are shown beside the raw totals because a bust who played scores below a stash who never did.

A team-conditional replay (the team uses the model at its own slots while every other team drafts as it did) was computed and not adopted: it lets a team pass on a lottery pick and take him later because no other team may react, which overstates the model (league 2,162 WAR against 1,129).
