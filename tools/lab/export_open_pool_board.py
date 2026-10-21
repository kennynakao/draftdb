"""Export the strict engine's ranking of the clean pre-draft universe (H7) as a board file for the site.

Writes data/open_pool_board_2019_2025.csv: one row per scored universe member (inputs present, outcome known),
columns year, model_rank, pid, player, was_drafted, actual_pick, war (k-season WAR to date, 0 for a player who
never reached the NBA), source (early_entry / senior / intl22 ...), route (test_row / torvik_rebuild / euro_rebuild),
score. Same scoring as evaluate_undrafted.clean_strict (train 2007-2018, base recipe, fold-learned weights); nothing
is selected on 2019-2025. Names for the new universe pids come from DraftDB-Data/identity/universe_names.csv.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import evaluate_undrafted as U

score, train_pids = U.strict_fit()
rows = []
for y in C.TEST_YEARS:
    X = pd.read_csv(f"{U.UNI}/universe_{y}_inputs.csv", low_memory=False)
    O = pd.read_csv(f"{U.UNI}/universe_{y}_outcomes.csv")
    assert not (set(X.pid) & train_pids)
    X["draft_year"] = y
    O = O.merge(pd.DataFrame(dict(pid=X.pid.values, score=score(X))), on="pid")
    k = C.horizon(y)
    O["war"] = O[C.S[:k]].fillna(0).sum(axis=1).where(O.outcome_known == 1)
    pool = O[(O.has_inputs == 1) & (O.outcome_known == 1)].sort_values("score", ascending=False).reset_index(drop=True)
    pool["model_rank"] = np.arange(1, len(pool) + 1)
    for r in pool.itertuples():
        rows.append(dict(year=y, model_rank=int(r.model_rank), pid=r.pid, player=U.NAMES.get(r.pid, ""), was_drafted=int(r.was_drafted),
                         actual_pick=(int(r.actual_pick) if pd.notna(r.actual_pick) else ""), war=round(float(r.war), 3),
                         source=r.universe_source, route=r.input_route, score=round(float(r.score), 6)))
    print(y, "pool", len(pool), "drafted", int(pool.was_drafted.sum()), "undrafted in top 60", int((pool.head(60).was_drafted == 0).sum()), flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"{C.REPO}/data/open_pool_board_2019_2025.csv", index=False)
print("wrote", f"{C.REPO}/data/open_pool_board_2019_2025.csv", len(out), "rows")
