"""Undrafted players: the outcome-selected check (old) and the clean pre-draft universe (H7).

Modes written to experiments/lab_2026-10-19/undrafted_eval.json
  strict, expanding   the earlier check: every row of the test files scored by the lab engine (boards
                      board_base_{strict,expanding}.csv). The undrafted rows there were collected knowing
                      who reached the NBA, so these numbers are indicative only.
  clean_strict        the H7 universe (DraftDB-Data/v4_build/universe_v422, built by universe_build.py with
                      no knowledge of later NBA participation: the league's final early-entry list, Division-I
                      seniors at >= 40% of team minutes who did not play college ball the next season, and
                      EuroLeague/EuroCup players aged 22), scored by the strict engine (train 2007-2018,
                      base genome, fold-learned weights from exp_base.json; nothing selected on 2019-2025).
  clean_strict_uniform  the same pool with every college player scored on a rebuilt row (the Torvik-only
                      information set), so real rows and rebuilt rows cannot be told apart: the sensitivity
                      of the clean numbers to the row source.
  consistency         the price of a rebuilt row: drafted college players scored in the pool on their real
                      v4.22 rows and again with rebuilt rows swapped in; the Spearman between the two scorings,
                      their mean rank in the pool each way, and how many sit in the first 60 each way.

Pool rule for clean_strict: rows with inputs and a known outcome. Universe rows whose name matches an NBA
player absent from every draft class (an undrafted NBA player the data never collected) have no outcome
and are excluded from the metrics, counted as reached_nba_no_outcome. The real draft is scored with every
undrafted player at pick 61. WAR uses the class's horizon k (common.horizon).
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, metrics as M
import engine as En
from scipy.stats import spearmanr

UNI = f"{C.DATA}/v4_build/universe_v422"
NAMES = C.names()
try:
    UNAMES = dict(pd.read_csv(f"{C.DATA}/identity/universe_names.csv")[["pid", "player_name"]].values)
except FileNotFoundError:
    UNAMES = {}
NAMES = {**UNAMES, **NAMES}
V = C.pick_value_curve()
TOP_N = 60
GEM_WAR = 2.0


def old_modes():
    out = {}
    for mode in ("strict", "expanding"):
        p = f"{C.OUT}/board_base_{mode}.csv"
        if not os.path.exists(p):
            continue
        b = pd.read_csv(p); out[mode] = {}
        for y in sorted(b.year.unique()):
            full = C.pool(y, drafted_only=False); bb = b[b.year == y]
            m = full.merge(bb[["pid", "score"]], on="pid")
            d = m[m.was_drafted == 1]; nd = len(d)
            teams = np.where(m.was_drafted == 1, -m.actual_pick, -61.0)
            top = m.sort_values("score", ascending=False).head(nd)
            m["model_rank_vs_drafted"] = [int((d.score > s).sum()) + 1 for s in m.score]
            gems = m[(m.was_drafted == 0) & (m.war >= GEM_WAR)].sort_values("war", ascending=False)
            out[mode][int(y)] = dict(n_full=len(m), n_drafted=nd, n_undrafted=int((m.was_drafted == 0).sum()),
                spearman_full_model=M.spearman(m.score, m.war), spearman_full_teams=M.spearman(teams, m.war),
                spearman_drafted_model=M.spearman(d.score, d.war), spearman_drafted_teams=M.spearman(-d.actual_pick, d.war),
                undrafted_in_model_top_n=int((top.was_drafted == 0).sum()), undrafted_war_in_model_top_n=float(top[top.was_drafted == 0].war.sum()),
                gems=[dict(pid=r.pid, name=NAMES.get(r.pid, "?"), war=round(float(r.war), 1), war_rank_full=int((m.war > r.war).sum()) + 1,
                           model_rank_vs_drafted=int(r.model_rank_vs_drafted), n_drafted=nd) for r in gems.itertuples()])
    return out


def strict_fit():
    g = dict(En.BASE); cols = En.families(("sc_", "tc_"))
    tr = En.TRAIN[(En.TRAIN.draft_year >= 2007) & (En.TRAIN.draft_year <= 2018)].reset_index(drop=True)
    kby = {int(c): 5 for c in tr.draft_year.unique()}
    pr, cq = En.fit_prior(tr, cols); Btr = En.build(tr, pr, cq, cols, g); fc = En.feature_cols(cols, g)
    ylab = En.label(tr, kby)
    wts = json.load(open(f"{C.OUT}/exp_base.json"))["weights"]
    wts = {k: tuple(v) for k, v in wts.items()}

    def score(te):
        te = te.copy()
        for c in cols:
            if c not in te.columns:
                te[c] = np.nan
        Bte = En.build(te, pr, cq, cols, g)
        P = En.member_preds(g, Btr, Bte, fc, ylab)
        return En.stack(P, Bte, g, wts)
    return score, set(tr.pid)


def swap_rebuilt(X, RB):
    """X with the rebuilt rows of RB in place of the real rows of the same pids; blocks the rebuild cannot reach go empty."""
    X2 = X.set_index("pid").copy()
    have = [c for c in RB.columns if c in En._INPUTS]
    X2.loc[RB.pid, have] = RB.set_index("pid")[have].values
    X2.loc[RB.pid, [c for c in En._INPUTS if c not in RB.columns]] = np.nan
    return X2.reset_index()


def clean_mode(score, train_pids, uniform=False):
    out, cons = {}, {}
    summ = json.load(open(f"{UNI}/universe_summary.json"))
    for y in C.TEST_YEARS:
        X = pd.read_csv(f"{UNI}/universe_{y}_inputs.csv", low_memory=False)
        O = pd.read_csv(f"{UNI}/universe_{y}_outcomes.csv")
        assert not (set(X.pid) & train_pids), "universe row in the training classes"
        X["draft_year"] = y
        rp = f"{UNI}/rebuilt_check_{y}.csv"
        RB = pd.read_csv(rp, low_memory=False) if os.path.exists(rp) else None
        if uniform and RB is not None:
            X = swap_rebuilt(X, RB)
        s = score(X)
        O = O.merge(pd.DataFrame(dict(pid=X.pid.values, score=s)), on="pid")
        if uniform:
            O.loc[O.pid.isin(RB.pid), "input_route"] = "torvik_rebuild"
        k = C.horizon(y)
        O["war"] = O[C.S[:k]].fillna(0).sum(axis=1).where(O.outcome_known == 1)
        pool = O[(O.has_inputs == 1) & (O.outcome_known == 1)].copy()
        pool["actual_pick"] = pool.actual_pick.fillna(61.0); pool["k"] = k
        pool = pool.sort_values("score", ascending=False).reset_index(drop=True)
        pool["model_rank"] = np.arange(1, len(pool) + 1)
        d = pool[pool.was_drafted == 1]
        em = M.evaluate(dict(zip(pool.pid, pool.score)), pool, V)
        et = M.evaluate(None, pool, V, pick_as_board=True)
        top = pool.head(TOP_N); utop = top[top.was_drafted == 0]
        gems = pool[(pool.was_drafted == 0) & (pool.war >= GEM_WAR)].sort_values("war", ascending=False)
        rec = lambda r: dict(pid=r.pid, name=NAMES.get(r.pid, "?"), war=round(float(r.war), 1), model_rank=int(r.model_rank), source=r.universe_source,
                             route=r.input_route, war_rank_full=int((pool.war > r.war).sum()) + 1, n_drafted_below=int((d.war < r.war).sum()))
        # the drafted players the model's undrafted picks displace: the drafted players ranked outside the model's top 60
        displaced = pool[(pool.was_drafted == 1) & (pool.model_rank > TOP_N)]
        sm = summ["years"][str(y)]
        out[int(y)] = dict(k=k, n_universe=len(O), n_scored=len(pool), n_drafted=int(pool.was_drafted.sum()), n_undrafted=int((pool.was_drafted == 0).sum()),
            n_no_inputs=int((O.has_inputs == 0).sum()), n_reached_nba_no_outcome=int(O.reached_nba_no_outcome.sum()),
            n_from_test_rows=int((pool.input_route == "test_row").sum()), n_rebuilt=int(pool.input_route.isin(["torvik_rebuild", "euro_rebuild"]).sum()),
            sources=sm["sources"], n_seniors_rule=sm["n_seniors_rule"], n_seniors_returned=sm["n_seniors_returned_Y1"], n_entrants=sm["n_entrants"], n_intl22=sm["n_intl22"],
            drafted_outside_rule=sm["drafted_outside_rule"], existing_test_rows_dropped=sm["existing_test_rows_dropped"],
            spearman_full_model=em["spearman"], spearman_full_teams=et["spearman"], vo_pick_full_model=em["vo_pick"], vo_pick_full_teams=et["vo_pick"],
            spearman_drafted_model=M.spearman(d.score, d.war), spearman_drafted_teams=M.spearman(-d.actual_pick, d.war),
            undrafted_in_model_top_n=int(len(utop)), undrafted_war_in_model_top_n=float(utop.war.sum()), undrafted_in_top_n_with_war=int((utop.war != 0).sum()),
            undrafted_in_top_n_list=[rec(r) for r in utop.itertuples()],
            displaced_drafted_war=float(displaced.war.sum()), displaced_drafted_n=int(len(displaced)),
            undrafted_war_total=float(pool[pool.was_drafted == 0].war.sum()), undrafted_with_war=int(((pool.was_drafted == 0) & (pool.war != 0)).sum()),
            gems_found=[rec(r) for r in gems[gems.model_rank <= TOP_N].itertuples()], gems_missed=[rec(r) for r in gems[gems.model_rank > TOP_N].itertuples()],
            drafted_top_n_hits=int((top.was_drafted == 1).sum()))
        # consistency: drafted college players scored in the pool on real rows and again with rebuilt rows swapped in
        if RB is not None and not uniform:
            RBd = RB[RB.was_drafted == 1]
            s2 = pd.Series(score(swap_rebuilt(X, RBd)), index=X.pid); s1 = pd.Series(s, index=X.pid)
            r1, r2 = s1.rank(ascending=False), s2.rank(ascending=False)
            cons[int(y)] = dict(n=len(RBd), pool=len(X), spearman_real_vs_rebuilt=float(spearmanr(s1[RBd.pid], s2[RBd.pid]).statistic),
                                mean_rank_real=float(r1[RBd.pid].mean()), mean_rank_rebuilt=float(r2[RBd.pid].mean()),
                                median_rank_real=float(r1[RBd.pid].median()), median_rank_rebuilt=float(r2[RBd.pid].median()),
                                top60_real=int((r1[RBd.pid] <= TOP_N).sum()), top60_rebuilt=int((r2[RBd.pid] <= TOP_N).sum()))
        print(f"{y}: pool {len(pool)} (drafted {out[int(y)]['n_drafted']}, undrafted {out[int(y)]['n_undrafted']}, k={k})  Sp model {em['spearman']:+.3f} teams {et['spearman']:+.3f}  "
              f"PV {em['vo_pick']:.1f}/{et['vo_pick']:.1f}  undrafted in top {TOP_N}: {len(utop)} (WAR {utop.war.sum():.1f}, {int((utop.war != 0).sum())} played)  gems found {len(out[int(y)]['gems_found'])} missed {len(out[int(y)]['gems_missed'])}"
              + (f"  consistency Sp {cons[int(y)]['spearman_real_vs_rebuilt']:.3f}, drafted in top 60 real {cons[int(y)]['top60_real']} vs rebuilt {cons[int(y)]['top60_rebuilt']}, mean rank {cons[int(y)]['mean_rank_real']:.0f} vs {cons[int(y)]['mean_rank_rebuilt']:.0f}" if int(y) in cons else ""), flush=True)
    return out, cons


def main():
    res = old_modes()
    score, train_pids = strict_fit()
    res["clean_strict"], res["consistency"] = clean_mode(score, train_pids)
    print("\nuniform rows (every college player on a rebuilt row):")
    res["clean_strict_uniform"], _ = clean_mode(score, train_pids, uniform=True)
    res["definitions"] = dict(top_n=TOP_N, gem_war=GEM_WAR, universe_rule=json.load(open(f"{UNI}/universe_summary.json"))["rule"])
    C.write_json(res, "undrafted_eval.json")
    cs = res["clean_strict"]
    print("\nclean universe, means over 7 classes: Sp model %.3f teams %.3f | PV %.1f / %.1f | undrafted in top 60: %.1f per class, WAR %.1f" % (
        np.mean([v["spearman_full_model"] for v in cs.values()]), np.mean([v["spearman_full_teams"] for v in cs.values()]),
        np.mean([v["vo_pick_full_model"] for v in cs.values()]), np.mean([v["vo_pick_full_teams"] for v in cs.values()]),
        np.mean([v["undrafted_in_model_top_n"] for v in cs.values()]), np.mean([v["undrafted_war_in_model_top_n"] for v in cs.values()])))


if __name__ == "__main__":
    main()
