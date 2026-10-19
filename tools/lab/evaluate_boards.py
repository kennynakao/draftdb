"""Score every available board on the 2019-2025 classes under the old and the new metrics.

Boards: the real draft (teams), the current model strict and expanding, Colin's baseline, dated
multi-site mock consensus (and each publisher alone), the legacy consensus column, RSCI, raw college
production, and two small ridge regressions fit on 2007-2018 (with / without consensus).

Writes lab/out/boards_eval.json (+ per-board sequential redraft CSVs). No model is selected here;
this is reporting on sealed classes.
"""
import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, metrics as M
from scipy.stats import norm, spearmanr

V = C.pick_value_curve()
NAMES = C.names()
POOLS = {y: C.pool(y) for y in C.TEST_YEARS}
DRAFTS = {y: C.draft_history(y) for y in C.TEST_YEARS}


def gaussrank_label(df):
    w = df[C.S].fillna(0).values * np.array([0.85 ** i for i in range(5)])
    v = np.clip(w.sum(1), -40, 40)
    o = np.zeros(len(df))
    for y in df.draft_year.unique():
        m = (df.draft_year == y).values; r = pd.Series(v[m]).rank(pct=True).values
        o[m] = norm.ppf(np.clip((r * len(r) - 0.5) / len(r), 0.01, 0.99))
    return o


# ------------------------------------------------------------------ boards
def boards_for_year(y):
    B = {}
    pool = POOLS[y]
    b = C.board_current_strict(); b = b[b.year == y]
    B["model_strict"] = dict(zip(b.pid, b.score))
    e = C.board_current_expanding(); e = e[e.year == y]
    if len(e):
        B["model_expanding"] = dict(zip(e.pid, e.score))
    c = C.board_colin(); c = c[c.year == y]
    B["colin"] = dict(zip(c.pid, c.score))
    # dated multi-site consensus
    m = C.consensus_multisite(); m = m[m.year == y]
    if len(m):
        sites = sorted(m.site.unique())
        n_site = m.groupby("site")["rank"].max().to_dict()
        pids = set(pool.pid)
        rk = {}
        for p in pids:
            rs = []
            for s in sites:
                r = m[(m.site == s) & (m.pid == p)]["rank"]
                rs.append(float(r.iloc[0]) if len(r) else n_site[s] + 1)
            rk[p] = np.mean(rs)
        B["consensus_multisite"] = {p: -r for p, r in rk.items()}
        for s in sites:
            ms = m[m.site == s]
            B[f"site_{s}"] = {p: -float(r) for p, r in zip(ms.pid, ms["rank"]) if p in pids}
    # legacy consensus (public inputs) and dated vcons (v4.22 inputs)
    ti = C.test_inputs(y)
    col = pd.to_numeric(ti["cons_mock_consensus_rank"], errors="coerce")
    B["consensus_legacy"] = {p: -float(r) for p, r in zip(ti.pid, col) if pd.notna(r)}
    rs = pd.to_numeric(ti["hs_rsci_rank"], errors="coerce")
    B["rsci"] = {p: -float(r) for p, r in zip(ti.pid, rs) if pd.notna(r)}
    imp = pd.to_numeric(ti["col_impact"], errors="coerce")
    iimp = pd.to_numeric(ti["intl_impact"], errors="coerce")
    B["production"] = {p: float(a if pd.notna(a) else b) for p, a, b in zip(ti.pid, imp, iimp) if pd.notna(a) or pd.notna(b)}
    tv = C.test_inputs(y, "v4")
    if "vcons_mock_mean_rank" in tv.columns:
        vc = pd.to_numeric(tv["vcons_mock_mean_rank"], errors="coerce")
        B["consensus_dated"] = {p: -float(r) for p, r in zip(tv.pid, vc) if pd.notna(r)}
    return B


# ------------------------------------------------------------------ small ridge baselines
RIDGE_COLS = ["bio_age_at_draft", "col_impact", "intl_impact", "col_ts_pct", "col_usg_pct", "col_ast_pct",
              "col_stl_pct", "col_blk_pct", "bio_height_in", "col_fg3a_per40", "col_ft_pct", "col_minutes_total"]


def fit_ridge_baselines():
    from sklearn.linear_model import Ridge
    t = C.train(); t = t[(t.draft_year >= 2007) & (t.draft_year <= 2018)].reset_index(drop=True)
    y = gaussrank_label(t)

    def mat(d, cols, med=None):
        X = d[cols].apply(pd.to_numeric, errors="coerce")
        X["log_rsci"] = np.log(pd.to_numeric(d["hs_rsci_rank"], errors="coerce").fillna(150).clip(1, 150))
        X["log_cons"] = np.log(pd.to_numeric(d["cons_mock_consensus_rank"], errors="coerce").clip(1, 90))
        X["cons_missing"] = X.log_cons.isna().astype(float)
        X["col_missing"] = X.col_impact.isna().astype(float)
        if med is None:
            med = X.median()
        X = X.fillna(med)
        return X, med

    out = {}
    for name, cols in [("ridge_lite", [c for c in RIDGE_COLS]), ("ridge_lite_cons", [c for c in RIDGE_COLS])]:
        X, med = mat(t, cols)
        if name == "ridge_lite":
            X = X.drop(columns=["log_cons", "cons_missing"])
        mu, sd = X.mean(), X.std().replace(0, 1)
        r = Ridge(alpha=30.0).fit((X - mu) / sd, y)
        boards = {}
        for yy in C.TEST_YEARS:
            ti = C.test_inputs(yy)
            Xt, _ = mat(ti, cols, med)
            if name == "ridge_lite":
                Xt = Xt.drop(columns=["log_cons", "cons_missing"])
            p = r.predict((Xt - mu) / sd)
            boards[yy] = dict(zip(ti.pid, p))
        out[name] = boards
    return out


# ------------------------------------------------------------------ main
def main():
    res = dict(pick_value_curve=[None] + [round(float(x), 2) for x in V[1:]], classes={}, means={}, boards={})
    ridge = fit_ridge_baselines()
    per_board = collections.defaultdict(dict)
    all_boards = {}
    for y in C.TEST_YEARS:
        pool = POOLS[y]
        B = boards_for_year(y)
        for nm, bb in ridge.items():
            B[nm] = {p: s for p, s in bb[y].items() if p in set(pool.pid)}
        all_boards[y] = B
        res["classes"][y] = {"k": int(pool.k.iloc[0]), "n": len(pool), "teams": M.evaluate(None, pool, V, pick_as_board=True)}
        for name, b in B.items():
            e = M.evaluate(b, pool, V)
            if e:
                # teams on the same covered subset, for a fair head-to-head
                sub = pool[pool.pid.isin(b.keys())]
                e["teams_same_subset"] = M.evaluate(None, sub, V, pick_as_board=True)
                res["classes"][y][name] = e
                per_board[name][y] = e

    # means over classes (only classes a board covers)
    keys = ["spearman", "spearman_top", "vo_linear", "vo_pick", "war_above_slot", "ndcg", "top5_hits", "top10_hits", "top20_hits", "war_top10"]
    res["means"]["teams"] = {k: float(np.mean([res["classes"][y]["teams"][k] for y in C.TEST_YEARS])) for k in keys}
    res["means"]["teams"]["n_classes"] = 7
    for name, d in per_board.items():
        ys = sorted(d)
        res["means"][name] = {k: float(np.mean([d[y][k] for y in ys])) for k in keys}
        res["means"][name]["n_classes"] = len(ys)
        res["means"][name]["years"] = ys
        res["means"][name]["teams_same_subset"] = {k: float(np.mean([d[y]["teams_same_subset"][k] for y in ys])) for k in keys}
        res["means"][name]["wins_vs_teams_spearman"] = int(sum(d[y]["spearman"] > d[y]["teams_same_subset"]["spearman"] for y in ys))
        res["means"][name]["wins_vs_teams_vo_pick"] = int(sum(d[y]["vo_pick"] > d[y]["teams_same_subset"]["vo_pick"] for y in ys))

    # paired bootstrap vs teams for the main boards, pooled over classes (mean of per-class metrics)
    res["bootstrap"] = {}
    for name in ["model_strict", "model_expanding", "colin", "consensus_multisite", "ridge_lite_cons"]:
        for metric in ["spearman", "vo_pick"]:
            diffs = []
            rng = np.random.default_rng(1)
            per_class = []
            for y in C.TEST_YEARS:
                if name not in all_boards[y] or y not in per_board[name]:
                    continue
                per_class.append(M.paired_bootstrap(all_boards[y][name], None, POOLS[y], V, metric=metric, B=400, seed=y, pick_b=True))
            res["bootstrap"][f"{name}:{metric}"] = per_class

    # ---------------------------------------------------------------- sequential redraft
    res["sequential"] = {}
    for name in ["model_expanding", "model_strict", "colin", "consensus_multisite"]:
        frames = []
        for y in C.TEST_YEARS:
            if name not in all_boards[y]:
                continue
            sr = M.sequential_redraft(DRAFTS[y], all_boards[y][name], POOLS[y])
            sr["model_name"] = sr.model_pid.map(NAMES)
            sr["slot_value"] = [V[min(p, 60)] for p in sr.pick]
            frames.append(sr)
        sr = pd.concat(frames, ignore_index=True)
        sr.to_csv(f"{C.OUT}/sequential_{name}.csv", index=False)
        tm = sr.groupby(["team_id", "team", "abbr"]).agg(n_picks=("pick", "size"), actual_war=("actual_war", "sum"),
                                                        model_war=("model_war", "sum"), slot_value=("slot_value", "sum")).reset_index()
        tm["delta"] = tm.model_war - tm.actual_war
        tm["actual_surplus"] = tm.actual_war - tm.slot_value
        tm["model_surplus"] = tm.model_war - tm.slot_value
        wins = sr.assign(w=np.sign(sr.model_war.fillna(0) - sr.actual_war.fillna(0))).groupby("team_id").w.agg(
            model_better=lambda s: int((s > 0).sum()), team_better=lambda s: int((s < 0).sum()))
        tm = tm.merge(wins, on="team_id")
        res["sequential"][name] = dict(
            league=dict(actual_war=float(sr.actual_war.fillna(0).sum()), model_war=float(sr.model_war.fillna(0).sum()),
                        n_slots=len(sr), model_better=int((sr.model_war.fillna(0) > sr.actual_war.fillna(0) + 0.05).sum()),
                        team_better=int((sr.actual_war.fillna(0) > sr.model_war.fillna(0) + 0.05).sum())),
            teams=tm.sort_values("delta", ascending=False).to_dict("records"),
            by_year={int(y): dict(actual=float(g.actual_war.fillna(0).sum()), model=float(g.model_war.fillna(0).sum())) for y, g in sr.groupby("year")},
            first_round={int(y): dict(actual=float(g[g["round"] == 1].actual_war.fillna(0).sum()), model=float(g[g["round"] == 1].model_war.fillna(0).sum())) for y, g in sr.groupby("year")})
    # the old (aggregating) method for the Warriors, for contrast
    old = pd.read_csv(f"{C.REPO}/data/team_cards_audit.csv")
    res["old_method_team_totals"] = old.groupby("abbr").agg(actual=("actual_war", "sum"), model=("model_war", "sum"), n=("pick", "size")).reset_index().to_dict("records")

    # ---------------------------------------------------------------- incremental signal over consensus
    res["incremental"] = {}
    for mname in ["model_strict", "model_expanding", "colin"]:
        for cname in ["consensus_multisite", "consensus_legacy", "ridge_lite_cons"]:
            sweep = collections.defaultdict(list); partial = []
            for y in C.TEST_YEARS:
                B = all_boards[y]
                if mname not in B or cname not in B:
                    continue
                pool = POOLS[y]
                keep = pool[pool.pid.isin(B[mname]) & pool.pid.isin(B[cname])].reset_index(drop=True)
                if len(keep) < 20:
                    continue
                rm = pd.Series([B[mname][p] for p in keep.pid]).rank(pct=True).values
                rc = pd.Series([B[cname][p] for p in keep.pid]).rank(pct=True).values
                w = keep.war.values
                for a in np.round(np.arange(0, 1.01, 0.1), 2):
                    sweep[float(a)].append(M.spearman(a * rm + (1 - a) * rc, w))
                # partial Spearman of model given consensus
                rw = pd.Series(w).rank(pct=True).values
                def resid(a, b):
                    A = np.c_[np.ones(len(b)), b]; return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
                partial.append(dict(year=int(y), n=len(keep), partial_model_given_cons=M.spearman(resid(rm, rc), resid(rw, rc)),
                                    partial_cons_given_model=M.spearman(resid(rc, rm), resid(rw, rm)),
                                    model=M.spearman(rm, w), cons=M.spearman(rc, w)))
            if sweep:
                res["incremental"][f"{mname}|{cname}"] = dict(blend={str(a): float(np.mean(v)) for a, v in sweep.items()}, partial=partial)

    # ---------------------------------------------------------------- horizon stability (training classes, complete windows)
    t = C.train(); t = t[(t.draft_year >= 2007) & (t.draft_year <= 2018) & t.actual_pick.notna()]
    hz = collections.defaultdict(list)
    for y, g in t.groupby("draft_year"):
        w5 = g[C.S].fillna(0).sum(axis=1).values
        for k in range(1, 6):
            wk = g[C.S[:k]].fillna(0).sum(axis=1).values
            hz[f"war{k}_vs_war5"].append(M.spearman(wk, w5))
            hz[f"teams_at_k{k}"].append(M.spearman(-g.actual_pick.values, wk))
            hz[f"top10_overlap_k{k}"].append(int(len(set(np.argsort(-wk)[:10]) & set(np.argsort(-w5)[:10]))))
    res["horizon_train"] = {k: dict(mean=float(np.mean(v)), per_class=[float(x) for x in v]) for k, v in hz.items()}
    # 2019-2021 test classes: model vs teams at every horizon
    res["horizon_test"] = {}
    for y in [2019, 2020, 2021]:
        a = C.answers(y); a = a[(a.was_drafted == 1) & a.actual_pick.notna()]
        row = {}
        for k in range(1, 6):
            pk = C.pool(y, k=k)
            row[k] = {nm: M.evaluate(all_boards[y][nm], pk, V)["spearman"] for nm in ["model_strict", "colin", "consensus_multisite"] if nm in all_boards[y]}
            row[k]["teams"] = M.evaluate(None, pk, V, pick_as_board=True)["spearman"]
            row[k]["vo_pick_model_strict"] = M.evaluate(all_boards[y]["model_strict"], pk, V)["vo_pick"]
            row[k]["vo_pick_teams"] = M.evaluate(None, pk, V, pick_as_board=True)["vo_pick"]
        res["horizon_test"][y] = row

    # ---------------------------------------------------------------- alternate outcomes
    res["alt_outcomes"] = {}
    for alt in ["war_pos", "minutes"]:
        rows = {}
        for y in C.TEST_YEARS:
            pool = POOLS[y].copy(); pool["war"] = pool[alt]
            rows[y] = {nm: M.evaluate(all_boards[y][nm], pool, V)["spearman"] for nm in ["model_strict", "model_expanding", "colin", "consensus_multisite"] if nm in all_boards[y]}
            rows[y]["teams"] = M.evaluate(None, pool, V, pick_as_board=True)["spearman"]
        res["alt_outcomes"][alt] = rows

    # ---------------------------------------------------------------- undrafted gems (scope question)
    res["undrafted"] = {}
    for y in C.TEST_YEARS:
        full = C.pool(y, drafted_only=False)
        full["war_rank_full"] = full.war.rank(ascending=False, method="min").astype(int)
        tv = C.test_inputs(y, "v4")
        vc = dict(zip(tv.pid, pd.to_numeric(tv.get("vcons_mock_mean_rank"), errors="coerce")))
        und = full[(full.was_drafted == 0) & (full.war >= 2.0)].sort_values("war", ascending=False)
        res["undrafted"][y] = dict(n_undrafted_rows=int((full.was_drafted == 0).sum()), n_drafted=int((full.was_drafted == 1).sum()),
            gems=[dict(pid=r.pid, name=NAMES.get(r.pid, "?"), war=round(float(r.war), 1), war_rank_full=int(r.war_rank_full),
                       n_drafted_below=int((POOLS[y].war < r.war).sum()), consensus_dated=(None if pd.isna(vc.get(r.pid, np.nan)) else float(vc[r.pid])))
                  for r in und.itertuples()])

    C.write_json(res, "boards_eval.json")
    # ---------------------------------------------------------------- console summary
    print(f"{'board':22s} {'n_cls':>5s} {'spearman':>9s} {'teams':>7s} {'sp_top':>7s} {'vo_pick':>8s} {'teams':>7s} {'WAS':>7s} {'teams':>7s} {'top10':>6s} {'teams':>6s}")
    for name in ["teams"] + sorted(res["means"], key=lambda n: -res["means"][n]["spearman"]):
        m = res["means"][name]
        if name == "teams":
            print(f"{name:22s} {7:5d} {m['spearman']:9.3f} {'':7s} {m['spearman_top']:7.3f} {m['vo_pick']:8.1f} {'':7s} {m['war_above_slot']:7.1f} {'':7s} {m['top10_hits']:6.1f}")
            continue
        ts = m["teams_same_subset"]
        print(f"{name:22s} {m['n_classes']:5d} {m['spearman']:9.3f} {ts['spearman']:7.3f} {m['spearman_top']:7.3f} {m['vo_pick']:8.1f} {ts['vo_pick']:7.1f} {m['war_above_slot']:7.1f} {ts['war_above_slot']:7.1f} {m['top10_hits']:6.1f} {ts['top10_hits']:6.1f}")
    print("\nsequential redraft, league totals (same players, different slots):")
    for name, s in res["sequential"].items():
        print(f"  {name:20s} slots {s['league']['n_slots']}  model better at {s['league']['model_better']} slots, team better at {s['league']['team_better']}")
    print("\nWarriors, old method vs sequential:")
    for r in res["old_method_team_totals"]:
        if r["abbr"] == "GSW":
            print("  old (aggregating):", r)
    for name in res["sequential"]:
        for r in res["sequential"][name]["teams"]:
            if r["abbr"] == "GSW":
                print(f"  sequential {name}: actual {r['actual_war']:.1f} model {r['model_war']:.1f} slot_value {r['slot_value']:.1f}")
    print("\nhorizon (train 2007-18): ", {k: round(v["mean"], 3) for k, v in res["horizon_train"].items() if "vs_war5" in k})
    print("teams at k:", {k: round(v["mean"], 3) for k, v in res["horizon_train"].items() if "teams_at" in k})
    print("\nincremental (blend sweep, model_expanding | consensus_multisite):")
    for k, v in res["incremental"].items():
        if k.startswith("model_expanding|consensus_multisite"):
            print("  ", {a: round(x, 3) for a, x in v["blend"].items()})
            print("   partial:", [(p["year"], round(p["partial_model_given_cons"], 3), round(p["partial_cons_given_model"], 3)) for p in v["partial"]])


if __name__ == "__main__":
    main()
