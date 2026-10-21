"""A compact re-implementation of the DraftDB training pipeline for local experiments.

Mirrors the published champion as closely as the public code allows (honest.py / evolve3.py):
data v4.22 inputs restricted to the champion's families (the 169 base inputs + NBADraft.net grades
sc_ + Torvik context tc_), the international-line mask, sample-size shrinkage, the shoot+prod
engineered features, a discounted/clipped/Gaussian-ranked within-class label, ridge-on-top-columns,
quantile XGBoost and (optionally) TabICL, rank-averaged with weights learned on the walk-forward
folds. Guarantees carried over: no outcome or pick column is ever a feature; folds train only on
earlier classes with labels that use only seasons completed by that fold's draft night; train/test
pids are disjoint.

Differences from the box: TabICL runs on CPU with 8 estimators (the box used 32 on a GPU, bagged
over 3 permutations) and the residual-XGBoost trick is omitted (its stack weight is 0 anyway).

Every experiment reports two fold objectives: Spearman (the repo's) and the pick-value-weighted
value of the order (vo_pick), on drafted players only, plus top-5 hits (H6's third judge).

Gene `star` (H6, 2026-10-20): when present, an XGBoost classifier for "top-3 discounted-WAR outcome in
class" (class-balanced) is trained on the same features and its probability rank is blended into the
board for the consensus top 10 only: final = (1-star)*stack + star*rank(p_top3), then re-ranked.
"""
import os, sys, json, time, itertools, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, metrics as M
warnings.filterwarnings("ignore")

S = C.S
MIN = [f"y_s{i}_minutes" for i in range(1, 6)]
V = C.pick_value_curve()
FAMILIES_BASE = ("col_", "intl_", "hs_", "pre_", "cons_", "bio_")
SHRINK = ["col_ts_pct", "col_efg_pct", "col_fg2_pct", "col_fg3_pct", "col_ft_pct", "col_ast_pct", "col_stl_pct", "col_blk_pct",
          "col_orb_pct", "col_drb_pct", "col_tov_pct", "col_usg_pct", "col_ortg", "col_drtg", "col_impact", "col_pts36",
          "col_reb36", "col_ast36", "col_stl36", "col_blk36"]
LEVEL = {4: 1.45, 3: 1.20, 2: 0.95, 1: 0.60}
INTL_VOL = ["intl_pts36", "intl_reb36", "intl_ast36", "intl_stl36", "intl_blk36"]
BANNED = ("y_", "actual_pick", "actual_round", "pid", "draft_year", "was_drafted", "declared_only", "path", "outcome_status")

# ------------------------------------------------------------------ data
_INPUTS = json.load(open(f"{C.V4}/input_columns.json"))["inputs"]


def families(extra=("sc_", "tc_")):
    cols = [c for c in _INPUTS if c.startswith(FAMILIES_BASE) or c.startswith(tuple(extra))]
    for c in cols:
        assert not c.startswith(BANNED) and c not in BANNED, c
    return cols


def load_train():
    t = pd.read_csv(f"{C.V4}/train_2000_2018.csv", low_memory=False)
    idn = C.identity()[["pid", "actual_pick"]]
    t = t.merge(idn, on="pid", how="left")
    for c in S + MIN:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def load_tests():
    out = {}
    for y in C.TEST_YEARS:
        d = pd.read_csv(f"{C.V4}/tests/test_{y}_inputs.csv", low_memory=False)
        a = C.answers(y)[["pid", "actual_pick", "was_drafted"] + S + [c for c in MIN if c in C.answers(y).columns]]
        out[y] = d.drop(columns=[c for c in d.columns if c.startswith("y_") or c in ("actual_pick",)], errors="ignore").merge(a, on="pid", how="left", suffixes=("", "_ans"))
    return out


TRAIN = load_train()
TESTS = load_tests()


# ------------------------------------------------------------------ labels
HW = {"uniform": [1, 1, 1, 1, 1], "disc85": [.85 ** i for i in range(5)], "disc70": [.7 ** i for i in range(5)],
      "late": [0.6, 0.8, 1.0, 1.0, 1.0], "back": [0.5, 0.75, 1.0, 1.25, 1.5], "cum3": [1, 1, 1, 0, 0]}


MIN = [f"y_s{i}_minutes" for i in range(1, 6)]


def discounted(df, k_by_year, hw="disc85", clip=40, floor0=0, minw=0.0):
    """Discounted, clipped k-season WAR per row, using for class c only its first k_by_year[c] seasons.
    floor0: clip each season's WAR at 0 before discounting (negative WAR = played a lot, badly, often on a
    bad team). minw: add minw * minutes/1000 per season (a value label that credits playing time)."""
    X = df[S].fillna(0).values.astype(float)
    if floor0:
        X = np.clip(X, 0, None)
    if minw:
        Mn = df[[c for c in MIN]].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float) if all(c in df.columns for c in MIN) else np.zeros_like(X)
        X = X + minw * Mn / 1000.0
    w = np.array(HW[hw], float)
    v = np.zeros(len(df))
    for i, (c) in enumerate(df.draft_year.values):
        k = int(k_by_year.get(int(c), 5))
        v[i] = (X[i, :k] * w[:k]).sum()
    return np.clip(v, -clip, clip)


def label(df, k_by_year, hw="disc85", clip=40, kind="gaussrank", floor0=0, minw=0.0):
    """Within-class Gaussian-rank of the discounted WAR (see discounted())."""
    v = discounted(df, k_by_year, hw=hw, clip=clip, floor0=floor0, minw=minw)
    o = np.zeros(len(df))
    for y in df.draft_year.unique():
        m = (df.draft_year == y).values; r = pd.Series(v[m]).rank(pct=True).values
        o[m] = norm.ppf(np.clip((r * len(r) - 0.5) / len(r), 0.01, 0.99)) if kind == "gaussrank" else r
    return o


def top_label(df, k_by_year, top=3, **kw):
    """H6: 1 for the players whose discounted WAR ranks in the top `top` of their class, else 0.
    Same discounted WAR as label(), so the folds stay season-consistent."""
    v = discounted(df, k_by_year, **kw)
    o = np.zeros(len(df))
    for y in df.draft_year.unique():
        m = (df.draft_year == y).values
        o[m] = (pd.Series(-v[m]).rank(method="first").values <= top).astype(float)
    return o


# ------------------------------------------------------------------ features
def _num(d, c):
    return pd.to_numeric(d.get(c), errors="coerce")


def build(d, prior, cq, cols, g):
    d = d.copy()
    for c in cols:
        if c in d and d[c].dtype == object:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if g.get("intl_mask", 1):
        college = _num(d, "col_gp").notna()
        for c in [c for c in d.columns if c.startswith("intl_")]:
            d.loc[college, c] = np.nan
    mins = _num(d, "col_minutes_total").fillna(_num(d, "col_gp") * 20.0)
    r = (mins / (mins + g.get("M", 400.0))).clip(0, 1).fillna(0.0); d["sample_reliability"] = r
    for c in SHRINK:
        d["shr_" + c] = r * _num(d, c).fillna(prior[c]) + (1 - r) * prior[c]
    lv = _num(d, "intl_level"); d["intl_strength"] = lv.map(LEVEL)
    for c in INTL_VOL:
        d["adj_" + c] = _num(d, c) * d["intl_strength"]
    base = [c for c in cols if c.startswith(FAMILIES_BASE)]
    cov = d[base].apply(pd.to_numeric, errors="coerce").notna().mean(axis=1)
    d["feat_coverage"] = ((cov - cq[0]) / (cq[1] - cq[0])).clip(0, 1)
    # shoot + prod engineered features
    ft, p3, vol, age, proj = _num(d, "col_ft_pct"), _num(d, "col_fg3_pct"), _num(d, "col_fg3a_per40"), _num(d, "bio_age_at_draft"), _num(d, "col_shot_proj_3p")
    st = prior["_stats"]
    z = lambda s, k: (s - st[k][0]) / (st[k][1] if st[k][1] > 0 else 1)
    d["x_ft_vol"] = ft * vol; d["x_proj_vol"] = proj * vol; d["x_ft_gap"] = ft - p3
    d["x_shoot_score"] = z(ft, "ft") + z(vol, "vol") + z(proj, "proj")
    imp, usg, ts, blk, ht = _num(d, "col_impact"), _num(d, "col_usg_pct"), _num(d, "col_ts_pct"), _num(d, "col_blk36"), _num(d, "bio_height_in")
    d["x_prod_over_age"] = imp - st["impact_age_slope"] * (age - st["age"][0])
    d["x_usg_x_ts"] = usg * ts
    d["x_rim_prot"] = blk * ht
    d["x_young_prod"] = imp * (age < 20.0).astype(float)
    d["x_old_prod"] = imp * (age >= 22.0).astype(float)
    if g.get("cons_value", 0):      # novel: consensus mapped through the empirical pick-value curve
        cr = _num(d, "cons_mock_consensus_rank")
        d["x_cons_value"] = cr.apply(lambda r: V[int(min(max(r, 1), 60))] if pd.notna(r) else np.nan)
        d["x_cons_value_log"] = np.log1p(d["x_cons_value"])
    if g.get("combine_flags", 0):   # novel: combine-participation pattern (measured but no athletic drills = "promise" signal)
        meas = _num(d, "bio_combine_height_in").notna() | _num(d, "bio_combine_wingspan_in").notna()
        drill = _num(d, "bio_combine_vertical_max_in").notna() | _num(d, "bio_combine_lane_agility_s").notna() | _num(d, "bio_combine_sprint_s").notna()
        d["x_cmb_measured"] = meas.astype(float); d["x_cmb_measured_no_drills"] = (meas & ~drill).astype(float)
    return d


def fit_prior(tr, cols):
    pr = {c: float(pd.to_numeric(tr[c], errors="coerce").mean()) for c in SHRINK}
    st = {k: (float(_num(tr, c).mean()), float(_num(tr, c).std())) for k, c in [("ft", "col_ft_pct"), ("vol", "col_fg3a_per40"), ("proj", "col_shot_proj_3p"), ("age", "bio_age_at_draft")]}
    a, i = _num(tr, "bio_age_at_draft"), _num(tr, "col_impact"); ok = a.notna() & i.notna()
    st["impact_age_slope"] = float(np.polyfit(a[ok], i[ok], 1)[0]) if ok.sum() > 50 else 0.0
    pr["_stats"] = st
    base = [c for c in cols if c.startswith(FAMILIES_BASE)]
    cq = tr[base].apply(pd.to_numeric, errors="coerce").notna().mean(axis=1).quantile([0.05, 0.95]).values
    return pr, cq


def feature_cols(cols, g):
    c = list(cols) + ["sample_reliability"] + ["shr_" + x for x in SHRINK] + ["intl_strength"] + ["adj_" + x for x in INTL_VOL] + ["feat_coverage"]
    c += ["x_ft_vol", "x_proj_vol", "x_ft_gap", "x_shoot_score", "x_prod_over_age", "x_usg_x_ts", "x_rim_prot", "x_young_prod", "x_old_prod"]
    if g.get("cons_value", 0): c += ["x_cons_value", "x_cons_value_log"]
    if g.get("combine_flags", 0): c += ["x_cmb_measured", "x_cmb_measured_no_drills"]
    for x in c:
        assert not x.startswith(BANNED) and x not in BANNED, x
    return c


# ------------------------------------------------------------------ members
def xgb_importance_topk(X, y, k, seed=11):
    import xgboost as xgb
    m = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, n_jobs=8, random_state=seed).fit(X, y)
    imp = m.feature_importances_
    return [c for _, c in sorted(zip(imp, X.columns), reverse=True)[:k]]


def member(m, Xtr, ytr, Xte, g, wtr=None):
    if m in ("xgb", "xgbm"):     # xgbm: the same member with the squared-error objective, so both objectives can sit in one stack
        import xgboost as xgb
        mono = {"bio_age_at_draft": -1}; mono.update(g.get("mono", {}))
        kw = dict(objective="reg:squarederror", max_depth=g.get("xgb_depth", 3), learning_rate=g.get("xgb_lr", 0.02), n_estimators=g.get("xgb_n", 800), subsample=0.7,
                  colsample_bytree=0.5, min_child_weight=g.get("xgb_mcw", 8), reg_lambda=5.0, reg_alpha=1.0, n_jobs=8,
                  monotone_constraints={c: v for c, v in mono.items() if c in Xtr.columns})
        if m == "xgb" and g.get("risk", "q25") == "q25":
            kw["objective"] = "reg:quantileerror"; kw["quantile_alpha"] = 0.25
        off = int(g.get("seed_offset", 0))     # seed-shifted confirmation: every learner's seeds move together
        return np.mean([xgb.XGBRegressor(**kw, random_state=s + off).fit(Xtr, ytr, sample_weight=wtr).predict(Xte) for s in range(11, 11 + g.get("bag", 3))], axis=0)
    if m == "ridge":
        from sklearn.linear_model import Ridge
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
        A, B = ((Xtr - mu) / sd).fillna(0.0), ((Xte - mu) / sd).fillna(0.0)
        return Ridge(alpha=g.get("ridge_alpha", 300.0)).fit(A, ytr, sample_weight=wtr).predict(B)
    if m == "tabicl":
        from tabicl import TabICLRegressor
        import torch; torch.set_num_threads(8)
        # device from the environment (TABICL_DEVICE=cuda on the A100 job); icl_bag averages over
        # feature permutations / random states, as the box does after the permutation-lottery fix
        dev = os.environ.get("TABICL_DEVICE", "cpu")
        preds = []
        for b in range(int(g.get("icl_bag", 1))):
            r = TabICLRegressor(n_estimators=g.get("icl_n", 8), device=dev, random_state=g.get("seed", 42) + 100 * b + int(g.get("seed_offset", 0)), outlier_threshold=2.0)
            r.fit(Xtr.astype(float).values, ytr)
            preds.append(r.predict(Xte.astype(float).values))
        return np.mean(preds, axis=0)
    raise ValueError(m)


def star_member(Xtr, ytop, Xte, g):
    """H6 top-of-board specialist: an XGBoost classifier for "top-3 outcome in class" with class-balanced
    weights (positives and negatives carry equal total weight), bagged over seeds; returns P(top-3) on Xte."""
    import xgboost as xgb
    pos = ytop.sum(); neg = len(ytop) - pos
    w = np.where(ytop > 0, 0.5 / max(pos, 1), 0.5 / max(neg, 1)) * len(ytop)
    kw = dict(objective="binary:logistic", max_depth=g.get("star_depth", 3), learning_rate=0.03, n_estimators=g.get("star_n", 400),
              subsample=0.7, colsample_bytree=0.5, min_child_weight=4, reg_lambda=5.0, reg_alpha=1.0, n_jobs=8,
              monotone_constraints={"bio_age_at_draft": -1} if "bio_age_at_draft" in Xtr.columns else {})
    return np.mean([xgb.XGBClassifier(**kw, random_state=s).fit(Xtr, ytop, sample_weight=w).predict_proba(Xte)[:, 1]
                    for s in range(31, 31 + g.get("bag", 3))], axis=0)


def sample_weights(ylab, g, years=None):
    """Optional training-row weights. topw: weight by the pick-value of the label's rank. recency: half-life in years of
    an exponential decay by class age (the class before the fold's draft weighs 1, one half-life earlier 0.5): a soft
    version of the training window, aimed at the era shift diagnosed in September."""
    w = np.ones(len(ylab))
    if g.get("topw", 0):
        r = pd.Series(-ylab).rank(method="first").values      # 1 = best label
        n = len(r); slot = np.clip(np.round(r / n * 60), 1, 60).astype(int)
        pv = np.array([V[s] for s in slot]); pv = pv / pv.mean()
        w = w * ((1 - g["topw"]) + g["topw"] * pv)
    if g.get("recency", 0) and years is not None:
        age = years.max() - years          # 0 for the newest training class
        w = w * 0.5 ** (age / float(g["recency"]))
    if not g.get("topw", 0) and not g.get("recency", 0):
        return None
    return w / w.mean()


def member_preds(g, Btr, Bte, cols, ylab, ytop=None):
    out = {}
    wtr = sample_weights(ylab, g, _num(Btr, "draft_year").values)
    topk = xgb_importance_topk(Btr[cols], ylab, g.get("topk", 100), seed=11 + int(g.get("seed_offset", 0)))
    for m in g["members"]:
        cc = topk if m in ("ridge", "tabicl") else cols
        p = member(m, Btr[cc], ylab, Bte[cc], g, wtr)
        out[m] = pd.Series(p).rank(pct=True).values
    if ytop is not None:     # H6: the top-3 classifier's probability as a class rank, blended in stack()
        out["star"] = pd.Series(star_member(Btr[cols], ytop, Bte[cols], g)).rank(pct=True).values
    if g.get("consres", 0):   # consensus prior + learned residual: members were trained on the residual label
        pass
    return out


def stack(P, Bte, g, wts):
    cov = _num(Bte, "feat_coverage").fillna(0).values
    thin = cov < np.quantile(cov, g.get("thin_q", 0.5))
    r = np.zeros(len(cov))
    for i, m in enumerate(g["members"]):
        r += np.where(thin, wts["thin"][i], wts["rich"][i]) * P[m]
    r = pd.Series(r).rank(pct=True).values
    k = int(g.get("anchor_k", 0))
    if k:   # the consensus top-k can fall no lower than slot k+2 on the model's board
        cons = _num(Bte, "cons_mock_consensus_rank").values
        n = len(r); floor = 1 - (k + 2) / n
        r = np.where((cons <= k) & np.isfinite(cons), np.maximum(r, floor + 1e-6 * (k - cons)), r)
        r = pd.Series(r).rank(pct=True).values
    star = float(g.get("star", 0))
    if star and "star" in P:   # H6: inside the consensus top 10 only, blend the top-3 probability rank into the board
        cons = _num(Bte, "cons_mock_consensus_rank").values
        top = (cons <= g.get("star_top", 10)) & np.isfinite(cons)
        r = np.where(top, (1 - star) * r + star * P["star"], r)
        r = pd.Series(r).rank(pct=True).values
    return r


def _prior_rank(d):
    """Rank-percentile of the mock consensus (higher = better), 0.5 where missing."""
    c = _num(d, "cons_mock_consensus_rank")
    return pd.Series(-c).rank(pct=True).fillna(0.5).values


# ------------------------------------------------------------------ folds and blind
def fold_eval(pred, d, k):
    dd = d[d.actual_pick.notna()].reset_index(drop=True)
    pool = pd.DataFrame(dict(pid=dd.pid, actual_pick=dd.actual_pick, war=dd[S[:k]].fillna(0).sum(axis=1).values, k=k))
    board = dict(zip(dd.pid, pred[d.actual_pick.notna().values]))
    e = M.evaluate(board, pool, V)
    return e


def walk_forward(g, cols, fold_years=range(2012, 2019), verbose=True):
    win = g.get("win", 2007)
    W = TRAIN[(TRAIN.draft_year >= win) & (TRAIN.draft_year <= 2018)].reset_index(drop=True)
    F = {}
    for y in fold_years:
        trm = (W.draft_year < y).values; tem = (W.draft_year == y).values
        assert (W.draft_year.values[trm] < y).all() and not (set(W.pid[trm]) & set(W.pid[tem]))
        kby = {int(c): int(min(5, y - c)) for c in W.draft_year.unique()}        # season-consistent labels
        tr, te = W[trm], W[tem]
        pr, cq = fit_prior(tr, cols)
        Btr, Bte = build(tr, pr, cq, cols, g), build(te, pr, cq, cols, g)
        fc = feature_cols(cols, g)
        ylab = label(tr, kby, hw=g.get("hw", "disc85"), clip=g.get("clip", 40), floor0=g.get("floor0", 0), minw=g.get("minw", 0.0))
        ytop = top_label(tr, kby, top=g.get("star_k", 3), hw=g.get("hw", "disc85"), clip=g.get("clip", 40), floor0=g.get("floor0", 0), minw=g.get("minw", 0.0)) if "star" in g else None
        if g.get("consres", 0):
            ylab = ylab - norm.ppf(np.clip(_prior_rank(tr), 0.01, 0.99)) * g["consres"]
        t0 = time.time()
        P = member_preds(g, Btr, Bte, fc, ylab, ytop)
        if g.get("consres", 0):
            prior_te = norm.ppf(np.clip(_prior_rank(te), 0.01, 0.99))
            for m in P:
                raw = P[m]  # already a rank; combine on rank scale
                P[m] = pd.Series(norm.ppf(np.clip(raw, 0.01, 0.99)) + prior_te * g["consres"]).rank(pct=True).values
        F[y] = (P, Bte, te.reset_index(drop=True))
        if verbose:
            print(f"  fold {y}: train {trm.sum()} test {tem.sum()} ({time.time()-t0:.0f}s)", flush=True)
    return F


def learn_weights(g, F, objective="spearman"):
    n = len(g["members"]); eq = tuple([1 / n] * n)
    step = g.get("wgrid", 0.25)
    grid = [w for w in itertools.product(*[np.arange(0, 1.0 + step / 2, step)] * n) if abs(sum(w) - 1) < 1e-6]
    if n == 1:
        return {"rich": eq, "thin": eq}
    best = None
    for wr in grid:
        for wt in grid:
            wts = {"rich": wr, "thin": wt}
            sc = np.mean([np.mean([fold_eval(stack(F[y][0], F[y][1], g, wts), F[y][2], k)[objective] for k in range(1, 6)]) for y in F])
            if best is None or sc > best[0]:
                best = (sc, wts)
    return best[1]


def fold_scores(g, F, wts):
    out = {}
    for y in F:
        s = stack(F[y][0], F[y][1], g, wts)
        per_k = [fold_eval(s, F[y][2], k) for k in range(1, 6)]
        out[y] = dict(spearman_k5=per_k[4]["spearman"], spearman_avg=float(np.mean([e["spearman"] for e in per_k])),
                      vo_pick_k5=per_k[4]["vo_pick"], vo_pick_avg=float(np.mean([e["vo_pick"] for e in per_k])),
                      top5_hits_k5=per_k[4]["top5_hits"], top5_hits_avg=float(np.mean([e["top5_hits"] for e in per_k])),
                      top10_hits_k5=per_k[4]["top10_hits"],
                      n=per_k[4]["n"], members={m: fold_eval(F[y][0][m], F[y][2], 5)["spearman"] for m in F[y][0]})
    return out


def blind(g, cols, wts, mode="strict", verbose=True):
    """mode strict: train 2007-2018 (full labels), score 2019-2025. mode expanding: for each test year Y,
    train on 2007..Y-1 with labels using only seasons completed before draft Y."""
    rows = {}; boards = {}
    years = C.TEST_YEARS if mode == "strict" else C.TEST_YEARS[1:]
    win = g.get("win", 2007)
    base_tr = TRAIN[(TRAIN.draft_year >= win) & (TRAIN.draft_year <= 2018)]
    if mode == "strict":
        tr = base_tr.reset_index(drop=True); kby = {int(c): 5 for c in tr.draft_year.unique()}
        pr, cq = fit_prior(tr, cols); Btr = build(tr, pr, cq, cols, g); fc = feature_cols(cols, g)
        ylab = label(tr, kby, hw=g.get("hw", "disc85"), clip=g.get("clip", 40), floor0=g.get("floor0", 0), minw=g.get("minw", 0.0))
        if g.get("consres", 0): ylab = ylab - norm.ppf(np.clip(_prior_rank(tr), 0.01, 0.99)) * g["consres"]
        ytop = top_label(tr, kby, top=g.get("star_k", 3), hw=g.get("hw", "disc85"), clip=g.get("clip", 40), floor0=g.get("floor0", 0), minw=g.get("minw", 0.0)) if "star" in g else None
        fitted = (Btr, fc, ylab, ytop)
    for Y in years:
        te = TESTS[Y]
        if mode == "expanding":
            parts = [base_tr] + [TESTS[c] for c in C.TEST_YEARS if c < Y]
            tr = pd.concat(parts, ignore_index=True)
            kby = {int(c): int(min(5, Y - c)) for c in tr.draft_year.unique()}
            pr, cq = fit_prior(tr, cols); Btr = build(tr, pr, cq, cols, g); fc = feature_cols(cols, g)
            ylab = label(tr, kby, hw=g.get("hw", "disc85"), clip=g.get("clip", 40), floor0=g.get("floor0", 0), minw=g.get("minw", 0.0))
            if g.get("consres", 0): ylab = ylab - norm.ppf(np.clip(_prior_rank(tr), 0.01, 0.99)) * g["consres"]
            ytop = top_label(tr, kby, top=g.get("star_k", 3), hw=g.get("hw", "disc85"), clip=g.get("clip", 40), floor0=g.get("floor0", 0), minw=g.get("minw", 0.0)) if "star" in g else None
        else:
            Btr, fc, ylab, ytop = fitted
        assert not (set(tr.pid) & set(te.pid))
        Bte = build(te, pr, cq, cols, g)
        t0 = time.time()
        P = member_preds(g, Btr, Bte, fc, ylab, ytop)
        if g.get("consres", 0):
            prior_te = norm.ppf(np.clip(_prior_rank(te), 0.01, 0.99))
            for m in P: P[m] = pd.Series(norm.ppf(np.clip(P[m], 0.01, 0.99)) + prior_te * g["consres"]).rank(pct=True).values
        s = stack(P, Bte, g, wts)
        k = C.horizon(Y)
        pool = C.pool(Y)
        board = dict(zip(te.pid, s))
        e = M.evaluate(board, pool, V); et = M.evaluate(None, pool, V, pick_as_board=True)
        rows[Y] = dict(k=k, n=e["n"], model=e, teams=et, members={m: M.evaluate(dict(zip(te.pid, P[m])), pool, V)["spearman"] for m in P},
                       train_rows=len(tr))
        boards[Y] = pd.DataFrame(dict(pid=te.pid.values, year=Y, score=s, was_drafted=te.was_drafted.values, actual_pick=te.actual_pick.values))
        if verbose:
            print(f"  {mode} {Y}: k={k} n={e['n']} spearman {e['spearman']:+.3f} (teams {et['spearman']:+.3f}) vo_pick {e['vo_pick']:.1f} (teams {et['vo_pick']:.1f}) ({time.time()-t0:.0f}s)", flush=True)
    return rows, pd.concat(boards.values(), ignore_index=True)


def run(name, g, do_blind=True, weight_objective="spearman", extra=("sc_", "tc_")):
    cols = families(tuple(g.get("extra", list(extra))))
    t0 = time.time()
    print(f"== {name}: {g}  ({len(cols)} input cols)", flush=True)
    F = walk_forward(g, cols)
    wts = learn_weights(g, F, g.get("wobj", weight_objective))
    fs = fold_scores(g, F, wts)
    summary = dict(name=name, genome=g, weights=wts, folds={int(y): v for y, v in fs.items()},
                   wf_spearman=float(np.mean([v["spearman_avg"] for v in fs.values()])),
                   wf_spearman_k5=float(np.mean([v["spearman_k5"] for v in fs.values()])),
                   wf_vo_pick=float(np.mean([v["vo_pick_avg"] for v in fs.values()])),
                   wf_vo_pick_k5=float(np.mean([v["vo_pick_k5"] for v in fs.values()])),
                   wf_top5_hits=float(np.mean([v["top5_hits_avg"] for v in fs.values()])),
                   wf_top5_hits_k5=float(np.mean([v["top5_hits_k5"] for v in fs.values()])))
    print(f"   folds: spearman {summary['wf_spearman']:.4f} (k5 {summary['wf_spearman_k5']:.4f})  vo_pick {summary['wf_vo_pick']:.2f} (k5 {summary['wf_vo_pick_k5']:.2f})  top5 {summary['wf_top5_hits']:.2f} (k5 {summary['wf_top5_hits_k5']:.2f})  wts {wts}", flush=True)
    if do_blind:
        for mode in ("strict", "expanding"):
            rows, boards = blind(g, cols, wts, mode)
            summary[mode] = {int(y): v for y, v in rows.items()}
            summary[f"{mode}_mean_spearman"] = float(np.mean([v["model"]["spearman"] for v in rows.values()]))
            summary[f"{mode}_mean_vo_pick"] = float(np.mean([v["model"]["vo_pick"] for v in rows.values()]))
            summary[f"{mode}_teams_spearman"] = float(np.mean([v["teams"]["spearman"] for v in rows.values()]))
            summary[f"{mode}_teams_vo_pick"] = float(np.mean([v["teams"]["vo_pick"] for v in rows.values()]))
            summary[f"{mode}_mean_top5_hits"] = float(np.mean([v["model"]["top5_hits"] for v in rows.values()]))
            summary[f"{mode}_teams_top5_hits"] = float(np.mean([v["teams"]["top5_hits"] for v in rows.values()]))
            boards.to_csv(f"{C.OUT}/board_{name}_{mode}.csv", index=False)
            print(f"   {mode}: spearman {summary[f'{mode}_mean_spearman']:.3f} (teams {summary[f'{mode}_teams_spearman']:.3f}) vo_pick {summary[f'{mode}_mean_vo_pick']:.1f} (teams {summary[f'{mode}_teams_vo_pick']:.1f})", flush=True)
    summary["secs"] = round(time.time() - t0)
    C.write_json(summary, f"exp_{name}.json")
    with open(f"{C.OUT}/experiments.jsonl", "a") as f:
        f.write(json.dumps({k: v for k, v in summary.items() if k not in ("folds", "strict", "expanding")}, default=float) + "\n")
    return summary


BASE = dict(members=["xgb", "ridge"], hw="disc85", clip=40, M=400.0, intl_mask=1, risk="q25", bag=3, ridge_alpha=300.0, topk=100, win=2007)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "base"
    g = dict(BASE)
    if len(sys.argv) > 2:
        g.update(json.loads(sys.argv[2]))
    run(name, g)
