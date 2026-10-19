"""Order metrics for a draft class.

A *board* is a dict pid -> score (higher = earlier pick). The *pool* is a DataFrame with
pid, actual_pick, war (k-season WAR). The pick-value curve v[j] is the expected 5-season WAR of
overall pick j, fitted on the 2000-2018 classes (common.pick_value_curve).

Metrics
  spearman        plain rank correlation between the order and the WAR order (the repo's headline).
  spearman_top    top-weighted rank correlation (Pinto da Costa & Soares 2005): a disagreement at
                  the top of either ranking costs more than one at the bottom.
  vo_linear       value of the order with linear slot weights n-j+1 (the repo's second metric).
  vo_pick         value of the order with the empirical pick-value curve as slot weights:
                  100 = perfect order, 0 = reverse, ~50 = random. This is the "pick-value-weighted"
                  metric: a swap at picks 2 and 12 costs far more than one at 40 and 45.
  war_above_slot  sum over slots of v[j] * (war at slot j) in WAR units, minus the same for a random
                  order: how many WAR the order placed where the expensive picks are.
  ndcg            NDCG with gains = max(WAR, 0), log2 discount.
  topN_hits       how many of the order's first N are among the true top N by WAR.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def _order(board, pool):
    d = pool[pool.pid.isin(board.keys())].copy()
    d["score"] = d.pid.map(board).astype(float)
    d = d.sort_values(["score", "actual_pick"], ascending=[False, True]).reset_index(drop=True)
    return d


def slot_weights(n, v):
    j = np.arange(1, n + 1)
    return np.array([v[min(int(x), len(v) - 1)] for x in j])


def spearman(a, b):
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def spearman_top(a, b):
    """Weighted rank correlation (Pinto da Costa & Soares 2005). a, b: scores (higher = better)."""
    n = len(a)
    R = pd.Series(-np.asarray(a)).rank(method="average").values
    Q = pd.Series(-np.asarray(b)).rank(method="average").values
    num = 6 * np.sum((R - Q) ** 2 * ((n - R + 1) + (n - Q + 1)))
    den = n ** 4 + n ** 3 - n ** 2 - n
    return float(1 - num / den)


def value_of_order(war_in_order, w):
    """100 * (V - V_worst) / (V_perfect - V_worst) for V = sum_j w_j * war_j."""
    war_in_order = np.asarray(war_in_order, float); w = np.asarray(w, float)[: len(war_in_order)]
    V = (w * war_in_order).sum()
    s = np.sort(war_in_order)
    Vp, Vw = (w * s[::-1]).sum(), (w * s).sum()
    return float(100 * (V - Vw) / (Vp - Vw)) if Vp > Vw else 50.0


def war_above_slot(war_in_order, v):
    """sum_j v_j * war_j  minus  mean(v) * sum(war): WAR-weighted placement relative to a random order,
    in WAR units (scaled by the pick-value curve). Positive = good players at the expensive slots."""
    war_in_order = np.asarray(war_in_order, float); n = len(war_in_order)
    w = slot_weights(n, v)
    return float((w * war_in_order).sum() - w.mean() * war_in_order.sum())


def ndcg(war_in_order):
    g = np.clip(np.asarray(war_in_order, float), 0, None)
    disc = 1 / np.log2(np.arange(2, len(g) + 2))
    dcg = (g * disc).sum(); idcg = (np.sort(g)[::-1] * disc).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def top_hits(d, N):
    top_true = set(d.sort_values(["war", "actual_pick"], ascending=[False, True]).pid.head(N))
    return int(len(top_true & set(d.pid.head(N))))


def evaluate(board, pool, v, pick_as_board=False):
    """All metrics for one board on one class. If pick_as_board, the board is the real draft order."""
    if pick_as_board:
        board = {p: -pk for p, pk in zip(pool.pid, pool.actual_pick)}
    d = _order(board, pool)
    n = len(d)
    if n < 10:
        return None
    war = d.war.values
    lin = np.arange(n, 0, -1)
    return dict(n=n, k=int(d.k.iloc[0]) if "k" in d else None,
                spearman=spearman(d.score, war),
                spearman_top=spearman_top(d.score.values, war),
                vo_linear=value_of_order(war, lin),
                vo_pick=value_of_order(war, slot_weights(n, v)),
                war_above_slot=war_above_slot(war, v),
                ndcg=ndcg(war),
                top5_hits=top_hits(d, 5), top10_hits=top_hits(d, 10), top20_hits=top_hits(d, 20),
                war_top10=float(war[:10].sum()))


def paired_bootstrap(board_a, board_b, pool, v, metric="spearman", B=2000, seed=0, pick_b=False):
    """Bootstrap over players (within class): CI for metric(a), metric(b) and their difference."""
    rng = np.random.default_rng(seed)
    pids = pool.pid.values
    if pick_b:
        board_b = {p: -pk for p, pk in zip(pool.pid, pool.actual_pick)}
    keep = [p for p in pids if p in board_a and p in board_b]
    base = pool[pool.pid.isin(keep)].reset_index(drop=True)
    out_a, out_b = [], []
    for _ in range(B):
        idx = rng.integers(0, len(base), len(base))
        s = base.iloc[idx].reset_index(drop=True)
        s = s.drop_duplicates("pid")        # ties in a resample would corrupt ordering; approximate
        if len(s) < 10:
            continue
        ea, eb = evaluate(board_a, s, v), evaluate(board_b, s, v)
        if ea and eb:
            out_a.append(ea[metric]); out_b.append(eb[metric])
    a, b = np.array(out_a), np.array(out_b)
    return dict(a_ci=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                b_ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                diff_ci=[float(np.percentile(a - b, 2.5)), float(np.percentile(a - b, 97.5))],
                p_a_gt_b=float((a > b).mean()))


def sequential_redraft(draft, board, pool):
    """Snake through the real draft in pick order; at each slot the team on the clock takes the
    model's best remaining player. Each player is taken once. Returns one row per slot."""
    war = dict(zip(pool.pid, pool.war))
    avail = {p for p in pool.pid if p in board}
    rows = []
    for r in draft.itertuples():
        if not avail:
            break
        best = max(avail, key=lambda p: (board[p], -war.get(p, 0)))
        avail.discard(best)
        rows.append(dict(year=r.year, pick=r.pick, round=r.round, team_id=r.team_id, abbr=r.abbr, team=r.team,
                         actual_pid=r.pid, actual_name=r.name, actual_war=war.get(r.pid, np.nan),
                         model_pid=best, model_war=war.get(best, np.nan)))
    return pd.DataFrame(rows)
