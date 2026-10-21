"""Draftability prior on the open pool (one blind look, report only).

H7 showed the strict board ties the real draft on the clean pre-draft universe and spends 35 of 60 picks per
class on undrafted seniors: the model ranks talent and has no notion of who is draftable. This adds the
simplest possible notion: a logistic P(drafted | age, consensus presence and rank, RSCI presence and rank,
college production, height) fit on the 2007-2018 training rows (drafted players vs the undrafted players who
later reached the NBA; the only undrafted rows the training data has), and blends its rank with the model's
rank on the open pool: final = (1-lam) * rank(model) + lam * rank(P(drafted)).

Nothing here is selected on folds (the universe exists only for 2019-2025), so every number below is a single
blind look and is reported as such. Writes experiments/lab_2026-10-19/draftability_prior.json.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, metrics as M
import engine as En
import evaluate_undrafted as U
from sklearn.linear_model import LogisticRegression

V = C.pick_value_curve()
FEATS = ["bio_age_at_draft", "cons_mock_consensus_rank", "hs_rsci_rank", "col_impact", "bio_height_in", "col_minutes_total", "intl_impact"]


def design(d):
    X = pd.DataFrame(index=d.index)
    age = pd.to_numeric(d["bio_age_at_draft"], errors="coerce"); X["age"] = age.fillna(age.median() if age.notna().any() else 21.0)
    cons = pd.to_numeric(d["cons_mock_consensus_rank"], errors="coerce")
    X["cons_present"] = cons.notna().astype(float); X["log_cons"] = np.log(cons.clip(1, 90)).fillna(np.log(90))
    rs = pd.to_numeric(d["hs_rsci_rank"], errors="coerce")
    X["rsci_present"] = rs.notna().astype(float); X["log_rsci"] = np.log(rs.clip(1, 150)).fillna(np.log(150))
    imp = pd.to_numeric(d["col_impact"], errors="coerce").fillna(pd.to_numeric(d["intl_impact"], errors="coerce"))
    X["impact"] = imp.fillna(0.0); X["impact_missing"] = imp.isna().astype(float)
    ht = pd.to_numeric(d["bio_height_in"], errors="coerce"); X["height"] = ht.fillna(78.0)
    mins = pd.to_numeric(d["col_minutes_total"], errors="coerce"); X["minutes"] = mins.fillna(0.0) / 1000.0
    return X


tr = En.TRAIN[(En.TRAIN.draft_year >= 2007) & (En.TRAIN.draft_year <= 2018)].reset_index(drop=True)
ytr = tr.actual_pick.notna().astype(int).values
Xtr = design(tr); mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
clf = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
print("P(drafted) fit on", len(tr), "training rows (drafted", int(ytr.sum()), "); coefficients:",
      {k: round(float(v), 2) for k, v in zip(Xtr.columns, clf.coef_[0])}, flush=True)

score, train_pids = U.strict_fit()
out = {}
for y in C.TEST_YEARS:
    X = pd.read_csv(f"{U.UNI}/universe_{y}_inputs.csv", low_memory=False); X["draft_year"] = y
    O = pd.read_csv(f"{U.UNI}/universe_{y}_outcomes.csv")
    O = O.merge(pd.DataFrame(dict(pid=X.pid.values, score=score(X))), on="pid")
    for c in FEATS:
        if c not in X.columns:
            X[c] = np.nan
    pd_ = clf.predict_proba((design(X) - mu) / sd)[:, 1]
    O = O.merge(pd.DataFrame(dict(pid=X.pid.values, p_drafted=pd_)), on="pid")
    k = C.horizon(y)
    O["war"] = O[C.S[:k]].fillna(0).sum(axis=1).where(O.outcome_known == 1)
    pool = O[(O.has_inputs == 1) & (O.outcome_known == 1)].copy()
    pool["actual_pick"] = pool.actual_pick.fillna(61.0); pool["k"] = k
    rm = pool.score.rank(pct=True).values; rp = pool.p_drafted.rank(pct=True).values
    row = {"n_pool": len(pool), "n_drafted": int(pool.was_drafted.sum()),
           "auc_p_drafted": float(((rp[pool.was_drafted.values == 1][:, None] > rp[pool.was_drafted.values == 0][None, :]).mean())),
           "teams": dict(spearman=M.evaluate(None, pool, V, pick_as_board=True)["spearman"], vo_pick=M.evaluate(None, pool, V, pick_as_board=True)["vo_pick"])}
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = (1 - lam) * rm + lam * rp
        e = M.evaluate(dict(zip(pool.pid, s)), pool, V)
        top = pool.assign(s=s).sort_values("s", ascending=False).head(60)
        row[f"lam_{lam}"] = dict(spearman=e["spearman"], vo_pick=e["vo_pick"], undrafted_in_top60=int((top.was_drafted == 0).sum()),
                                 undrafted_war_in_top60=float(top[top.was_drafted == 0].war.sum()), drafted_hits_top60=int((top.was_drafted == 1).sum()))
    out[int(y)] = row
    print(y, "pool", len(pool), "AUC(P drafted) %.3f" % row["auc_p_drafted"], "| teams Sp %.3f PV %.1f" % (row["teams"]["spearman"], row["teams"]["vo_pick"]),
          "| " + " ".join(f"lam{l}: Sp {row[f'lam_{l}']['spearman']:.3f} PV {row[f'lam_{l}']['vo_pick']:.1f} undrafted-in-60 {row[f'lam_{l}']['undrafted_in_top60']}" for l in [0.0, 0.5, 1.0]), flush=True)
means = {lam: {k: float(np.mean([out[y][f"lam_{lam}"][k] for y in out])) for k in ["spearman", "vo_pick", "undrafted_in_top60", "undrafted_war_in_top60"]} for lam in [0.0, 0.25, 0.5, 0.75, 1.0]}
means["teams"] = {k: float(np.mean([out[y]["teams"][k] for y in out])) for k in ["spearman", "vo_pick"]}
print("\nmeans over 7 classes:"); [print(" ", k, {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in means.items()]
C.write_json(dict(per_class=out, means=means, note="one blind look on the open pool; no fold selection possible (universe exists for 2019-2025 only)"), "draftability_prior.json")
