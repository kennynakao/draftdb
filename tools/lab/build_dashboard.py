"""Assemble experiments/lab_2026-10-19/*.json into a single static dashboard: REPO/lab.html (no build step, no library)."""
import os, sys, json, glob, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = json.load(open(f"{C.OUT}/boards_eval.json"))
U = json.load(open(f"{C.OUT}/undrafted_eval.json")) if os.path.exists(f"{C.OUT}/undrafted_eval.json") else {}
EXP = []
GPU_DIRS = ("gpu", "gpu_h", "gpu_s", "gpu_t", "gpu_u")       # the A100 runs of the TabICL recipe (32 estimators x 3 permutations)
CPU_DIRS = ("cpu_t",)                        # CPU-recipe replications of a GPU queue (tools/lab/run_queue_cpu.py), names prefixed c_
for p in sorted(glob.glob(f"{C.OUT}/exp_*.json")) + [q for d in CPU_DIRS for q in sorted(glob.glob(f"{C.OUT}/{d}/exp_*.json"))] + [q for d in GPU_DIRS for q in sorted(glob.glob(f"{C.OUT}/{d}/exp_*.json"))]:
    e = json.load(open(p))
    if any(f"/{d}/" in p for d in GPU_DIRS):
        e["name"] = e["name"] + " [A100]"
    EXP.append(e)
BASE_G = next((e["genome"] for e in EXP if e["name"] == "base"), {})

def r1(x): return None if x is None else round(float(x), 1)
def r3(x): return None if x is None else round(float(x), 3)

LABEL = {"teams": "NBA teams (real draft)", "model_expanding": "DraftDB model, expanding", "model_strict": "DraftDB model, strict",
         "colin": "Colin baseline (strict)", "consensus_multisite": "Mock consensus, 2-6 sites", "consensus_legacy": "Mock consensus (legacy column)",
         "consensus_dated": "Mock consensus (dated, v4)", "ridge_lite_cons": "14-feature ridge + consensus", "ridge_lite": "12-feature ridge, no consensus",
         "production": "College/intl impact only", "rsci": "RSCI recruiting rank", "site_tankathon": "Tankathon mock", "site_cbs": "CBS mock",
         "site_espn_ba": "ESPN best available", "site_nbadraft": "NBADraft.net mock", "site_sportingnews": "Sporting News mock", "site_draftroom": "NBA Draft Room mock"}

# ---------------------------------------------------------------- assemble
D = {}
means = R["means"]
D["tiles"] = {k: dict(spearman=r3(means[k]["spearman"]), vo_pick=r1(means[k]["vo_pick"]), was=r1(means[k]["war_above_slot"]),
                      top10=r1(means[k]["top10_hits"]), sp_top=r3(means[k]["spearman_top"]),
                      teams_vo=r1(means[k].get("teams_same_subset", means["teams"])["vo_pick"]),
                      teams_sp=r3(means[k].get("teams_same_subset", means["teams"])["spearman"]))
              for k in ["teams", "model_expanding", "model_strict", "colin", "ridge_lite_cons", "consensus_multisite"]}
D["per_class"] = []
for y, c in R["classes"].items():
    row = dict(year=int(y), k=c["k"], n=c["n"])
    for b in ["teams", "model_strict", "model_expanding", "colin", "ridge_lite_cons", "consensus_multisite"]:
        if b in c:
            row[b] = dict(spearman=r3(c[b]["spearman"]), vo_pick=r1(c[b]["vo_pick"]), was=r1(c[b]["war_above_slot"]), top10=c[b]["top10_hits"])
    D["per_class"].append(row)
D["baselines"] = sorted([dict(name=k, label=LABEL.get(k, k), n=means[k]["n_classes"], spearman=r3(means[k]["spearman"]),
                               teams_sp=r3(means[k]["teams_same_subset"]["spearman"]) if "teams_same_subset" in means[k] else r3(means["teams"]["spearman"]),
                               vo_pick=r1(means[k]["vo_pick"]), teams_vo=r1(means[k]["teams_same_subset"]["vo_pick"]) if "teams_same_subset" in means[k] else r1(means["teams"]["vo_pick"]),
                               wins_sp=means[k].get("wins_vs_teams_spearman"), wins_vo=means[k].get("wins_vs_teams_vo_pick"))
                         for k in means if k != "teams"], key=lambda r: -r["spearman"])
D["curve"] = R["pick_value_curve"][1:61]
# sequential redraft
sr = pd.read_csv(f"{C.OUT}/sequential_model_expanding.csv")
sr["bin"] = pd.cut(sr["pick"], [0, 5, 10, 20, 30, 45, 60], labels=["1-5", "6-10", "11-20", "21-30", "31-45", "46-60"])
g = sr.groupby("bin", observed=True).agg(n=("pick", "size"), actual=("actual_war", "sum"), model=("model_war", "sum")).reset_index()
D["seq_bins"] = [dict(bin=str(r.bin), n=int(r.n), actual=r1(r.actual), model=r1(r.model)) for r in g.itertuples()]
D["seq_league"] = R["sequential"]["model_expanding"]["league"]
D["seq_teams"] = [dict(team=t["team"], abbr=t["abbr"], n=int(t["n_picks"]), actual=r1(t["actual_war"]), model=r1(t["model_war"]), slot=r1(t["slot_value"]),
                       delta=r1(t["delta"]), mb=int(t["model_better"]), tb=int(t["team_better"])) for t in R["sequential"]["model_expanding"]["teams"]]
D["seq_by_year"] = R["sequential"]["model_expanding"]["by_year"]
old = {t["abbr"]: t for t in R["old_method_team_totals"]}
D["warriors"] = dict(old_actual=r1(old["GSW"]["actual"]), old_model=r1(old["GSW"]["model"]),
                     seq=next(t for t in D["seq_teams"] if t["abbr"] == "GSW"))
gs = sr[sr.abbr == "GSW"].sort_values(["year", "pick"])
D["warriors_picks"] = [dict(year=int(r.year), pick=int(r.pick), actual=r.actual_name, actual_war=r1(r.actual_war), model=r.model_name, model_war=r1(r.model_war)) for r in gs.itertuples()]
# horizon
D["horizon_train"] = {k: r3(v["mean"]) for k, v in R["horizon_train"].items()}
D["horizon_test"] = {y: {k: dict(model=r3(v["model_strict"]), teams=r3(v["teams"]), colin=r3(v.get("colin")), vo_model=r1(v["vo_pick_model_strict"]), vo_teams=r1(v["vo_pick_teams"])) for k, v in row.items()} for y, row in R["horizon_test"].items()}
# incremental
inc = R["incremental"].get("model_expanding|consensus_multisite", {})
D["blend"] = [dict(a=float(a), sp=r3(v)) for a, v in sorted(inc.get("blend", {}).items(), key=lambda kv: float(kv[0]))]
D["partial"] = inc.get("partial", [])
inc2 = R["incremental"].get("model_expanding|ridge_lite_cons", {})
D["blend_ridge"] = [dict(a=float(a), sp=r3(v)) for a, v in sorted(inc2.get("blend", {}).items(), key=lambda kv: float(kv[0]))]
# alt outcomes
D["alt"] = {alt: {nm: r3(np.mean([rows[y][nm] for y in rows if nm in rows[y]])) for nm in ["model_strict", "model_expanding", "colin", "consensus_multisite", "teams"]} for alt, rows in R["alt_outcomes"].items()}
# bootstrap
D["boot"] = {k: [dict(lo=r3(b["diff_ci"][0]), hi=r3(b["diff_ci"][1]), p=r3(b["p_a_gt_b"])) for b in v] for k, v in R["bootstrap"].items()}
# undrafted
D["undrafted"] = {}
for mode, res in U.items():
    D["undrafted"][mode] = {int(y): dict(n_full=r["n_full"], n_drafted=r["n_drafted"], n_undrafted=r["n_undrafted"], sp_model=r3(r["spearman_full_model"]), sp_teams=r3(r["spearman_full_teams"]),
                                       sp_model_d=r3(r["spearman_drafted_model"]), sp_teams_d=r3(r["spearman_drafted_teams"]), und_top=r["undrafted_in_model_top_n"], und_war=r1(r["undrafted_war_in_model_top_n"]),
                                       gems=r["gems"]) for y, r in res.items()}
# experiments
def gdiff(g):
    return ", ".join(f"{k}={v}" for k, v in g.items() if BASE_G.get(k) != v) or "base"
D["experiments"] = [dict(name=e["name"], change=gdiff(e["genome"]), wf_sp=r3(e["wf_spearman"]), wf_sp5=r3(e["wf_spearman_k5"]), wf_vo=r1(e["wf_vo_pick"]), wf_vo5=r1(e["wf_vo_pick_k5"]),
                         folds_won_sp=None, strict_sp=r3(e.get("strict_mean_spearman")), strict_vo=r1(e.get("strict_mean_vo_pick")), exp_sp=r3(e.get("expanding_mean_spearman")), exp_vo=r1(e.get("expanding_mean_vo_pick")),
                         wf_top5=None if e.get("wf_top5_hits") is None else round(float(e["wf_top5_hits"]), 2),
                         strict_top5=None if e.get("strict_mean_top5_hits") is None else round(float(e["strict_mean_top5_hits"]), 2),
                         exp_top5=None if e.get("expanding_mean_top5_hits") is None else round(float(e["expanding_mean_top5_hits"]), 2),
                         secs=e.get("secs")) for e in EXP]
base = next((e for e in EXP if e["name"] == "base"), None)
if base:
    for x, e in zip(D["experiments"], EXP):
        x["folds_won_sp"] = int(sum(e["folds"][y]["spearman_avg"] > base["folds"][y]["spearman_avg"] for y in e["folds"]))
        x["folds_won_vo"] = int(sum(e["folds"][y]["vo_pick_avg"] > base["folds"][y]["vo_pick_avg"] for y in e["folds"]))
        x["strict_teams_sp"] = r3(e.get("strict_teams_spearman")); x["strict_teams_vo"] = r1(e.get("strict_teams_vo_pick"))
        x["exp_teams_sp"] = r3(e.get("expanding_teams_spearman")); x["exp_teams_vo"] = r1(e.get("expanding_teams_vo_pick"))
D["generated"] = pd.Timestamp.today().strftime("%Y-%m-%d")

# ---------------------------------------------------------------- page
HTML = open(f"{os.path.dirname(os.path.abspath(__file__))}/dashboard_template.html").read()
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
page = HTML.replace("/*__DATA__*/", json.dumps(clean(D), allow_nan=False, default=str))
out = f"{C.REPO}/lab.html"
open(out, "w").write(page)
print("wrote", out, len(page) // 1024, "KB")
