"""Markdown table of every engine variant vs base (folds first, blind looks last).

Usage: summarize_experiments.py [subdir] [base-name]
  subdir     folder under the lab output dir holding exp_*.json (default: the output dir itself; e.g. gpu_s)
  base-name  the variant the others are compared with (default "base"; e.g. s_star00 for queue S)
Top-5 hits (how many of the board's first five are in the class's true top five) are shown where recorded."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
sub = sys.argv[1] if len(sys.argv) > 1 else ""
base_name = sys.argv[2] if len(sys.argv) > 2 else "base"
D = os.path.join(C.OUT, sub) if sub else C.OUT
E = {json.load(open(p))["name"]: json.load(open(p)) for p in sorted(glob.glob(f"{D}/exp_*.json"))}
base = E[base_name]
def diff(g): return ", ".join(f"{k}={v}" for k, v in g.items() if base["genome"].get(k) != v) or "base"
def won(e, key): return sum(e["folds"][y].get(key, 0) > base["folds"][y].get(key, 0) for y in e["folds"])
def f(x, d): return "–" if x is None else f"{x:.{d}f}"
rows = []
for n, e in E.items():
    rows.append((n, diff(e["genome"]), e["wf_spearman"], won(e, "spearman_avg"), e["wf_vo_pick"], won(e, "vo_pick_avg"),
                 e.get("wf_top5_hits"), won(e, "top5_hits_avg") if "wf_top5_hits" in e else None,
                 e.get("strict_mean_spearman"), e.get("strict_mean_vo_pick"), e.get("strict_mean_top5_hits"),
                 e.get("expanding_mean_spearman"), e.get("expanding_mean_vo_pick"), e.get("expanding_mean_top5_hits")))
rows.sort(key=lambda r: -r[2])
print("| variant | change | folds Spearman | won | folds pick-value | won | folds top-5 hits | won | strict Sp. / PV / top-5 | expanding Sp. / PV / top-5 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    w7 = lambda x: "–" if x is None else f"{x}/7"
    print(f"| {r[0]} | {r[1]} | {f(r[2],3)} | {w7(r[3])} | {f(r[4],1)} | {w7(r[5])} | {f(r[6],2)} | {w7(r[7])} | {f(r[8],3)} / {f(r[9],1)} / {f(r[10],2)} | {f(r[11],3)} / {f(r[12],1)} / {f(r[13],2)} |")
print(f"\nteams: strict Spearman {base['strict_teams_spearman']:.3f} / pick-value {base['strict_teams_vo_pick']:.1f}"
      + (f" / top-5 {base['strict_teams_top5_hits']:.2f}" if "strict_teams_top5_hits" in base else "")
      + f"; expanding {base['expanding_teams_spearman']:.3f} / {base['expanding_teams_vo_pick']:.1f}"
      + (f" / top-5 {base['expanding_teams_top5_hits']:.2f}" if "expanding_teams_top5_hits" in base else ""))

print("\nper-fold Spearman (avg over k=1..5), 2012..2018:")
for r in rows:
    e = E[r[0]]
    line = f"  {r[0]:22s} " + " ".join(f"{e['folds'][y]['spearman_avg']:.3f}" for y in sorted(e["folds"])) + "   | PV " + " ".join(f"{e['folds'][y]['vo_pick_avg']:.1f}" for y in sorted(e["folds"]))
    if "top5_hits_avg" in next(iter(e["folds"].values())):
        line += "   | top5 " + " ".join(f"{e['folds'][y]['top5_hits_avg']:.1f}" for y in sorted(e["folds"]))
    print(line)
