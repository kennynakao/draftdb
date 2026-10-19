"""Markdown table of every engine variant vs base (folds first, blind looks last)."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
E = {json.load(open(p))["name"]: json.load(open(p)) for p in sorted(glob.glob(f"{C.OUT}/exp_*.json"))}
base = E["base"]
def diff(g): return ", ".join(f"{k}={v}" for k, v in g.items() if base["genome"].get(k) != v) or "base"
def won(e, key): return sum(e["folds"][y][key] > base["folds"][y][key] for y in e["folds"])
rows = []
for n, e in E.items():
    rows.append((n, diff(e["genome"]), e["wf_spearman"], won(e, "spearman_avg"), e["wf_vo_pick"], won(e, "vo_pick_avg"),
                 e.get("strict_mean_spearman"), e.get("strict_mean_vo_pick"), e.get("expanding_mean_spearman"), e.get("expanding_mean_vo_pick")))
rows.sort(key=lambda r: -r[2])
print("| variant | change | folds Spearman | won | folds pick-value | won | strict Sp. | strict PV | expanding Sp. | expanding PV |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    f = lambda x, d: "–" if x is None else f"{x:.{d}f}"
    print(f"| {r[0]} | {r[1]} | {f(r[2],3)} | {r[3]}/7 | {f(r[4],1)} | {r[5]}/7 | {f(r[6],3)} | {f(r[7],1)} | {f(r[8],3)} | {f(r[9],1)} |")
print(f"\nteams: strict Spearman {base['strict_teams_spearman']:.3f} / pick-value {base['strict_teams_vo_pick']:.1f}; expanding {base['expanding_teams_spearman']:.3f} / {base['expanding_teams_vo_pick']:.1f}")

print("\nper-fold Spearman (avg over k=1..5), 2012..2018:")
for r in rows:
    e = E[r[0]]
    print(f"  {r[0]:22s} " + " ".join(f"{e['folds'][y]['spearman_avg']:.3f}" for y in sorted(e["folds"])) + "   | PV " + " ".join(f"{e['folds'][y]['vo_pick_avg']:.1f}" for y in sorted(e["folds"])))
