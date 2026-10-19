"""Run a named list of genome variants sequentially; each writes out/exp_<name>.json and a line in out/experiments.jsonl."""
import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
QUEUES = {
 "A": [
  ("hw_uniform", {"hw": "uniform"}),
  ("hw_late", {"hw": "late"}),
  ("hw_back", {"hw": "back"}),
  ("hw_disc70", {"hw": "disc70"}),
  ("risk_mean", {"risk": "mean"}),
  ("topw_05", {"topw": 0.5}),
  ("topw_10", {"topw": 1.0}),
  ("consres_05", {"consres": 0.5}),
  ("consres_10", {"consres": 1.0}),
  ("cons_value", {"cons_value": 1}),
  ("combine_flags", {"combine_flags": 1}),
  ("wobj_vopick", {"wobj": "vo_pick"}),
  ("xgb_only", {"members": ["xgb"]}),
  ("ridge_only", {"members": ["ridge"]}),
  ("xgb_depth4", {"xgb_depth": 4}),
  ("risk_mean_hw_uniform", {"risk": "mean", "hw": "uniform"}),
 ],
 "C": [
  ("anchor3", {"anchor_k": 3}),
  ("anchor5", {"anchor_k": 5}),
  ("anchor3_wobj_vopick", {"anchor_k": 3, "wobj": "vo_pick"}),
 ],
 "G": [   # the GPU queue: the champion's TabICL recipe (32 estimators, 3 permutations) with and without the quantile objective
  ("tabicl32_only", {"members": ["tabicl"], "icl_n": 32, "icl_bag": 3}),
  ("base_tabicl32", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3}),
  ("risk_mean_tabicl32", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "risk": "mean"}),
 ],
 "B": [
  ("base_tabicl", {"members": ["xgb", "ridge", "tabicl"]}),
 ],
}
q = sys.argv[1]
for name, over in QUEUES[q]:
    g = dict(E.BASE); g.update(over)
    try:
        E.run(name, g)
    except Exception:
        traceback.print_exc()
print("QUEUE_DONE", q, flush=True)
