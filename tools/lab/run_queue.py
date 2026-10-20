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
 "H": [   # GPU screen under both metrics: labels, objective, and every collected block (families from data v4.22)
  ("h_base", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3}),
  ("h_floor0", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "floor0": 1}),
  ("h_minw05", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "floor0": 1, "minw": 0.5}),
  ("h_risk_mean_floor0", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "risk": "mean", "floor0": 1}),
  ("h_topw05", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "topw": 0.5}),
  ("h_topk60", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "topk": 60}),
  ("h_topk150", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "topk": 150}),
  ("h_win2010", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "win": 2010}),
  ("h_no_sc_tc", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": []}),
  ("h_plus_f50", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "f50_"]}),
  ("h_plus_gls", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "gls_"]}),
  ("h_plus_fys", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "fys_"]}),
  ("h_plus_dxs", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "dxs_"]}),
  ("h_plus_bb", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "bb_"]}),
  ("h_plus_cp", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "cp_"]}),
  ("h_plus_tv", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "tv_"]}),
  ("h_plus_vcons", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "vcons_"]}),
  ("h_plus_mock", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "mock_"]}),
  ("h_plus_dv", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "dv_"]}),
  ("h_plus_wts", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "wts_"]}),
  ("h_plus_tv_vcons", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "extra": ["sc_", "tc_", "tv_", "vcons_"]}),
 ],
 "S": [   # H6 top-of-board specialist on the TabICL 32x3 recipe: a class-balanced top-3 classifier blended into the
          # consensus top 10 at weight star (0 = the recipe's base, trained but not blended, so its own fold score is reported)
  ("s_star00", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "star": 0.0}),
  ("s_star025", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "star": 0.25}),
  ("s_star05", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "star": 0.5}),
  ("s_star075", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "star": 0.75}),
  ("s_star10", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "star": 1.0}),
  ("s_star05_risk_mean", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "star": 0.5, "risk": "mean"}),
 ],
 "T": [   # 2026-10-20 A/B screen of untested recipe changes with a mechanism (see docs/EXPERIMENTS.md): recency weights for the
          # era shift, both XGBoost objectives in one stack, more bagging, ridge strength, the international mask, the thin/rich split,
          # a finer stack-weight grid, extra monotone constraints, a slower XGBoost
  ("t_base", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3}),
  ("t_recency4", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "recency": 4}),
  ("t_recency8", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "recency": 8}),
  ("t_xgb_both", {"members": ["xgb", "xgbm", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3}),
  ("t_risk_mean_bag9", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "risk": "mean", "bag": 9}),
  ("t_bag9", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "bag": 9}),
  ("t_icl_bag5", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 5}),
  ("t_ridge100", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "ridge_alpha": 100.0}),
  ("t_ridge1000", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "ridge_alpha": 1000.0}),
  ("t_ridge30", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "ridge_alpha": 30.0}),
  ("t_nomask", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "intl_mask": 0}),
  ("t_thin_q33", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "thin_q": 0.33}),
  ("t_wgrid125", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "wgrid": 0.125}),
  ("t_mono_impact", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "mono": {"col_impact": 1, "x_prod_over_age": 1}}),
  ("t_xgb_slow", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "xgb_lr": 0.01, "xgb_n": 1600}),
 ],
 "U": [   # 2026-10-20 combinations of the queue T leads (slower XGBoost, squared-error objective, thin/rich split at the lower
          # third, nine bags, both objectives, extra monotone constraints); compared with t_base, which the recipe reproduces exactly
  ("u_slow_mean", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "xgb_lr": 0.01, "xgb_n": 1600, "risk": "mean"}),
  ("u_slow_mean_bag9", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "xgb_lr": 0.01, "xgb_n": 1600, "risk": "mean", "bag": 9}),
  ("u_slow_thin33", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "xgb_lr": 0.01, "xgb_n": 1600, "thin_q": 0.33}),
  ("u_slow_mean_thin33", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "xgb_lr": 0.01, "xgb_n": 1600, "risk": "mean", "thin_q": 0.33}),
  ("u_slow_mean_bag9_thin33", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "xgb_lr": 0.01, "xgb_n": 1600, "risk": "mean", "bag": 9, "thin_q": 0.33}),
  ("u_both_slow_thin33", {"members": ["xgb", "xgbm", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "xgb_lr": 0.01, "xgb_n": 1600, "thin_q": 0.33}),
  ("u_slow_mean_thin33_mono", {"members": ["xgb", "ridge", "tabicl"], "icl_n": 32, "icl_bag": 3, "xgb_lr": 0.01, "xgb_n": 1600, "risk": "mean", "thin_q": 0.33, "mono": {"col_impact": 1, "x_prod_over_age": 1}}),
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
