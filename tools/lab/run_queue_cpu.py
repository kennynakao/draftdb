"""Run a GPU queue's arms on the CPU recipe (TabICL member dropped, everything else unchanged), as an independent second
read of each A/B change. Names are prefixed c_; outputs go to DRAFTDB_LAB_OUT (use experiments/lab_2026-10-19/cpu_<q>)."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_queue.py")).read().split("q = sys.argv[1]")[0]
ns = {"__file__": __file__}; exec(src, ns)
q = sys.argv[1]
for name, over in ns["QUEUES"][q]:
    g = dict(E.BASE); g.update(over)
    g["members"] = [m for m in g["members"] if m != "tabicl"]
    for k in ("icl_n", "icl_bag"): g.pop(k, None)
    try:
        E.run("c_" + name, g)
    except Exception:
        traceback.print_exc()
print("QUEUE_DONE", q, flush=True)
