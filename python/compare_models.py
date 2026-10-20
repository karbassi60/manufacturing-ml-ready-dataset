"""Bootstrap comparison of the multi-task network with the conventional baselines (seed-averaged test predictions).
Usage: python python/compare_models.py [repo_root]  ->  results/model_comparison.json
"""
import json, sys, pathlib
import numpy as np, scipy.io as sio
HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
sys.path.insert(0, str(HERE))
import common as C
D = C.load(ROOT / "data" / "ml_ready_full.csv"); te = C.TEST
yr, yb, yo = D["yr"][te], D["yb"][te], D["yo"][te]
M = sio.loadmat(ROOT / "matlab_outputs" / "analysis_results.mat", squeeze_me=True)
P = np.load(ROOT / "results" / "baselines_seed_avg_predictions.npz")
mtl = dict(reg=M["PRm"].mean(1), bin=M["PBm"].mean(1), ord=M["POm"].mean(2))
stl = dict(reg=M["PRs"].mean(1), bin=M["PBs"].mean(1), ord=M["POs"].mean(2))
rng = np.random.default_rng(20261020); B = 2000
idx = [rng.integers(0, te.size, te.size) for _ in range(B)]
def metric(task, pred, i):
    if task == "reg":  return dict(r2=C.r2(yr[i], pred[i]))
    if task == "bin":  return dict(ap=C.ap(yb[i], pred[i]), roc_auc=C.auc(yb[i], pred[i])) if 0 < yb[i].sum() < i.size else dict(ap=np.nan, roc_auc=np.nan)
    return dict(ord_macro_f1=C.macro_f1(yo[i], pred[i].argmax(1) + 1))
def ci(x): return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
out = {}
for key in P.files:
    task, name = key.split("__")
    pred = P[key]
    ref = mtl[task]
    full = {k: v for k, v in metric(task, pred, np.arange(te.size)).items()}
    diffs = {}
    for i in idx:
        a, b = metric(task, ref, i), metric(task, pred, i)
        for k in a:
            diffs.setdefault(k, []).append(a[k] - b[k])
    out[f"{task}/{name}"] = dict(seed_avg=full, mtl_minus_baseline_ci=({k: ci(np.array(v)) for k, v in diffs.items()}))
json.dump(out, open(ROOT / "results" / "model_comparison.json", "w"), indent=1)
for k, v in out.items():
    print(k, {m: round(x, 3) for m, x in v["seed_avg"].items()}, {m: [round(a, 3) for a in c] for m, c in v["mtl_minus_baseline_ci"].items()})
