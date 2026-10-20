"""Conventional baselines on the same 15 predictors, targets and split as matlab/newfinal_upd.m.

Naive predictors, ridge / (multinomial) logistic regression, Random Forest, XGBoost and LightGBM.
Hyper-parameters (9 configurations per model) are chosen on the validation block (mean validation loss over three
seeds); final models are fitted on the training block with ten seeds and evaluated once on the test block.

Usage:  python python/baselines.py [repo_root]
Writes  results/baselines_by_seed.csv, results/baselines_summary.json, results/baselines_seed_avg_predictions.npz
"""
import json, sys, pathlib, itertools, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
sys.path.insert(0, str(HERE))
import common as C
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
D = C.load(ROOT / "data" / "ml_ready_full.csv")
X, yr, yb, yo = D["X"], D["yr"], D["yb"], D["yo"]
tr, va, te = C.TRAIN, C.VAL, C.TEST
SEEDS = list(range(1, 11)); TUNE_SEEDS = [1, 2, 3]
op_cols = [D["pred_names"].index(c) for c in ["Workforce_Size", "Raw_Materials_Cost", "Shift_Hours"]]

# ---- grids: nine configurations per model ---------------------------------------------------------
def grid_rf():  return [dict(max_depth=d, min_samples_leaf=l) for d, l in itertools.product([3, 5, 8], [5, 20, 50])]
def grid_xgb(): return [dict(max_depth=d, learning_rate=lr) for d, lr in itertools.product([2, 3, 4], [0.03, 0.1, 0.3])]
def grid_lgb(): return [dict(num_leaves=n, learning_rate=lr) for n, lr in itertools.product([4, 8, 16], [0.03, 0.1, 0.3])]
def grid_lin(): return [dict(alpha=a) for a in [1e-2, 1e-1, 1, 10, 100, 300, 1000, 3000, 10000]]
def grid_log(): return [dict(C=c) for c in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]]

def make(task, name, cfg, seed):
    if name == "RF":
        kw = dict(n_estimators=300, random_state=seed, n_jobs=1, **cfg)
        return RandomForestRegressor(**kw) if task == "reg" else RandomForestClassifier(**kw)
    if name == "XGBoost":
        kw = dict(n_estimators=150, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbosity=0, **cfg)
        return XGBRegressor(**kw) if task == "reg" else XGBClassifier(**kw)
    if name == "LightGBM":
        kw = dict(n_estimators=150, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                  random_state=seed, n_jobs=1, verbose=-1, **cfg)
        return LGBMRegressor(**kw) if task == "reg" else LGBMClassifier(**kw)
    if name in ("Linear", "Linear (3 operating variables)"):
        return Ridge(**cfg) if task == "reg" else LogisticRegression(max_iter=2000, **cfg)
    raise ValueError(name)

GRIDS = {"RF": grid_rf, "XGBoost": grid_xgb, "LightGBM": grid_lgb}
def grid_for(task, name):
    if name.startswith("Linear"):
        return grid_lin() if task == "reg" else grid_log()
    return GRIDS[name]()

def cols_for(name):
    return op_cols if name == "Linear (3 operating variables)" else list(range(X.shape[1]))

def fit_predict(task, name, cfg, seed, rows_fit, rows_pred):
    cols = cols_for(name)
    m = make(task, name, cfg, seed)
    target = {"reg": yr, "bin": yb, "ord": yo - 1}[task]   # ordinal classes coded 0..2 for the learners
    m.fit(X[rows_fit][:, cols], target[rows_fit])
    if task == "reg":
        return m.predict(X[rows_pred][:, cols])
    pr = m.predict_proba(X[rows_pred][:, cols])
    if task == "bin":
        return pr[:, list(m.classes_).index(1)]
    full = np.zeros((len(rows_pred), 3))
    for j, c in enumerate(m.classes_):
        full[:, int(c)] = pr[:, j]
    return full

def val_loss(task, p):
    if task == "reg":
        return mean_squared_error(yr[va], p)
    if task == "bin":
        return log_loss(yb[va], np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
    return log_loss(yo[va], np.clip(p, 1e-6, 1), labels=[1, 2, 3])

results, avg_preds, chosen = [], {}, {}
MODELS = ["Linear", "Linear (3 operating variables)", "RF", "XGBoost", "LightGBM"]
for task in ["reg", "bin", "ord"]:
    for name in MODELS:
        if name == "Linear (3 operating variables)" and task != "ord":
            continue
        best, best_loss = None, np.inf
        for cfg in grid_for(task, name):
            losses = [val_loss(task, fit_predict(task, name, cfg, s, tr, va)) for s in TUNE_SEEDS]
            if np.mean(losses) < best_loss:
                best, best_loss = cfg, float(np.mean(losses))
        chosen[f"{task}/{name}"] = dict(config=best, mean_val_loss=best_loss)
        preds = [fit_predict(task, name, best, s, tr, te) for s in SEEDS]
        if task == "bin":   # extra, otherwise unused blocks to check whether apparent discrimination replicates
            p_va = [fit_predict(task, name, best, s, tr, va) for s in SEEDS]
            p_ca = [fit_predict(task, name, best, s, tr, C.CAL) for s in SEEDS]
        avg_preds[f"{task}/{name}"] = np.mean(preds, axis=0)
        for s, p in zip(SEEDS, preds):
            row = dict(task=task, model=name, seed=s)
            if task == "reg":
                row.update(r2=C.r2(yr[te], p), rmse=C.rmse(yr[te], p), mae=C.mae(yr[te], p), spearman=C.spearman(yr[te], p))
            elif task == "bin":
                b = C.binary_at(yb[te], p, 0.5)
                k = s - 1
                row.update(ap=C.ap(yb[te], p), roc_auc=C.auc(yb[te], p), brier=C.brier(yb[te], p), mcc_05=b["mcc"], bin_flagged_05=b["n_flagged"],
                           roc_auc_val=C.auc(yb[va], p_va[k]), roc_auc_cal=C.auc(yb[C.CAL], p_ca[k]),
                           ap_val=C.ap(yb[va], p_va[k]), ap_cal=C.ap(yb[C.CAL], p_ca[k]))
            else:
                pred = p.argmax(1) + 1
                row.update(ord_accuracy=C.acc(yo[te], pred), ord_macro_f1=C.macro_f1(yo[te], pred))
            results.append(row)

# ---- naive baselines (training mean / prior / majority class) -------------------------------------
naive = {}
mu = float(yr[tr].mean())
naive["reg"] = dict(r2=C.r2(yr[te], np.full(te.size, mu)), rmse=C.rmse(yr[te], np.full(te.size, mu)), mae=C.mae(yr[te], np.full(te.size, mu)))
pi = float(yb[tr].mean())
naive["bin"] = dict(ap=C.ap(yb[te], np.full(te.size, pi)), roc_auc=0.5, brier=C.brier(yb[te], np.full(te.size, pi)), accuracy=float(1 - yb[te].mean()))
maj = int(np.bincount(yo[tr]).argmax())
naive["ord"] = dict(ord_accuracy=C.acc(yo[te], np.full(te.size, maj)), ord_macro_f1=C.macro_f1(yo[te], np.full(te.size, maj)), majority_class=maj)

R = pd.DataFrame(results)
R.to_csv(OUT / "baselines_by_seed.csv", index=False)
summ = {"naive": naive, "chosen": chosen, "models": {}}
for (task, model), g in R.groupby(["task", "model"]):
    summ["models"][f"{task}/{model}"] = {c: [float(g[c].mean()), float(g[c].std(ddof=1)) if len(g) > 1 else 0.0]
                                         for c in g.columns if c not in ("task", "model", "seed") and g[c].notna().any()}
json.dump(summ, open(OUT / "baselines_summary.json", "w"), indent=1)
np.savez(OUT / "baselines_seed_avg_predictions.npz", **{k.replace("/", "__"): v for k, v in avg_preds.items()})
print(json.dumps({k: {m: round(v[0], 3) for m, v in d.items()} for k, d in summ["models"].items()}, indent=0))
print(json.dumps(summ["naive"]))
print(json.dumps({k: v["config"] for k, v in chosen.items()}))
