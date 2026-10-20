"""Imbalance handling for the binary task (Department_Maintenance) with conventional classifiers.

The MATLAB network does not implement any rebalancing, so this script applies the strategies to logistic regression
and Random Forest on the same predictors and split: (i) none, (ii) SMOTE on the training block only,
(iii) class weights ('balanced'). The decision threshold maximises the F1-score on the validation block.
Usage: python python/imbalance.py [repo_root]  ->  results/imbalance_summary.json, results/imbalance_by_seed.csv
"""
import json, sys, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
sys.path.insert(0, str(HERE))
import common as C
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
D = C.load(ROOT / "data" / "ml_ready_full.csv")
X, yb = D["X"], D["yb"]; tr, va, te = C.TRAIN, C.VAL, C.TEST
SEEDS = list(range(1, 11))

def model(name, weight, seed):
    cw = "balanced" if weight else None
    if name == "Logistic regression":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight=cw)
    return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20, class_weight=cw, random_state=seed, n_jobs=1)

rows = []
for name in ["Logistic regression", "Random Forest"]:
    for strat in ["None", "SMOTE (training block only)", "Class weights"]:
        for s in SEEDS:
            Xt, yt = X[tr], yb[tr]
            if strat.startswith("SMOTE"):
                Xt, yt = SMOTE(random_state=s, k_neighbors=5).fit_resample(Xt, yt)
            m = model(name, strat == "Class weights", s).fit(Xt, yt)
            pv, pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
            taus = np.unique(np.round(pv, 6))
            f1s = [f1_score(yb[va], (pv >= t).astype(int), zero_division=0) for t in taus]
            tau = float(taus[int(np.argmax(f1s))])
            b = C.binary_at(yb[te], pt, tau)
            rows.append(dict(model=name, strategy=strat, seed=s, ap=C.ap(yb[te], pt), roc_auc=C.auc(yb[te], pt),
                             roc_auc_val=C.auc(yb[va], pv), brier=C.brier(yb[te], pt), tau=tau,
                             balanced_accuracy=b["balanced_accuracy"], mcc=b["mcc"], recall=b["recall"], precision=b["precision"], flagged=b["n_flagged"]))
R = pd.DataFrame(rows); R.to_csv(OUT / "imbalance_by_seed.csv", index=False)
summ = {}
for (m, st), g in R.groupby(["model", "strategy"], sort=False):
    summ[f"{m} | {st}"] = {c: [float(g[c].mean()), float(g[c].std(ddof=1))] for c in g.columns if c not in ("model", "strategy", "seed")}
summ["no_skill"] = dict(ap=float(yb[te].mean()), roc_auc=0.5)
json.dump(summ, open(OUT / "imbalance_summary.json", "w"), indent=1)
for k, v in summ.items():
    if k != "no_skill":
        print(k, {c: round(x[0], 3) for c, x in v.items()})
