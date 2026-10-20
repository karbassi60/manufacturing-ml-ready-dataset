"""Check of two statements about the ORIGINAL submission (Supplementary Material, Section S3).

The original manuscript reported (i) an accuracy of 98.5% for efficiency classification and (ii) a test R2 of 0.7075 for
the regression model, with Production_Hours as the target of the regression (Table 2 of the original submission).
The original script is not part of this repository; this script only shows, on random 80/20 splits of ml_ready_full.csv,
that (i) is what leakage through Equipment_Age_Category produces and (ii) is of the size of a ridge regression for
Production_Hours. It is a consistency check, not a replication.

Usage: python python/original_submission_check.py [repo_root]   ->  results/original_submission_check.json
"""
import json, sys, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
d = pd.read_csv(ROOT / "data" / "ml_ready_full.csv")
num = d.select_dtypes("number")

# efficiency regime: terciles of Machine_Efficiency (full sample here; the check concerns leakage, not the split)
cut = np.quantile(num["Machine_Efficiency"], [1 / 3, 2 / 3])
regime = np.digitize(num["Machine_Efficiency"], cut)
base = [c for c in num.columns if c not in ("Machine_Efficiency",) and not c.startswith("Department_")]
age = [c for c in base if c.startswith("Equipment_Age_Category")]
no_age = [c for c in base if c not in age]
SPL = range(10)
res = {}
for name, cols in (("with_Equipment_Age_Category", base), ("without_Equipment_Age_Category", no_age)):
    acc = []
    for s in SPL:
        Xtr, Xte, ytr, yte = train_test_split(num[cols].values, regime, test_size=0.2, random_state=s)
        acc.append(accuracy_score(yte, LogisticRegression(max_iter=5000).fit(Xtr, ytr).predict(Xte)))
    res[name] = dict(n_features=len(cols), accuracy_mean=float(np.mean(acc)), accuracy_min=float(np.min(acc)),
                     accuracy_max=float(np.max(acc)), accuracy_by_split=[float(a) for a in acc])

r2 = []
cols = [c for c in num.columns if c not in ("Production_Hours",) and not c.startswith("Department_") and c != "Production_Volume_Per_Hour"]
for s in SPL:
    Xtr, Xte, ytr, yte = train_test_split(num[cols].values, num["Production_Hours"].values, test_size=0.2, random_state=s)
    r2.append(r2_score(yte, Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)))
res["ridge_Production_Hours"] = dict(n_features=len(cols), r2_mean=float(np.mean(r2)), r2_min=float(np.min(r2)), r2_max=float(np.max(r2)),
                                     r2_by_split=[float(a) for a in r2])
json.dump(res, open(ROOT / "results" / "original_submission_check.json", "w"), indent=1)
print(json.dumps({k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in x.items() if not m.endswith("by_split")} for k, x in res.items()}, indent=1))
