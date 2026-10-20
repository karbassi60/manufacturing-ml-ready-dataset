"""Descriptive facts about data/ml_ready_full.csv quoted in the manuscript (Sections 4.1-4.6, 5, Supplementary Table S2).

Usage: python python/data_facts.py [repo_root]      Writes results/data_facts.json
Every number in the data description of the paper is produced here (the 15 predictors, the three targets,
the excluded recodings, the calendar columns, the split blocks).
"""
import json, sys, pathlib, hashlib
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
sys.path.insert(0, str(HERE))
import common as C

csv = ROOT / "data" / "ml_ready_full.csv"
D = C.load(csv)
T = D["T"]
F = {}
F["sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
F["shape"] = list(T.shape)
F["dtype_counts"] = {k: int(v) for k, v in T.dtypes.astype(str).value_counts().items()}
F["string_columns"] = [c for c in T.columns if T[c].dtype == object or str(T[c].dtype).startswith("str")]
F["missing_cells"] = int(T.isna().sum().sum())
F["exact_duplicate_rows"] = int(T.duplicated().sum())

num = ["Production_Hours", "Workforce_Size", "Machine_Efficiency", "Raw_Materials_Cost", "Shift_Hours", "Production_Volume_Per_Hour"]
F["numeric_mean_sd"] = {c: [float(T[c].mean()), float(T[c].std(ddof=1))] for c in num}
y = T["Production_Volume_Per_Hour"].to_numpy(float)
F["pvh"] = dict(min=float(y.min()), max=float(y.max()), q1=float(np.quantile(y, .25)), median=float(np.median(y)),
                q3=float(np.quantile(y, .75)), skew=float(stats.skew(y)), excess_kurtosis=float(stats.kurtosis(y)),
                n_abs_gt5=int((np.abs(y) > 5).sum()),
                prodhours_range_abs_gt5=[float(T.Production_Hours[np.abs(y) > 5].min()), float(T.Production_Hours[np.abs(y) > 5].max())],
                n_abs_gt1=int((np.abs(y) > 1).sum()))
# normal-score transform of the target
F["normal_score_sd_test"] = float(D["yr"][C.TEST].std(ddof=1))
F["normal_score_mean_square_test_about_train_mean"] = float(np.mean((D["yr"][C.TEST] - D["yr"][C.TRAIN].mean()) ** 2))

# calendar columns
d, w = T["DayOfYear"].to_numpy(float), T["Weekday"].to_numpy(float)
F["calendar"] = dict(n_distinct_dayofyear=int(T.DayOfYear.nunique()), n_distinct_weekday_col=int(T.Weekday.nunique()),
                     n_distinct_hours=int(T.Timestamp_Hour.nunique()),
                     month_equals_dayofyear=bool(np.allclose(T.DayOfYear, T.Month)),
                     dup_cols=bool(np.allclose(T.DayOfYear, T["DayOfYear.1"]) and np.allclose(T.Month, T["Month.1"]) and np.allclose(T.Weekday, T["Weekday.1"])),
                     spearman_weekday_dayofyear=float(stats.spearmanr(d, w).statistic),
                     spearman_fileorder_dayofyear=float(stats.spearmanr(np.arange(len(T)), d).statistic),
                     n_duplicate_day_hour_pairs=int(T.duplicated(["DayOfYear", "Timestamp_Hour"]).sum()))
blocks = {"train": C.TRAIN, "validation": C.VAL, "calibration": C.CAL, "test": C.TEST}
F["block_dayofyear_range"] = {k: [float(d[i].min()), float(d[i].max())] for k, i in blocks.items()}
# time-ordered autocorrelation
order = np.lexsort((T.Timestamp_Hour.to_numpy(), d))
yo = D["yo"]
tv = {"production_volume": y[order], "department_maintenance": D["yb"][order], "efficiency_regime": yo[order]}
F["time_order"] = {k: dict(lag1_spearman=float(stats.spearmanr(v[:-1], v[1:]).statistic),
                           trend_spearman=float(stats.spearmanr(np.arange(len(v)), v).statistic)) for k, v in tv.items()}
fv = {"production_volume": y, "department_maintenance": D["yb"], "efficiency_regime": yo}
F["file_order"] = {k: dict(lag1_spearman=float(stats.spearmanr(v[:-1], v[1:]).statistic),
                           trend_spearman=float(stats.spearmanr(np.arange(len(v)), v).statistic)) for k, v in fv.items()}

# groups
sup = [c for c in T.columns if c.startswith("Supplier_")]
dep = [c for c in T.columns if c.startswith("Department_")]
F["supplier_counts_min_max"] = [int(T[sup].sum().min()), int(T[sup].sum().max())]
F["department_counts"] = {c: int(T[c].sum()) for c in dep}
F["n_predictors"] = len(D["pred_names"])
F["pred_names"] = D["pred_names"]

# split blocks: class counts
bn = {}
for k, i in blocks.items():
    bn[k] = dict(n=int(i.size), dept_neg=int((D["yb"][i] == 0).sum()), dept_pos=int((D["yb"][i] == 1).sum()),
                 regime_counts=np.bincount(yo[i], minlength=4)[1:].tolist())
F["blocks"] = bn
F["tercile_cut"] = D["cut"].tolist()

# leakage-related recodings
eff = T["Machine_Efficiency"].to_numpy(float)
agecode = np.where(T["Equipment_Age_Category_Old"] == 1, 1, np.where(T["Equipment_Age_Category_Medium"] == 1, 2, 3))
F["equipment_age_vs_regime_agree"] = int((agecode == yo).sum())
F["equipment_age_efficiency_ranges"] = {n: [float(eff[agecode == k].min()), float(eff[agecode == k].max())] for k, n in [(1, "Old"), (2, "Medium"), (3, "New")]}
el = T["Efficiency_Level"].map({"Low": 1, "Medium": 2, "High": 3}).to_numpy()
F["efficiency_level_vs_regime_agree"] = int((el == yo).sum())
F["efficiency_level_ranges"] = {n: [float(eff[el == k].min()), float(eff[el == k].max())] for k, n in [(1, "Low"), (2, "Medium"), (3, "High")]}
F["efficiency_level_counts"] = T["Efficiency_Level"].value_counts().to_dict()
shcode = np.where(T["Shift_Category_Low"] == 1, 1, np.where(T["Shift_Category_Medium"] == 1, 2, 3))
F["shift_category_ranges"] = {n: [float(T.Shift_Hours[shcode == k].min()), float(T.Shift_Hours[shcode == k].max())] for k, n in [(1, "Low"), (2, "Medium"), (3, "High")]}
scr = np.where(T["Supply_Chain_Resilience_Low"] == 1, 1, np.where(T["Supply_Chain_Resilience_Medium"] == 1, 2, 3))
F["scr_spearman_with"] = {c: float(stats.spearmanr(scr, T[c]).statistic) for c in ["Workforce_Size", "Raw_Materials_Cost", "Shift_Hours", "Machine_Efficiency", "Production_Hours"]}
F["scr_raw_materials_ranges"] = {n: [float(T.Raw_Materials_Cost[scr == k].min()), float(T.Raw_Materials_Cost[scr == k].max())] for k, n in [(1, "Low"), (2, "Medium"), (3, "High")]}
F["maintenance_event"] = dict(positives=int(T.Maintenance_Event.sum()), overlap_with_department_maintenance=int(((T.Maintenance_Event == 1) & (T.Department_Maintenance == 1)).sum()),
                              chi2_p_vs_department_maintenance=float(chi2_contingency(pd.crosstab(T.Maintenance_Event, T.Department_Maintenance))[1]))
# The undocumented Maintenance_Event column is NOT used as a target in the MATLAB analysis. We only document why it is not treated
# as a maintenance event: it is strongly related to the operating variables, as a threshold recoding of an operating variable would be.
me = T.Maintenance_Event.to_numpy(float)
F["maintenance_event"]["pointbiserial_with_operating"] = {c: float(np.corrcoef(T[c], me)[0, 1]) for c in
    ["Production_Hours", "Workforce_Size", "Machine_Efficiency", "Raw_Materials_Cost", "Shift_Hours"]}
F["maintenance_event"]["pointbiserial_with_pvh"] = float(np.corrcoef(T["Production_Volume_Per_Hour"], me)[0, 1])
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
_Xs = D["X"]            # the 15 standardised predictors (training-only statistics)
_lr = LogisticRegression(C=1.0, max_iter=2000).fit(_Xs[C.TRAIN], me[C.TRAIN])
F["maintenance_event"]["logistic_auc_validation"] = float(roc_auc_score(me[C.VAL], _lr.predict_proba(_Xs[C.VAL])[:, 1]))
F["maintenance_event"]["logistic_auc_test"] = float(roc_auc_score(me[C.TEST], _lr.predict_proba(_Xs[C.TEST])[:, 1]))
F["pvh"]["share_negative"] = float((y < 0).mean())

# associations of predictors / retained variables with targets
tr = C.TRAIN
cols = ["Workforce_Size", "Raw_Materials_Cost", "Shift_Hours", "Timestamp_Hour", "Weekday"]
F["train_spearman_with_pvh"] = {c: float(stats.spearmanr(T[c][tr], y[tr]).statistic) for c in cols}
F["train_corr_with_department"] = {c: float(np.corrcoef(T[c][tr], D["yb"][tr])[0, 1]) for c in cols + sup}
F["max_abs_train_corr_with_department"] = float(max(abs(v) for v in F["train_corr_with_department"].values()))
F["corr_with_machine_efficiency"] = {c: float(np.corrcoef(T[c], eff)[0, 1]) for c in ["Workforce_Size", "Raw_Materials_Cost", "Shift_Hours", "Production_Hours"]}
X3 = np.c_[np.ones(len(T)), T[["Workforce_Size", "Raw_Materials_Cost", "Shift_Hours"]].to_numpy()]
b = np.linalg.lstsq(X3, eff, rcond=None)[0]
F["r2_efficiency_on_3_operating_vars"] = float(1 - ((eff - X3 @ b) ** 2).sum() / ((eff - eff.mean()) ** 2).sum())
F["pearson_with_targets_all_cols"] = {c: [float(np.corrcoef(T[c], D["yr"])[0, 1]), float(np.corrcoef(T[c], D["yb"])[0, 1]), float(np.corrcoef(T[c], yo)[0, 1])]
                                     for c in D["pred_names"] + ["Machine_Efficiency", "Production_Hours"]}

def eta2(x, g):
    ss = ((x - x.mean()) ** 2).sum()
    return float(sum(len(x[g == k]) * (x[g == k].mean() - x.mean()) ** 2 for k in np.unique(g)) / ss)
dcode = T[dep].to_numpy().argmax(1)
scode = T[sup].to_numpy().argmax(1)
ncols = ["Production_Hours", "Workforce_Size", "Machine_Efficiency", "Raw_Materials_Cost", "Shift_Hours", "Production_Volume_Per_Hour", "Timestamp_Hour", "Weekday"]
F["eta2_department"] = {c: eta2(T[c].to_numpy(float), dcode) for c in ncols}
F["eta2_supplier"] = {c: eta2(T[c].to_numpy(float), scode) for c in ncols}
mcode = T.Weekday.round(4).astype("category").cat.codes.to_numpy()
F["chi2_p_department_vs"] = {n: float(chi2_contingency(pd.crosstab(dcode, g))[1]) for n, g in
                             [("supplier", scode), ("equipment_age", agecode), ("shift_category", shcode), ("supply_chain_resilience", scr), ("calendar_month_col", mcode)]}
F["kruskal_p_department_numeric"] = {c: float(stats.kruskal(*[T[c].to_numpy(float)[dcode == k] for k in range(4)]).pvalue) for c in ncols}
F["kruskal_p_supplier_numeric"] = {c: float(stats.kruskal(*[T[c].to_numpy(float)[scode == k] for k in range(10)]).pvalue) for c in ncols}
F["min_p_all_department_supplier_tests"] = float(min(list(F["chi2_p_department_vs"].values()) + list(F["kruskal_p_department_numeric"].values()) + list(F["kruskal_p_supplier_numeric"].values())))
# PCA of the 15 standardized inputs (training set), as in Fig. 2
Xt = D["X"][tr]
u, s, vt = np.linalg.svd(Xt - Xt.mean(0), full_matrices=False)
ev = s ** 2 / (s ** 2).sum()
F["pca_explained"] = ev[:3].tolist()
F["pca_pc1_loadings"] = dict(zip(D["pred_names"], vt[0].tolist()))
five = ["Production_Hours", "Workforce_Size", "Machine_Efficiency", "Raw_Materials_Cost", "Shift_Hours"]
cm = T[five].corr().to_numpy()
off = cm[np.triu_indices(5, 1)]
F["corr_five_operating"] = dict(min=float(off.min()), max=float(off.max()))
F["pvh_pearson_with_five_max_abs"] = float(max(abs(np.corrcoef(T[c], y)[0, 1]) for c in five))
sp = [stats.spearmanr(T[c], y).statistic for c in five]
F["pvh_spearman_with_five"] = dict(zip(five, [float(x) for x in sp]))
F["pca_pc2_loadings"] = dict(zip(D["pred_names"], vt[1].tolist()))
json.dump(F, open(ROOT / "results" / "data_facts.json", "w"), indent=1)
print("data facts written")
