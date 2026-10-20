"""Column audit of data/ml_ready_full.csv for Supplementary Table S2.

For every column (or group of one-hot columns of one categorical variable): association with the three targets, relation to
other columns, and use in the shared input. Associations use all 1,000 records (descriptive; no model is fitted):
numerical columns - Spearman correlation with Production_Volume_Per_Hour, Pearson (point-biserial) correlation with
Department_Maintenance and Pearson correlation with Machine_Efficiency; categorical columns - eta^2 (share of the variance
of the rank-based normal scores of Production_Volume_Per_Hour, or of Machine_Efficiency, explained by the categories) and
Cramer's V with Department_Maintenance.

Usage: python python/column_audit.py [repo_root]  ->  results/column_audit.csv
"""
import sys, pathlib
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, norm

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
T = pd.read_csv(ROOT / "data" / "ml_ready_full.csv")
n = len(T)
pvh = T["Production_Volume_Per_Hour"].to_numpy(float)
ns = norm.ppf((stats.rankdata(pvh) - 0.5) / n)          # rank-based normal scores (full sample; descriptive use only)
eff = T["Machine_Efficiency"].to_numpy(float)
dept = T["Department_Maintenance"].to_numpy(float)


def eta2(y, g):
    g = np.asarray(g)
    ss_t = ((y - y.mean()) ** 2).sum()
    ss_b = sum(((y[g == k] - y.mean()) ** 2).sum() * 0 + (g == k).sum() * (y[g == k].mean() - y.mean()) ** 2 for k in np.unique(g))
    return float(ss_b / ss_t)


def cramer(a, b):
    tab = pd.crosstab(a, b).to_numpy()
    chi2 = chi2_contingency(tab, correction=False)[0]
    return float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))))


def label_from_onehot(cols):
    return np.array(cols)[T[cols].to_numpy().argmax(1)]


rows = []


def numeric(col, relation="", used="", ctype="numerical", ncols=1, label=None):
    v = T[col].to_numpy(float)
    rows.append(dict(column=label or col, n_columns=ncols, type=ctype,
                     pvh=f"{stats.spearmanr(v, pvh).statistic:.2f}", dept=f"{np.corrcoef(v, dept)[0, 1]:.2f}", eff=f"{np.corrcoef(v, eff)[0, 1]:.2f}",
                     relation=relation, used=used))


def categorical(cols, label, relation="", used="", textcol=None):
    g = T[textcol].to_numpy() if textcol else label_from_onehot(cols)
    k = len(np.unique(g))
    rows.append(dict(column=label, n_columns=len(cols) if cols else 1, type=f"categorical ({k} levels)",
                     pvh=f"η² = {eta2(ns, g):.3f}", dept=f"V = {cramer(g, T['Department_Maintenance']):.2f}", eff=f"η² = {eta2(eff, g):.2f}",
                     relation=relation, used=used))


numeric("Production_Hours", "enters the construction of Production_Volume_Per_Hour (Section 4.4)", "no (source of the regression target)")
numeric("Workforce_Size", "", "yes")
numeric("Machine_Efficiency", "", "no (source of the ordinal target)")
numeric("Raw_Materials_Cost", "", "yes")
numeric("Shift_Hours", "", "yes")
numeric("Production_Volume_Per_Hour", "", "no (regression target)")
numeric("Timestamp_Hour", "hour of the day (0–23)", "yes")
numeric("DayOfYear", "stored twice; Month is identical", "no (duplicate)", ncols=2, label="DayOfYear (2 copies)")
numeric("Month", "identical to DayOfYear", "no (duplicate)", ncols=2, label="Month (2 copies)")
numeric("Weekday", "12 distinct values; ρ = 0.996 with DayOfYear, i.e. a calendar-month index; second copy not used", "yes (first copy; calendar index)", ncols=2, label="Weekday (2 copies)")
categorical([c for c in T.columns if c.startswith("Supplier_")], "Supplier", "", "yes (10 indicators)")
categorical([c for c in T.columns if c.startswith("Department_")], "Department", "defines Department_Maintenance", "no (source of the binary target)")
categorical([c for c in T.columns if c.startswith("Equipment_Age_Category_")], "Equipment_Age_Category", "recoding of Machine_Efficiency (cut points ≈ −0.43 and 0.45; agrees with the regime for 961 records)", "no (leakage)")
categorical(None, "Efficiency_Level", "recoding of Machine_Efficiency (cut points ≈ −0.42 and 0.47; agrees with the regime for 949 records); not in the Kaggle file", "no (leakage)", textcol="Efficiency_Level")
categorical([c for c in T.columns if c.startswith("Shift_Category_")], "Shift_Category", "recoding of Shift_Hours (cut points ≈ −1.09 and 0.79)", "no (redundant)")
categorical([c for c in T.columns if c.startswith("Supply_Chain_Resilience_")], "Supply_Chain_Resilience", "recoding of Raw_Materials_Cost (cut points ≈ −1.08 and 0.95); also stored as text", "no (redundant)")
rows[-1]["n_columns"] = 4   # three indicators and a text copy
categorical(None, "Maintenance_Event", "undocumented; not in the Kaggle file; strongly related to the operating variables (Section 4.6)", "no (definition and timing not established)", textcol="Maintenance_Event")
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "column_audit.csv", index=False)
print(out.to_string())
print("columns covered:", int(out.n_columns.sum()), "of", T.shape[1])
