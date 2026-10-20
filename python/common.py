"""Shared data preparation and metrics for the OMEGA-D-26-00675.R1 analysis.

Mirrors the preprocessing in matlab/newfinal_upd.m exactly (same predictors, same split by
file order, same training-only scaling, same target construction), so that the Python
baselines and the summary statistics refer to the same data as the MATLAB networks.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             balanced_accuracy_score, matthews_corrcoef, f1_score, accuracy_score)

TRAIN = np.arange(0, 550)      # MATLAB idxTr = 1:550
VAL = np.arange(550, 700)      # idxVa = 551:700
CAL = np.arange(700, 800)      # idxCa = 701:800 (not used by the networks)
TEST = np.arange(800, 1000)    # idxTe = 801:1000

# Reference mean of the regression target in R^2: the TRAINING-set mean (Eq. (7) of the paper). It is set by load().
# The MATLAB function r2() in newfinal_upd.m uses the mean of the evaluated set instead; on the test set the two
# differ by 0.0001-0.0002 in R^2 (see results/summary.json, "r2_reference_mean_check").
TRAIN_MEAN_YR = 0.0


def interp_linear_extrap(x, xp, fp):
    """MATLAB interp1(xp, fp, x, 'linear', 'extrap')."""
    r = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    if lo.any():
        r[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if hi.any():
        r[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return r


def load(csv_path):
    T = pd.read_csv(csv_path)
    assert T.shape[0] == 1000, "expected 1,000 rows"
    sup = [c for c in T.columns if c.startswith("Supplier_")]
    pred_names = ["Workforce_Size", "Raw_Materials_Cost", "Shift_Hours", "Timestamp_Hour", "Weekday"] + sup
    Xraw = T[pred_names].to_numpy(float)
    mu = Xraw[TRAIN].mean(0)
    sd = Xraw[TRAIN].std(0, ddof=1)
    sd[sd < 1e-9] = 1.0
    X = (Xraw - mu) / sd
    # binary target: Department_Maintenance
    yb = T["Department_Maintenance"].to_numpy(int)
    # ordinal target: training terciles of Machine_Efficiency (MATLAB quantile = Hazen definition)
    eff = T["Machine_Efficiency"].to_numpy(float)
    cut = np.quantile(eff[TRAIN], [1 / 3, 2 / 3], method="hazen")
    yo = 1 + (eff > cut[0]).astype(int) + (eff > cut[1]).astype(int)
    # regression target: training-only empirical rank-normal transform of Production_Volume_Per_Hour
    y = T["Production_Volume_Per_Hour"].to_numpy(float)
    ys = np.sort(y[TRAIN])
    u = (np.arange(1, ys.size + 1) - 0.5) / ys.size
    yu, inv = np.unique(ys, return_inverse=True)
    uu = np.bincount(inv, weights=u) / np.bincount(inv)
    yr = norm.ppf(np.clip(interp_linear_extrap(y, yu, uu), 1e-6, 1 - 1e-6))
    global TRAIN_MEAN_YR
    TRAIN_MEAN_YR = float(yr[TRAIN].mean())
    return dict(T=T, X=X, yr=yr, yb=yb, yo=yo, cut=cut, pred_names=pred_names, y_raw=y)


# ---------------------------------------------------------------- metrics
def r2(y, p, ref=None):
    """Out-of-sample R^2 relative to the training-set mean of the target (ref overrides it)."""
    m = TRAIN_MEAN_YR if ref is None else ref
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - m) ** 2)


def r2_evalmean(y, p):
    """R^2 relative to the mean of the evaluated set (what the MATLAB script computes)."""
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def spearman(y, p):
    """Spearman rho; defined as 0 for a constant prediction (undefined correlation)."""
    from scipy.stats import spearmanr
    if np.ptp(p) == 0:
        return 0.0
    return float(spearmanr(y, p).statistic)


def ap(y, s):
    """Average precision with tied scores handled by threshold (sklearn)."""
    return float(average_precision_score(y, s))


def auc(y, s):
    return float(roc_auc_score(y, s))


def brier(y, s):
    return float(brier_score_loss(y, s))


def binary_at(y, s, tau):
    p = (s >= tau).astype(int)
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(accuracy=float(accuracy_score(y, p)), precision=prec, recall=rec, f1=f1,
                balanced_accuracy=float(balanced_accuracy_score(y, p)),
                mcc=float(matthews_corrcoef(y, p)) if len(np.unique(p)) > 1 else 0.0,
                n_flagged=int(p.sum()))


def macro_f1(y, p):
    """Same as matlab macroF1(): mean over classes 1..3 of 2TP/(n_true+n_pred), 0 if empty."""
    return float(f1_score(y, p, labels=[1, 2, 3], average="macro", zero_division=0))


def acc(y, p):
    return float(accuracy_score(y, p))
