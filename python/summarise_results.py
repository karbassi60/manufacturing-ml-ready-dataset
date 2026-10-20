"""Summarise the MATLAB run (matlab/newfinal_upd.m) and recompute metrics from its saved test predictions.

Usage:  python python/summarise_results.py [repo_root]

Reads   matlab_outputs/analysis_results.mat, matlab_outputs/metrics_by_seed.csv, data/ml_ready_full.csv
Writes  results/summary.json, results/nn_metrics_by_seed.csv, results/*.csv, results/Figure_5_*.png, Figure_6_*.png

Why this script exists: (i) it computes the additional test metrics reported in the paper (RMSE, MAE,
Spearman rho, ROC-AUC, Brier score, accuracy/precision/recall/F1/MCC at tau = 0.5, confusion matrix,
Wilcoxon tests, bootstrap intervals); (ii) it recomputes the precision-recall AUC as average precision with
tied scores grouped. The prAuc() function in newfinal_upd.m orders tied scores by row index, so for a network
that outputs one constant probability for every record it returns a value above the prevalence that depends
on the row order (0.328 for three single-task seeds). Figures 5 and 6 are redrawn here for the same reason.
"""
import json, sys, pathlib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
sys.path.insert(0, str(HERE))
import common as C

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
D = C.load(ROOT / "data" / "ml_ready_full.csv")
M = sio.loadmat(ROOT / "matlab_outputs" / "analysis_results.mat", squeeze_me=True)
te = C.TEST
yr, yb, yo = D["yr"][te], D["yb"][te], D["yo"][te]
assert np.allclose(D["cut"], M["cut"]), "tercile cut-points differ from the MATLAB run"
S = M["PRm"].shape[1]

PRm, PBm, POm = M["PRm"], M["PBm"], M["POm"]   # test predictions, multi-task, per seed
PRs, PBs, POs = M["PRs"], M["PBs"], M["POs"]   # single-task networks

def per_seed(PR, PB, PO):
    rows = []
    for k in range(S):
        b05 = C.binary_at(yb, PB[:, k], 0.5)
        pred = PO[:, :, k].argmax(1) + 1
        rows.append(dict(seed=k + 1, r2=C.r2(yr, PR[:, k]), rmse=C.rmse(yr, PR[:, k]), mae=C.mae(yr, PR[:, k]),
                         spearman=C.spearman(yr, PR[:, k]),
                         ap=C.ap(yb, PB[:, k]), roc_auc=C.auc(yb, PB[:, k]), brier=C.brier(yb, PB[:, k]),
                         bin_accuracy_05=b05["accuracy"], bin_precision_05=b05["precision"], bin_recall_05=b05["recall"],
                         bin_f1_05=b05["f1"], bin_mcc_05=b05["mcc"], bin_flagged_05=b05["n_flagged"],
                         prob_sd=float(PB[:, k].std()), reg_pred_sd=float(PR[:, k].std()),
                         ord_accuracy=C.acc(yo, pred), ord_macro_f1=C.macro_f1(yo, pred)))
    return pd.DataFrame(rows)

mt = per_seed(PRm, PBm, POm)
st = per_seed(PRs, PBs, POs)
# compare with the metrics written by MATLAB
mat = pd.read_csv(ROOT / "matlab_outputs" / "metrics_by_seed.csv")
# MATLAB's r2() uses the mean of the test set as reference, this script the training-set mean (Eq. (7) of the paper);
# the two agree to within 5e-4 on the test set.
mt_evalmean = np.array([C.r2_evalmean(yr, PRm[:, k]) for k in range(S)])
st_evalmean = np.array([C.r2_evalmean(yr, PRs[:, k]) for k in range(S)])
assert np.allclose(mat["MTL_R2"], mt_evalmean, atol=1e-6) and np.allclose(mat["STL_R2"], st_evalmean, atol=1e-6), "replication of MATLAB R2 failed"
assert np.abs(mt_evalmean - mt["r2"]).max() < 5e-4 and np.abs(st_evalmean - st["r2"]).max() < 5e-4
assert np.allclose(mat["MTL_MacroF1"], mt["ord_macro_f1"], atol=1e-9) and np.allclose(mat["STL_MacroF1"], st["ord_macro_f1"], atol=1e-9)
mt["matlab_pr_auc"], st["matlab_pr_auc"] = mat["MTL_PR_AUC"], mat["STL_PR_AUC"]
mt.to_csv(OUT / "nn_metrics_by_seed_multitask.csv", index=False)
st.to_csv(OUT / "nn_metrics_by_seed_singletask.csv", index=False)

# ------------------------------------------------ summary and paired tests
metrics = ["r2", "rmse", "mae", "spearman", "ap", "roc_auc", "brier", "bin_accuracy_05", "bin_precision_05", "bin_recall_05",
           "bin_f1_05", "bin_mcc_05", "ord_accuracy", "ord_macro_f1"]
summary = {"n_seeds": S, "test_n": int(te.size), "test_prevalence": float(yb.mean()),
           "test_positives": int(yb.sum()), "train_prevalence": float(D["yb"][C.TRAIN].mean()),
           "val_prevalence": float(D["yb"][C.VAL].mean()),
           "ordinal_test_counts": np.bincount(yo, minlength=4)[1:].tolist(), "mtl": {}, "stl": {}, "wilcoxon": {}}
for m in metrics:
    summary["mtl"][m] = [float(mt[m].mean()), float(mt[m].std(ddof=1))]
    summary["stl"][m] = [float(st[m].mean()), float(st[m].std(ddof=1))]
    d = mt[m].to_numpy() - st[m].to_numpy()
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(d, alternative="two-sided", method="auto").pvalue)
    summary["wilcoxon"][m] = dict(p=p, mtl_ahead=int((d > 0).sum()), stl_ahead=int((d < 0).sum()), mean_diff=float(d.mean()))
good = [k for k in range(S) if PRs[:, k].std() >= 1e-3]
summary["stl_regression_excluding_constant_seeds"] = dict(n=len(good), r2_mean=float(st["r2"][good].mean()), r2_sd=float(st["r2"][good].std(ddof=1)),
                                                          spearman_mean=float(st["spearman"][good].mean()))
summary["cv"] = {a: {m: float(summary[a][m][1] / abs(summary[a][m][0])) for m in ["r2", "ap", "ord_macro_f1", "roc_auc", "ord_accuracy"]} for a in ["mtl", "stl"]}
summary["constant_binary_seeds"] = {"mtl": [int(k + 1) for k in range(S) if PBm[:, k].std() < 1e-6],
                                    "stl": [int(k + 1) for k in range(S) if PBs[:, k].std() < 1e-6]}
summary["binary_prob_sd_range"] = {"mtl": [float(mt.prob_sd.min()), float(mt.prob_sd.max())],
                                   "stl": [float(st.prob_sd.min()), float(st.prob_sd.max())]}
summary["constant_regression_seeds"] = {"mtl": [int(k + 1) for k in range(S) if PRm[:, k].std() < 1e-3],
                                        "stl": [int(k + 1) for k in range(S) if PRs[:, k].std() < 1e-3]}
summary["matlab_prauc_vs_ap_differs_seeds"] = {
    "mtl": [int(k + 1) for k in range(S) if abs(mt.matlab_pr_auc[k] - mt.ap[k]) > 1e-9],
    "stl": [int(k + 1) for k in range(S) if abs(st.matlab_pr_auc[k] - st.ap[k]) > 1e-9]}

# ------------------------------------------------ seed-averaged predictions and bootstrap
def avg(PR, PB, PO):
    return PR.mean(1), PB.mean(1), PO.mean(2)

aR_m, aB_m, aO_m = avg(PRm, PBm, POm)
aR_s, aB_s, aO_s = avg(PRs, PBs, POs)
rng = np.random.default_rng(20261020)
B = 2000
boot = {k: [] for k in ["r2_m", "r2_s", "sp_m", "sp_s", "ap_m", "ap_s", "auc_m", "auc_s", "f1_m", "f1_s", "acc_m", "acc_s"]}
for _ in range(B):
    i = rng.integers(0, te.size, te.size)
    boot["r2_m"].append(C.r2(yr[i], aR_m[i])); boot["r2_s"].append(C.r2(yr[i], aR_s[i]))
    boot["sp_m"].append(C.spearman(yr[i], aR_m[i])); boot["sp_s"].append(C.spearman(yr[i], aR_s[i]))
    if 0 < yb[i].sum() < i.size:
        boot["ap_m"].append(C.ap(yb[i], aB_m[i])); boot["ap_s"].append(C.ap(yb[i], aB_s[i]))
        boot["auc_m"].append(C.auc(yb[i], aB_m[i])); boot["auc_s"].append(C.auc(yb[i], aB_s[i]))
    else:
        for k in ["ap_m", "ap_s", "auc_m", "auc_s"]:
            boot[k].append(np.nan)
    boot["f1_m"].append(C.macro_f1(yo[i], aO_m[i].argmax(1) + 1)); boot["f1_s"].append(C.macro_f1(yo[i], aO_s[i].argmax(1) + 1))
    boot["acc_m"].append(C.acc(yo[i], aO_m[i].argmax(1) + 1)); boot["acc_s"].append(C.acc(yo[i], aO_s[i].argmax(1) + 1))
boot = {k: np.array(v) for k, v in boot.items()}
def ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
summary["seed_averaged"] = {
    "mtl": dict(r2=C.r2(yr, aR_m), rmse=C.rmse(yr, aR_m), mae=C.mae(yr, aR_m), spearman=C.spearman(yr, aR_m), ap=C.ap(yb, aB_m), roc_auc=C.auc(yb, aB_m),
                ord_accuracy=C.acc(yo, aO_m.argmax(1) + 1), ord_macro_f1=C.macro_f1(yo, aO_m.argmax(1) + 1)),
    "stl": dict(r2=C.r2(yr, aR_s), rmse=C.rmse(yr, aR_s), mae=C.mae(yr, aR_s), spearman=C.spearman(yr, aR_s), ap=C.ap(yb, aB_s), roc_auc=C.auc(yb, aB_s),
                ord_accuracy=C.acc(yo, aO_s.argmax(1) + 1), ord_macro_f1=C.macro_f1(yo, aO_s.argmax(1) + 1))}
summary["bootstrap_ci"] = {
    "mtl": dict(r2=ci(boot["r2_m"]), spearman=ci(boot["sp_m"]), ap=ci(boot["ap_m"]), roc_auc=ci(boot["auc_m"]),
                ord_macro_f1=ci(boot["f1_m"]), ord_accuracy=ci(boot["acc_m"])),
    "stl": dict(r2=ci(boot["r2_s"]), spearman=ci(boot["sp_s"]), ap=ci(boot["ap_s"]), roc_auc=ci(boot["auc_s"]),
                ord_macro_f1=ci(boot["f1_s"]), ord_accuracy=ci(boot["acc_s"])),
    "diff_mtl_minus_stl": dict(r2=ci(boot["r2_m"] - boot["r2_s"]), spearman=ci(boot["sp_m"] - boot["sp_s"]),
                               ap=ci(boot["ap_m"] - boot["ap_s"]), roc_auc=ci(boot["auc_m"] - boot["auc_s"]),
                               ord_macro_f1=ci(boot["f1_m"] - boot["f1_s"]), ord_accuracy=ci(boot["acc_m"] - boot["acc_s"]))}
summary["bootstrap_resamples"] = B
summary["r2_reference_mean_check"] = dict(
    max_abs_diff_mtl=float(np.abs(mt_evalmean - mt["r2"]).max()), max_abs_diff_stl=float(np.abs(st_evalmean - st["r2"]).max()),
    train_mean=float(C.TRAIN_MEAN_YR), test_mean=float(yr.mean()))

# ------------------------------------------------ confusion matrix (seed-averaged multi-task probabilities)
pred = aO_m.argmax(1) + 1
cm = np.zeros((3, 3), int)
for a, p in zip(yo, pred):
    cm[a - 1, p - 1] += 1
summary["confusion_mtl_seed_avg"] = cm.tolist()
summary["adjacent_errors"] = {"misclassified": int((pred != yo).sum()), "non_adjacent": int((abs(pred - yo) > 1).sum())}
predS = aO_s.argmax(1) + 1
cmS = np.zeros((3, 3), int)
for a, p in zip(yo, predS):
    cmS[a - 1, p - 1] += 1
summary["confusion_stl_seed_avg"] = cmS.tolist()

# ------------------------------------------------ reliability bins (equal count, as in the MATLAB figure)
order = np.argsort(aB_m, kind="stable")
parts = np.round(np.linspace(0, te.size, 6)).astype(int)
summary["reliability_mtl"] = [dict(mean_pred=float(aB_m[order[parts[i]:parts[i + 1]]].mean()),
                                   observed=float(yb[order[parts[i]:parts[i + 1]]].mean())) for i in range(5)]
q = np.percentile(aB_m, [5, 95])
summary["mtl_seed_avg_prob_5_95pct"] = [float(q[0]), float(q[1])]

# ------------------------------------------------ tuning surface, sensitivity grids
L = M["landscape"]
lr, l2 = M["learningRates"], M["l2Grid"]
iR, iL = np.unravel_index(np.argmin(L), L.shape)
summary["tuning"] = dict(selected_lr=float(lr[iL]), selected_l2=float(l2[iR]), min_val_loss=float(L[iR, iL]),
                         max_val_loss=float(L.max()), n_cells=int(L.size),
                         lr_grid=lr.tolist(), l2_grid=l2.tolist())
pd.DataFrame(L, index=[f"lambda={x:g}" for x in l2], columns=[f"lr={x:.4g}" for x in lr]).to_csv(OUT / "tuning_validation_loss_grid.csv")
A, Bm = M["A"], M["B"]
pd.DataFrame(A, index=[f"noise={x:g}" for x in M["noiseLevels"]], columns=[f"share={x:g}" for x in M["fractions"]]).to_csv(OUT / "sensitivity_test_R2.csv")
pd.DataFrame(Bm, index=[f"noise={x:g}" for x in M["noiseLevels"]], columns=[f"share={x:g}" for x in M["fractions"]]).to_csv(OUT / "sensitivity_test_macroF1.csv")
summary["sensitivity"] = dict(noise=M["noiseLevels"].tolist(), share=M["fractions"].tolist(), r2=A.tolist(), macro_f1=Bm.tolist())
# intercept-only joint validation loss (reference for the right-hand plateau of the tuning surface)
va = C.VAL
reg0 = float(np.mean(D["yr"][va] ** 2))
p1 = float(D["yb"][C.TRAIN].mean())
bce = lambda y, p: float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
bin0 = bce(D["yb"][va], p1)
c1, c2 = float((D["yo"][C.TRAIN] > 1).mean()), float((D["yo"][C.TRAIN] > 2).mean())
ord0 = bce((D["yo"][va] > 1).astype(float), c1) + bce((D["yo"][va] > 2).astype(float), c2)
summary["intercept_only_val_loss"] = dict(regression=reg0, binary=bin0, ordinal=ord0, joint=reg0 + bin0 + ord0)

json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

# ------------------------------------------------ Figures 5 and 6 (redrawn with tied scores handled)
BLUE, ORANGE = "#0072BD", "#D95319"
def pr_curve(y, s):
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y, s)
    return r, p

fig, ax = plt.subplots(1, 2, figsize=(11, 4.7))
for s, lab, col in [(aB_m, "Multi-task", BLUE), (aB_s, "Single-task", ORANGE)]:
    r, p = pr_curve(yb, s)
    ax[0].step(r, p, where="post", lw=1.7, color=col, label=lab)
ax[0].axhline(yb.mean(), color="k", ls="--", lw=1, label="No-skill")
ax[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="A. Test precision-recall (seed-averaged)")
ax[0].grid(alpha=.3); ax[0].legend(loc="upper right")
rel = summary["reliability_mtl"]
ax[1].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].plot([x["mean_pred"] for x in rel], [x["observed"] for x in rel], "o-", color=BLUE, lw=1.7, label="Multi-task")
ax[1].set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", xlim=(0, 1), ylim=(0, 1),
          title="B. Test reliability, multi-task (equal-count bins)")
ax[1].grid(alpha=.3); ax[1].legend(loc="upper left")
fig.suptitle("Figure 5 | Binary department classification")
fig.tight_layout(); fig.savefig(OUT / "Figure_5_Binary_PR_reliability.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.3))
lab = ["Production R$^2$", "Binary PR-AUC (average precision)", "Efficiency macro-F1"]
cols = ["r2", "ap", "ord_macro_f1"]
for j in range(3):
    a = ax[j]
    for k in range(S):
        a.plot([1, 2], [mt[cols[j]][k], st[cols[j]][k]], "-", color=(.7, .7, .7), lw=1)
    a.scatter(np.ones(S), mt[cols[j]], s=22, color=BLUE, zorder=3)
    a.scatter(2 * np.ones(S), st[cols[j]], s=22, color=ORANGE, zorder=3)
    a.errorbar([1, 2], [mt[cols[j]].mean(), st[cols[j]].mean()], [mt[cols[j]].std(ddof=1), st[cols[j]].std(ddof=1)],
               fmt="ks", capsize=4, lw=1.5, mfc="none")
    if cols[j] == "ap":
        a.axhline(yb.mean(), color="k", ls="--", lw=1)
    a.set(xlim=(.7, 2.3), ylabel=lab[j], title=f"{lab[j]}; n={S}")
    a.set_xticks([1, 2]); a.set_xticklabels(["MTL", "STL"]); a.grid(alpha=.3)
fig.suptitle("Figure 6 | Stability across random initializations")
fig.tight_layout(); fig.savefig(OUT / "Figure_6_Seed_stability.png", dpi=200); plt.close(fig)
print("summary written to", OUT / "summary.json")
