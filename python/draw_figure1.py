"""Redraws Fig. 1 (conceptual framework) of the manuscript. Usage: python python/draw_figure1.py [out.png]"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1] if len(sys.argv) > 1 else "Figure_1_Conceptual_framework.png"
plt.rcParams["font.family"] = "DejaVu Serif"
fig = plt.figure(figsize=(6.2, 5.26), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 124); ax.set_ylim(0, 105); ax.axis("off")
COL = {"blue": "#e3eaf4", "orange": "#f6e9dc", "pink": "#f7e2e2", "green": "#e2ece6", "purple": "#ebe6f3"}

def box(x, y, w, h, title, lines, fc, ts=9.2, ls=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=COL[fc], ec="#444", lw=1.4))
    ax.text(x + w / 2, y + h - 3.3, title, ha="center", va="center", fontsize=ts, fontweight="bold")
    n = len(lines)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, y + h - 7.4 - i * 3.1, s, ha="center", va="center", fontsize=ls)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5, mutation_scale=9))

box(19.5, 90, 85, 12.5, "Data and target construction",
    ["Preprocessed Kaggle “Dynamic Manufacturing Data” (ml_ready_full.csv, 1,000 records);",
     "targets defined; target-derived columns excluded (Sec. 4.6)"], "blue")
box(19.5, 72.5, 85, 12.5, "Split and preprocessing",
    ["train / validation / reserved (not used) / test = 550 / 150 / 100 / 200",
     "in file order; training-block standardisation"], "blue")
box(19.5, 54.5, 85, 12.5, "Shared encoder",
    ["15 inputs → 2 fully connected ReLU layers (64, 32 units)"], "orange")
box(5, 36, 35, 12.8, "Regression head", ["production volume per hour", "(normal scores), MSE"], "pink", ts=8.8)
box(44.5, 36, 35, 12.8, "Binary head", ["maintenance-department", "indicator, BCE"], "pink", ts=8.8)
box(84, 36, 35, 12.8, "Ordinal head", ["efficiency regime,", "cumulative link"], "pink", ts=8.8)
box(8, 17.5, 108, 13.2, "Evaluation",
    ["single-task networks, linear models, Random Forest, XGBoost, LightGBM, naive baselines;",
     "imbalance handling; 10 seeds; bootstrap confidence intervals"], "green")
box(8, 0.8, 108, 12.4, "Robustness and diagnostics",
    ["permutation importance, tuning landscape, seed stability, training dynamics,",
     "training size × input noise sensitivity"], "purple")
arrow(62, 90, 62, 85.2); arrow(62, 72.5, 62, 67.2)
arrow(40, 54.5, 24, 49); arrow(62, 54.5, 62, 49); arrow(84, 54.5, 100, 49)
arrow(22.5, 36, 36, 31); arrow(62, 36, 62, 31); arrow(101.5, 36, 88, 31)
arrow(62, 17.5, 62, 12.9)
fig.savefig(out, dpi=200, facecolor="white")
print("saved", out)
