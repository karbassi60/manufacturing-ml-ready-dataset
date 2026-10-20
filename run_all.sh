#!/usr/bin/env bash
# Reproduces every number, table and data-based figure of manuscript OMEGA-D-26-00675.R1.
#   Stage 1 (MATLAB): the neural networks, tuning, permutation importance, sensitivity grid; Figures 2, 3, 4, 7, 8.
#   Stage 2 (Python): baselines, imbalance analysis, bootstrap and tests, data checks, summary; Figures 1, 5, 6.
# Usage:   bash run_all.sh            (both stages; needs MATLAB on the PATH for stage 1)
#          SKIP_MATLAB=1 bash run_all.sh   (Python stage only, using the saved MATLAB outputs in matlab_outputs/)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p results

if [[ "${SKIP_MATLAB:-0}" != "1" ]]; then
  command -v matlab >/dev/null || { echo "matlab not found on the PATH; run matlab/newfinal_upd.m by hand (see README.md) or use SKIP_MATLAB=1"; exit 1; }
  # newfinal_upd.m looks for ml_ready_full.csv next to itself and writes to matlab/matlab_analysis_figures/
  cp data/ml_ready_full.csv matlab/ml_ready_full.csv
  (cd matlab && matlab -batch "run('newfinal_upd.m')")
  rm -f matlab/ml_ready_full.csv
  mkdir -p matlab_outputs
  cp -f matlab/matlab_analysis_figures/* matlab_outputs/
fi

python python/data_facts.py "$ROOT"
python python/summarise_results.py "$ROOT"
python python/baselines.py "$ROOT"
python python/imbalance.py "$ROOT"
python python/compare_models.py "$ROOT"
python python/column_audit.py "$ROOT"
python python/original_submission_check.py "$ROOT"
python python/draw_figure1.py "$ROOT/results/Figure_1_Conceptual_framework.png"
echo "Done. Results in results/ (JSON and CSV) and figures in matlab_outputs/ and results/."
