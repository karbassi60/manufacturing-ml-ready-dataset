# Code and data for OMEGA-D-26-00675.R1

Analysis code and data file behind the revised manuscript "A Data-Enabled Multi-Task Learning Framework for
Production Volume Estimation and Operational Efficiency Classification" (Omega, manuscript OMEGA-D-26-00675.R1).

Version 2.0.0. This version replaces the demonstration scripts of version 1.0.0, which were not the code of the paper
and have been removed. Every table and every data-based figure of the revised manuscript is produced by the code in this
repository from the file `data/ml_ready_full.csv`. The original submission is **not** reproduced by this repository (see
"Original submission" below).

## Contents

| Path | What it is |
|---|---|
| `data/ml_ready_full.csv` | The analysis file: 1,000 records, 39 columns. SHA-256 `94588e1be0867d534aada8f39e636c11e3becb36866c59521f0a6afb3e8523e7`. |
| `matlab/newfinal_upd.m` | The MATLAB script for the neural networks (multi-task and single-task), the tuning grid, permutation importance, the sensitivity grid and Figures 2, 3, 4, 7 and 8. SHA-256 `78af6d879073d69f43b0077054b5d138e9b64439ecdb20ce3eddfa2fe1855cfb`. |
| `matlab_outputs/` | Saved outputs of the run reported in the manuscript: `analysis_results.mat` (predictions of all 20 networks, metrics, tuning grid, importance, sensitivity grid), `metrics_by_seed.csv`, `validation_loss_grid.csv`, and the MATLAB figures. |
| `python/` | Baselines (ridge/logistic regression, Random Forest, XGBoost, LightGBM), imbalance analysis, bootstrap intervals and tests, data checks, the summary of all reported numbers, and Figures 1, 5 and 6. |
| `results/` | Output of the Python stage: `summary.json` (all neural-network numbers), `baselines_summary.json`, `imbalance_summary.json`, `model_comparison.json` (bootstrap and Wilcoxon results), `data_facts.json`, `column_audit.csv` (Supplementary Table S2), `original_submission_check.json` (Supplementary Table S3), per-seed CSV files and Figures 5 and 6. |
| `run_all.sh` | Runs both stages. |
| `requirements.txt` | Python package versions used. |
| `docs/` | Data dictionary and provenance notes. |

## How to reproduce

### 1. Neural networks (MATLAB)

Requires MATLAB R2022b or later with the Deep Learning Toolbox and the Statistics and Machine Learning Toolbox
(the release used for the reported run was not recorded). The script reads `ml_ready_full.csv` from its own folder and
writes its outputs to `matlab/matlab_analysis_figures/`. It is a full run of 10 seeds, 72 tuning combinations with 3
seeds each, 20 permutations per predictor group and the sensitivity grid, and it is computationally intensive.

```
cp data/ml_ready_full.csv matlab/ml_ready_full.csv
cd matlab && matlab -batch "run('newfinal_upd.m')"     # or open and run the script in the MATLAB desktop
cd .. && cp matlab/matlab_analysis_figures/* matlab_outputs/ && rm matlab/ml_ready_full.csv
```

The script sets `rng(42)` at the start and `rng(seed)` for each network. Different MATLAB releases or hardware
(CPU vs. GPU, number of threads) can change single-precision results in the last digits and, through early stopping,
occasionally the epoch at which training stops; a rerun may therefore differ slightly from the saved outputs. The script contains an optional search for the file on an external drive (`/Volumes/Untitled 2/...`) that
is not used when the file is next to the script.

### 2. Everything else (Python)

```
pip install -r requirements.txt
SKIP_MATLAB=1 bash run_all.sh          # uses the saved MATLAB outputs in matlab_outputs/
```

`bash run_all.sh` without `SKIP_MATLAB` runs stage 1 first (MATLAB must be on the PATH). The Python stage takes a
few minutes on a laptop. Scripts accept the repository root as their first argument and run in this order:

1. `data_facts.py` – facts quoted in Section 4 (counts, correlations, ordering of the rows, the Equipment_Age_Category check).
2. `summarise_results.py` – reads `matlab_outputs/analysis_results.mat` and `metrics_by_seed.csv` and recomputes every neural-network metric from the saved predictions (R² with the training mean, average precision, macro-F1, Spearman correlation, Brier score, MCC, calibration), the multi-task vs. single-task comparison, Figures 5 and 6.
3. `baselines.py` – tuning on the validation block (seeds 1–3), ten final fits, test metrics.
4. `imbalance.py` – no rebalancing / SMOTE / class weights with a validation-selected threshold.
5. `compare_models.py` – 2,000 bootstrap resamples of the test block (seed 20261020) and Wilcoxon tests.
6. `column_audit.py` – audit of all 39 columns (Supplementary Table S2).
7. `original_submission_check.py` – consistency check of two statements about the original submission (Supplementary Table S3).
8. `draw_figure1.py` – Figure 1 (conceptual framework).

### Where each table and figure of the manuscript comes from

| Manuscript | Source |
|---|---|
| Table 1 | Literature comparison (no computation) |
| Table 2 | Dataset and set-up; the full configuration is in Supplementary Table S1 |
| Tables 3–8 (neural networks: production, binary, efficiency, confusion matrix, class distribution, primary metric by task) | `results/summary.json` and `results/data_facts.json`, computed from `matlab_outputs/analysis_results.mat` |
| Table 9 (imbalance strategies) | `results/imbalance_summary.json` |
| Table 10 (single-task networks and baselines) | `results/baselines_summary.json`, `results/model_comparison.json`, `results/summary.json` |
| Figure 1 | `python/draw_figure1.py` |
| Figures 2, 3, 4, 7, 8 | `matlab_outputs/` (MATLAB) |
| Figures 5, 6 | `results/` (`python/summarise_results.py`) |

## The raw Kaggle data

The revised paper analyses `data/ml_ready_full.csv`, not the raw Kaggle download: the original analysis used this
preprocessed derivative (39 columns, categorical variables one-hot encoded, numerical columns standardized), its preparation
steps were not recorded, and a different preprocessing would give a different analysis. The raw data are not included.
To obtain them: open https://www.kaggle.com/datasets/kingsleyokonkwocom/dynamic-manaufacturing-data (the spelling
"manaufacturing" is the one used by Kaggle), sign in, and either press "Download" or run

```
pip install kaggle        # needs an API token in ~/.kaggle/kaggle.json
kaggle datasets download -d kingsleyokonkwocom/dynamic-manaufacturing-data
```

The download has 1,000 rows and 21 columns. It lacks `Efficiency_Level` and `Maintenance_Event` and the one-hot columns of
the preprocessed file, and it cannot be fed to the scripts of this repository without the undocumented preparation steps.

## Things to know before relying on the numbers

* **Data provenance.** `ml_ready_full.csv` is a preprocessed version, prepared by one of the authors for the original
  submission, of the Kaggle dataset "Dynamic Manufacturing Data" (KingsleyOkonkwo.com, 2023;
  https://www.kaggle.com/datasets/kingsleyokonkwocom/dynamic-manaufacturing-data). The preparation steps were not
  recorded. The file contains two columns that we did not find in the Kaggle file (`Efficiency_Level`,
  `Maintenance_Event`); neither is used by the reported analysis. The file is deposited here because the analysis
  cannot be reproduced without it. The structure of the file (see `docs/DATA_DICTIONARY.md`) indicates that the
  data were generated synthetically, so results should not be read as evidence about real production systems.
* **Row order.** The split is in file order (records 1–550 training, 551–700 validation, 701–800 reserved and unused,
  801–1000 test). The file has no dates and its row order is not chronological; the split is a split of exchangeable
  records and says nothing about forecasting.
* **PR-AUC.** The MATLAB function `prAuc` in `newfinal_upd.m` handles tied scores one at a time and differs from
  average precision for 7 of the 20 networks (multi-task seed 1; single-task seeds 1, 3, 4, 6, 7, 8). The values in the
  manuscript are average precision (scikit-learn), recomputed from the saved test-set predictions by
  `python/summarise_results.py`. The MATLAB values are kept in `matlab_outputs/metrics_by_seed.csv`. Figure 3
  (permutation importance) still uses the MATLAB `prAuc` on the validation block.
* **R².** The MATLAB function `r2` uses the mean of the evaluated block; the manuscript (Eq. (7)) uses the training
  mean. The two differ by at most 0.00015 over the 20 networks.
* **Figure 5 and 6** of the manuscript are redrawn in Python from the saved predictions; the MATLAB versions of these
  figures in `matlab_outputs/` are not used.
* **Values read from images.** Figures 3 and 7 are plotted by MATLAB without exporting the plotted values; the
  numbers quoted from them in the text were read from the images and are approximate.
* **Not recorded by the script:** the number of active hidden units, the predictions on the training block, and the
  MATLAB release used for the reported run.
* **Single-task networks** use the learning rate and regularization strength selected for the multi-task network and
  were not tuned separately (Section 7.2 of the manuscript).

## Original submission

The results of the original submission were not produced by this code, and its script is not part of this
repository. `python/original_submission_check.py` is a consistency check, not a replication: it reproduces the size of
two reported numbers (the efficiency accuracy of 98.5%, caused by leakage through `Equipment_Age_Category`, and the R²
of 0.7075 obtained when the regression target is `Production_Hours`) with simple models on random 80/20 splits of
this file.

## Software

Python 3.13, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0,
imbalanced-learn 0.14.2, Matplotlib 3.11.2 (see `requirements.txt`). MATLAB R2022b or later (see above).

## Licence

Code: Apache License 2.0 (file `LICENSE` of the repository). Data: `data/ml_ready_full.csv` is derived from a dataset
published on Kaggle; the rights to the original data are those stated by its publisher on the Kaggle page, and the
authors have not established separate rights to the derived file. Please check the Kaggle terms before redistributing it.

## How to cite

See `CITATION.cff`. Each GitHub release of this repository is archived on Zenodo; the DOI of the archived release of
version 2.0.0 is: *to be added after the release has been archived*.
