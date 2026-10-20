# Data dictionary and provenance notes for `data/ml_ready_full.csv`

1,000 records, 39 columns, no missing values, no duplicated rows. SHA-256 `94588e1be0867d534aada8f39e636c11e3becb36866c59521f0a6afb3e8523e7`.

The numerical columns are standardized in the file (the file gives no physical units). The file has no date column; 
`DayOfYear` and `Month` are stored values, and `Weekday` is a calendar-month index (Spearman correlation with `DayOfYear` 0.996), not a day of the week. 
The row order of the file is not chronological.

The table below lists every column (one-hot columns are grouped by variable), its association with the three targets, its relation to other columns, and whether the reported analysis uses it as a predictor. 
Definitions of `Efficiency_Level` and `Maintenance_Event` are not documented and we did not find the two columns in the Kaggle file; they are not used by the reported analysis. 
The values come from `python/column_audit.py` (`results/column_audit.csv`, Supplementary Table S2); see there for the meaning of each association measure.

| Column | Columns in file | Type | Production volume | Department_Maintenance | Machine_Efficiency | Relation | Used as predictor |
|---|---|---|---|---|---|---|---|
| Production_Hours | 1 | numerical | 0.03 | 0.02 | 0.81 | enters the construction of Production_Volume_Per_Hour (Section 4.4) | no (source of the regression target) |
| Workforce_Size | 1 | numerical | 0.14 | 0.03 | 0.82 |  | yes |
| Machine_Efficiency | 1 | numerical | 0.16 | -0.00 | 1.00 |  | no (source of the ordinal target) |
| Raw_Materials_Cost | 1 | numerical | 0.18 | 0.00 | 0.87 |  | yes |
| Shift_Hours | 1 | numerical | 0.19 | 0.00 | 0.82 |  | yes |
| Production_Volume_Per_Hour | 1 | numerical | 1.00 | 0.02 | -0.00 |  | no (regression target) |
| Timestamp_Hour | 1 | numerical | -0.04 | -0.02 | -0.04 | hour of the day (0–23) | yes |
| DayOfYear (2 copies) | 2 | numerical | -0.02 | -0.00 | -0.01 | stored twice; Month is identical | no (duplicate) |
| Month (2 copies) | 2 | numerical | -0.02 | -0.00 | -0.01 | identical to DayOfYear | no (duplicate) |
| Weekday (2 copies) | 2 | numerical | -0.01 | -0.00 | -0.01 | 12 distinct values; ρ = 0.996 with DayOfYear, i.e. a calendar-month index; second copy not used | yes (first copy; calendar index) |
| Supplier | 10 | categorical (10 levels) | η² = 0.013 | V = 0.13 | η² = 0.01 |  | yes (10 indicators) |
| Department | 4 | categorical (4 levels) | η² = 0.001 | V = 1.00 | η² = 0.00 | defines Department_Maintenance | no (source of the binary target) |
| Equipment_Age_Category | 3 | categorical (3 levels) | η² = 0.050 | V = 0.01 | η² = 0.79 | recoding of Machine_Efficiency (cut points ≈ −0.43 and 0.45; agrees with the regime for 961 records) | no (leakage) |
| Efficiency_Level | 1 | categorical (3 levels) | η² = 0.049 | V = 0.01 | η² = 0.79 | recoding of Machine_Efficiency (cut points ≈ −0.42 and 0.47; agrees with the regime for 949 records); not in the Kaggle file | no (leakage) |
| Shift_Category | 3 | categorical (3 levels) | η² = 0.070 | V = 0.03 | η² = 0.54 | recoding of Shift_Hours (cut points ≈ −1.09 and 0.79) | no (redundant) |
| Supply_Chain_Resilience | 4 | categorical (3 levels) | η² = 0.101 | V = 0.02 | η² = 0.55 | recoding of Raw_Materials_Cost (cut points ≈ −1.08 and 0.95); also stored as text | no (redundant) |
| Maintenance_Event | 1 | categorical (2 levels) | η² = 0.053 | V = 0.01 | η² = 0.37 | undocumented; not in the Kaggle file; strongly related to the operating variables (Section 4.6) | no (definition and timing not established) |

Targets: regression, `Production_Volume_Per_Hour` (training-only rank-normal scores); binary, `Department_Maintenance`; ordinal, terciles of `Machine_Efficiency` computed on the training block.
