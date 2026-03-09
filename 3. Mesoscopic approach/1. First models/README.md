# First models

The script `main.py` orchestrates model evaluation and optional metrics/plots generation.  
It wraps around two other scripts:

- `evaluate.py`: runs model evaluations  
- `metrics_and_plots.py`: generates metrics and plots for each model, feature set, and interval  

---

## Usage

Run with the following arguments:

- `--models`: list of models to evaluate (e.g. `KNN_50 RF XGB RF_CLS TCN`)  
- `--features`: list of feature sets to use (e.g. `features_all features_corr_var features_RFE`)  
- `--intervals`: one or more split intervals (floats; e.g. `5.0 2.0`)  
- `--metrics`: whether to also generate metrics and plots (`True` or `False`)  

Example:

```bash
python main.py --models KNN_50 RF XGB --features features_all features_corr_var --intervals 5.0 2.0 --metrics True
```

---

## Behavior

1. Runs `evaluate.py` for each interval.  
2. If `--metrics` is set to `True`, also runs `metrics_and_plots.py` for every model and feature combination.  
   - If the feature set is `features_corr_var`, an additional PCA variant (`features_corr_var_PCA40`) is also evaluated.  

The outputs include evaluation results and, if requested, metrics and plots.

---

## Combining Metrics (`combine_metrics.py`)

After running `main.py`, you can aggregate and visualize the results using `combine_metrics.py`.  
This script:

- Collects all metrics CSVs from the `metrics/` folder  
- Produces a combined table of results (`all_metrics_table.txt`)  
- Generates per-metric plots comparing models and feature sets across intervals  

Outputs are stored in `metrics/combined_metrics/`.

---

## Statistical significance tests (`statistical_tests.py`)

After running the main scripts, outputs can be further analyzed by running `statistical_tests.py`. This script is configured to perform Wilcoxon Signed-Rank Tests for MAE and Steiger's Z-tests for Pearson's _r_ to compare the best performing models (TCN and XGBoost at 5-s intervals with all features), but the `.pkl` files that are analyzed can be customized.
