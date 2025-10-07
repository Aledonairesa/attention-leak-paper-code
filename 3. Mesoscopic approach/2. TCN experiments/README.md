# TCN hyperaparameters and architectural experiments

This script runs selected models on selected feature sets.  
It saves **per-split predictions** (`true` vs. `pred`) into `.pkl` files for later analysis.

The models used are defined in `models.py` and correspond to different variations of the base TCN model, changing key hyperparameters and architectural choices.

---

## Usage

Arguments:

- `--models`: list of models to run (default: all models in the registry)  
- `--features`: list of feature sets to use (default: all feature sets)  
- `--interval`: split interval in seconds (e.g. `2.0` or `5.0`)  

Example:

```bash
python save_predictions.py --models TCN_hidden24 TCN_hidden60 --features features_all features_corr_var --interval 5.0
```

---

## Behavior

1. Loads training and test splits.  
2. For each split:  
   - Trains on all other splits (excluding the target split and its neighbors).  
   - Clips outliers, standardizes features, and applies PCA if required.  
   - Saves predictions (`true` and `pred` values) into `.pkl` files under `prediction_pkls/{interval}s/`.  
3. If the feature set is `features_corr_var`, also evaluates the PCA-40 variant (`features_corr_var_PCA40`).  

---

## Outputs

- For each model and feature set: a pickle file containing per-split predictions, named:  
  ```
  prediction_pkls/{interval}s/{model}_{feature}.pkl
  ```
- These `.pkl` files store dictionaries with split IDs as keys and `true`/`pred` lists as values.  

---