# Network-active and network-inactive experiments

This folder provides two scripts to (1) produce predictions with a fixed model/feature/interval setup for target-modified datasets: network-active, network-inactive, as well as the stardad one, and (2) compute quick metrics and a visualization.

---

## 1) Predictions

**Script:** `evaluate_exploration.py`  
**What it does:**  
- Loads features, clips extremes, scales, fits the model, and predicts on the test sequence  
- Saves a compact `.pkl` bundle with predictions and ground-truth targets for target-modified datasets network-active, network-inactive, as well as the stardad one. 

**Defaults you can edit at the top of the file:**  
- Model: `TCN_hidden24`  
- Feature set: `features_all`  
- Interval: `5.0` seconds (reads from `Splits/5.0s/`)  
- Paths: feature list PKLs in `../../Data generation - App/select_features_results`; hold-outs in `../../Data generation - App/holdout_splits.txt`  
- Train/test split counts: `N_TRAIN_SPLITS`, `N_TEST_SPLITS`  

Outputs are written to:
```
holdout_prediction_pkls/{INTERVAL}s/{MODEL_NAME}_{FEATURE_LABEL}_holdout.pkl
```

**Run:**
```bash
python evaluate_exploration.py
```

---

## 2) Compute quick metrics and plot

**Script:** `metrics_exploration.py`  

**Defaults you can edit at the top of the file:**  
- Model/feature/interval (must match step 1)  
- Path to PKL: `holdout_prediction_pkls/{INTERVAL}s/{MODEL}_{FEATURE}_holdout.pkl`  
- Plot slice: `PLOT_START`, `PLOT_END`  

Outputs are written to:
```
metrics_exploration/{INTERVAL}s/{MODEL}_{FEATURE}_holdout_metrics.csv
plots_exploration/{INTERVAL}s/{MODEL}_{FEATURE}_holdout_plot.png
```

**Run:**
```bash
python metrics_exploration.py
```
