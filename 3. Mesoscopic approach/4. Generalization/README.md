# Cross-user experiments

This folder contains three small scripts to run general train/test experiments across users.

---

## 1) Create per-user train/test groups

**Script:** `create_train_test_groups.py`  
**Purpose:** create a simple 70/30 (configurable) partition of split indices for a given user; use it to define user-specific train/test ranges.

**Example:**
```bash
python create_train_test_groups.py --total 50 --ratio 0.7 --seed 42 --output train_test_groups.txt
```

This writes something like:
```
70: 1-3,5-24,26,28-35
30: 4,25,27,36-50
```

---

## 2) Train once, test across users/splits

**Script:** `evaluate_general.py`  
**Purpose:** train a model on the union of the specified train user/splits, then predict each specified test user/split; saves per-split `true`/`pred` vectors to a timestamped experiment folder. Format for specs: `User:1-3,5-10` or `User:all`.

**Example:**
```bash
python evaluate_general.py \
  --train "Alice:1-3,5-10" "Bob:3-10" \
  --test  "Charlie:2-8" \
  --model TCN \
  --features features_corr_var \
  --interval 5.0 \
  --target num_tasks
```

**Output:** `experiments/exp_<timestamp>/<model>_<features>.pkl` plus `args_used.txt`.

---

## 3) Make plots and metrics from an experiment

**Script:** `metrics_plots_general.py`  
**Purpose:** given an experiment (timestamp or path), load each stored predictions `.pkl`, create a scatter and residual plot, and write a consolidated `metrics_summary.csv` (MAE, MBE, PearsonR, RMSE, R2, ResidualCorr). Artefacts are saved next to the `.pkl`.

**Example (timestamp only):**
```bash
python metrics_plots_general.py --exp 20250528_104233
```

**Example (full path):**
```bash
python metrics_plots_general.py --exp experiments/exp_20250528_104233
```
