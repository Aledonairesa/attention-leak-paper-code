# Mitigation experiments

This folder provides two scripts:
- `evaluate.py`: trains on normal splits; saves per-split predictions for normal and/or injected test variants.
- `metrics_and_plots.py`: loads those predictions; writes metrics and scatter/residual plots.

---

## 1) Evaluate: produce prediction PKLs

**What it does:** trains once per split on *normal* data; tests on one or more targets:
- normal test (`Splits/{interval}s/`)
- injected tests (`User_Injected_<label>/Splits/{interval}s/`)

**Example (injected variants 100 and 200):**
```bash
python evaluate.py --models TCN \
  --features features_corr_var \
  --interval 5.0 \
  --injected 100 200
```

**Output:** `prediction_pkls/{interval}s/<MODEL>_<FEATURE>[ _inj<label> ].pkl`

---

## 2) Metrics and plots

**What it does:** for each available PKL, computes MAE, MBE, PearsonR, RMSE, R2, ResidualCorr and generates scatter and residual plots. Can compare normal and injected variants.

**Example (compare normal plus injected 100, 200):**
```bash
python metrics_and_plots.py --model TCN \
  --features features_corr_var \
  --interval 5.0 \
  --injected 100 200 \
  --include_normal
```

**Outputs:**
```
plots/{interval}s/<variant>/                # scatter + residual PNGs
metrics/{interval}s/<variant>/<...>.csv     # metrics per variant
```
