# ---------------------------------------------------------------------
# Std-lib & deps
# ---------------------------------------------------------------------
from pathlib import Path
import pickle
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------
# Config – edit if you saved to a different place / name
# ---------------------------------------------------------------------
MODEL_NAME    = "TCN_hidden24"
FEATURE_LABEL = "features_all"
INTERVAL      = 5.0                      # seconds
INTERVAL_STR  = f"{INTERVAL}s"

# Where to find the prediction pickle
PKL_PATH = (Path("holdout_prediction_pkls") /
            INTERVAL_STR /
            f"{MODEL_NAME}_{FEATURE_LABEL}_holdout.pkl")

# Plot range (slice of the sequence indices) – set start/end (inclusive start, exclusive end)
PLOT_START = 0
PLOT_END   = 1000

# Output directories
PLOTS_DIR  = Path("plots_exploration") / INTERVAL_STR
METRICS_DIR = Path("metrics_exploration") / INTERVAL_STR

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Filenames for outputs
PLOT_PATH    = PLOTS_DIR / f"{MODEL_NAME}_{FEATURE_LABEL}_holdout_plot.png"
METRICS_PATH = METRICS_DIR / f"{MODEL_NAME}_{FEATURE_LABEL}_holdout_metrics.csv"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    # Load bundle
    print(f"Loading predictions from: {PKL_PATH.resolve()}")
    with open(PKL_PATH, "rb") as f:
        bundle = pickle.load(f)

    data = bundle["data"]
    y_pred               = np.asarray(data["pred"])
    y_tasks_true         = np.asarray(data["num_tasks_true"])
    y_browser_true       = np.asarray(data["num_browser_tasks_true"])
    y_non_browser_true   = np.asarray(data["num_non_browser_tasks_true"])

    # Compute metrics
    records = []
    for name, y_true in [
        ("num_tasks_true",            y_tasks_true),
        ("num_browser_tasks_true",    y_browser_true),
        ("num_non_browser_tasks_true",y_non_browser_true),
    ]:
        # overall metrics
        mae_val  = mean_absolute_error(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        # non-zero metrics
        mask_nz  = y_true != 0
        if np.any(mask_nz):
            mae_nz  = mean_absolute_error(y_true[mask_nz], y_pred[mask_nz])
            rmse_nz = rmse(y_true[mask_nz], y_pred[mask_nz])
        else:
            mae_nz, rmse_nz = None, None

        records.append({
            "target":           name,
            "MAE":              mae_val,
            "RMSE":             rmse_val,
            "MAE_nonzero":      mae_nz,
            "RMSE_nonzero":     rmse_nz,
        })

    # Save metrics to CSV
    df_metrics = pd.DataFrame.from_records(records)
    df_metrics.to_csv(METRICS_PATH, index=False)
    print(f"Saved metrics to: {METRICS_PATH.resolve()}")

    # Prepare slice for plotting
    start = PLOT_START
    end   = PLOT_END
    x_axis = np.arange(len(y_pred))[start:end]

    y_b  = y_browser_true[start:end]
    y_nb = y_non_browser_true[start:end]
    y_p  = y_pred[start:end]

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(x_axis, y_b,  label="Network-active tasks (true)", linewidth=1)
    plt.plot(x_axis, y_nb, label="Network-inactive tasks (true)", linewidth=1)
    plt.plot(x_axis, y_p,  label="Model prediction (num_tasks)", linewidth=2, linestyle="--")

    plt.title(f"Hold-out predictions – {MODEL_NAME} @ {INTERVAL_STR}")
    plt.xlabel("Sequence index")
    plt.ylabel("Task count")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Saved plot to: {PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()