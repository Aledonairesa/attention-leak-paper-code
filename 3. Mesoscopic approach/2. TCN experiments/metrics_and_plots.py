import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_predictions(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    y_true_sums = []
    y_pred_sums = []
    for split, values in data.items():
        y = np.array(values["true"])
        p = np.array(values["pred"])
        y_true_sums.append(y.sum())
        y_pred_sums.append(p.sum())
    return np.array(y_true_sums), np.array(y_pred_sums)

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, model: str, features: str, output_dir: Path):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], '--')
    plt.xlabel("True Sum")
    plt.ylabel("Predicted Sum")
    plt.title(f"{model} – {features} Scatter")
    plt.savefig(output_dir / f"{model}_{features}_scatter.png")
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model: str, features: str, output_dir: Path):
    residuals = y_pred - y_true
    plt.figure()
    plt.scatter(y_true, residuals)
    plt.hlines(0, xmin=y_true.min(), xmax=y_true.max(), linestyles='--')
    plt.xlabel("True Sum")
    plt.ylabel("Residual (Pred – True)")
    plt.title(f"{model} – {features} Residuals")
    plt.savefig(output_dir / f"{model}_{features}_residual.png")
    plt.close()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    res = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    mbe = res.mean()
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    res_corr, _ = pearsonr(y_true, res)
    return {
        "MAE": mae,
        "MBE": mbe,
        "PearsonR": r,
        "RMSE": rmse,
        "R2": r2,
        "ResidualCorr": res_corr
    }

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct plots & metrics from prediction PKL files, per interval"
    )
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--features", required=True, help="Feature label")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Interval used in split folder (e.g. 2.0 or 5.0)"
    )
    args = parser.parse_args()

    interval_str = f"{args.interval}s"

    # Paths
    base_pkl_dir = Path("prediction_pkls") / interval_str
    pkl_file = base_pkl_dir / f"{args.model}_{args.features}.pkl"

    plots_dir = Path("plots") / interval_str
    metrics_dir = Path("metrics") / interval_str
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load and process
    y_true, y_pred = load_predictions(pkl_file)

    # Recreate plots
    plot_scatter(y_true, y_pred, args.model, args.features, plots_dir)
    plot_residuals(y_true, y_pred, args.model, args.features, plots_dir)

    # Compute & save metrics
    mets = compute_metrics(y_true, y_pred)
    df = pd.DataFrame([{"Model": args.model, "Features": args.features, **mets}])
    df.to_csv(metrics_dir / f"{args.model}_{args.features}_metrics.csv", index=False)

    print(f" Scatter + residual plots -> {plots_dir}")
    print(f" Metrics CSV -> {metrics_dir}")

if __name__ == "__main__":
    main()
