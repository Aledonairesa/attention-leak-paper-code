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

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, model: str, features: str, output_dir: Path, suffix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], '--')
    plt.xlabel("True Sum")
    plt.ylabel("Predicted Sum")
    title_suffix = f" ({suffix[1:]})" if suffix else " (normal)"
    plt.title(f"{model} – {features} Scatter{title_suffix}")
    fname_suffix = suffix if suffix else "_normal"
    plt.savefig(output_dir / f"{model}_{features}_scatter{fname_suffix}.png")
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model: str, features: str, output_dir: Path, suffix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_pred - y_true
    plt.figure()
    plt.scatter(y_true, residuals)
    plt.hlines(0, xmin=float(np.min(y_true)), xmax=float(np.max(y_true)), linestyles='--')
    plt.xlabel("True Sum")
    plt.ylabel("Residual (Pred – True)")
    title_suffix = f" ({suffix[1:]})" if suffix else " (normal)"
    plt.title(f"{model} – {features} Residuals{title_suffix}")
    fname_suffix = suffix if suffix else "_normal"
    plt.savefig(output_dir / f"{model}_{features}_residual{fname_suffix}.png")
    plt.close()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    res = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    mbe = float(np.mean(res))
    # Manejo robusto de correlaciones cuando hay poca varianza o longitud < 2
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r, _ = pearsonr(y_true, y_pred)
        res_corr, _ = pearsonr(y_true, res)
    else:
        r, res_corr = np.nan, np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # r2_score requiere al menos dos muestras
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
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
    parser.add_argument("--model", required=True, help="Model name (e.g., TCN)")
    parser.add_argument("--features", required=True, help="Feature label (e.g., features_corr_var)")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Interval used in split folder (e.g. 2.0 or 5.0)"
    )
    parser.add_argument(
        "--injected",
        nargs="+",
        type=str,
        default=None,
        help="One or more injected variants to load (e.g., 100 200 Mix). If omitted, uses normal."
    )
    parser.add_argument(
        "--include_normal",
        action="store_true",
        help="If set together with --injected, also evaluate the normal (non-injected) PKL."
    )
    args = parser.parse_args()

    interval_str = f"{args.interval}s"

    base_pkl_dir = Path("prediction_pkls") / interval_str
    plots_root   = Path("plots") / interval_str
    metrics_root = Path("metrics") / interval_str
    plots_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    # Construye la lista de (suffix, pkl_path, subdir_name)
    variants = []
    if args.injected:
        if args.include_normal:
            # normal primero
            variants.append(("", base_pkl_dir / f"{args.model}_{args.features}.pkl", "normal"))
        for inj in args.injected:
            suffix = f"_inj{inj}"
            pkl_file = base_pkl_dir / f"{args.model}_{args.features}{suffix}.pkl"
            variants.append((suffix, pkl_file, f"inj{inj}"))
    else:
        # Solo normal
        pkl_file = base_pkl_dir / f"{args.model}_{args.features}.pkl"
        variants.append(("", pkl_file, "normal"))

    for suffix, pkl_file, subdir in variants:
        if not pkl_file.exists():
            print(f"[WARN] PKL not found, skipping: {pkl_file}")
            continue

        y_true, y_pred = load_predictions(pkl_file)

        # Subcarpetas por variante (evita colisiones) + sufijo en filename
        plots_dir   = plots_root / subdir
        metrics_dir = metrics_root / subdir
        plots_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Plots
        plot_scatter(y_true, y_pred, args.model, args.features, plots_dir, suffix)
        plot_residuals(y_true, y_pred, args.model, args.features, plots_dir, suffix)

        # Métricas
        mets = compute_metrics(y_true, y_pred)
        df = pd.DataFrame([{"Model": args.model, "Features": args.features, "Variant": (suffix[1:] if suffix else "normal"), **mets}])
        fname_suffix = suffix if suffix else "_normal"
        df.to_csv(metrics_dir / f"{args.model}_{args.features}_metrics{fname_suffix}.csv", index=False)

        print(f"✔ {subdir}: Scatter + residual -> {plots_dir}")
        print(f"✔ {subdir}: Metrics CSV -> {metrics_dir}")

if __name__ == "__main__":
    main()
