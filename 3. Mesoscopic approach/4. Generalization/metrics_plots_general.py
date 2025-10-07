#!/usr/bin/env python
"""
Evaluate experiment results
--------------------------
Given an experiment timestamp (as created by ``run_experiment.py``), this
script searches the corresponding ``experiments/exp_<timestamp>`` directory
for all ``*.pkl`` files that contain the stored predictions.

For every pickle file it reconstructs:
    • A scatter plot of *sum of true values* vs *sum of predicted values*.
    • A residual plot (prediction − truth).
    • A CSV with common regression metrics.

All artefacts are **saved back into the same experiment folder** next to the
original ``.pkl`` file.

Example
-------
.. code:: bash

    # Timestamp only – the ``exp_`` prefix is added automatically
    python evaluate_experiment.py --exp 20250528_104233

    # Full relative or absolute path is also accepted
    python evaluate_experiment.py --exp experiments/exp_20250528_104233
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

###############################################################################
# ------------------------------ IO helpers --------------------------------- #
###############################################################################

def load_predictions(pkl_path: Path):
    """Return two 1‑D arrays with the *per‑split sums* of y_true and y_pred."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    y_true_sums, y_pred_sums = [], []
    for values in data.values():  # keys are split IDs – order does not matter
        y = np.asarray(values["true"], dtype=float)
        p = np.asarray(values["pred"], dtype=float)
        y_true_sums.append(y.sum())
        y_pred_sums.append(p.sum())

    return np.asarray(y_true_sums), np.asarray(y_pred_sums)


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_file: Path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "--")
    plt.xlabel("True Sum")
    plt.ylabel("Predicted Sum")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_file: Path):
    residuals = y_pred - y_true
    plt.figure()
    plt.scatter(y_true, residuals, alpha=0.7)
    plt.hlines(0, xmin=y_true.min(), xmax=y_true.max(), linestyles="--")
    plt.xlabel("True Sum")
    plt.ylabel("Residual (Pred − True)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
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
        "ResidualCorr": res_corr,
    }

###############################################################################
# ------------------------------- Main -------------------------------------- #
###############################################################################

def _infer_model_and_feat(stem: str) -> tuple[str, str]:
    """Split a file stem ``model_feature[_…].pkl`` into ``(model, feature)``.

    If the feature label itself contains underscores (e.g. ``features_corr_var``)
    we split only on the *first* underscore.
    """
    if "_" not in stem:
        raise ValueError(f"Cannot infer model / features from “{stem}” – expected at least one underscore.")
    model, feat = stem.split("_", 1)
    return model, feat


def find_experiment_dir(exp_arg: str) -> Path:
    """Return the absolute experiment directory given the user input."""
    p = Path(exp_arg)

    # If the user already provided a path that exists – use it as‑is.
    if p.exists():
        if p.is_file():
            raise ValueError("--exp should be a directory or timestamp, not a file.")
        return p.resolve()

    # Otherwise treat it as a timestamp (with or without the ``exp_`` prefix)
    exp_name = exp_arg
    if not exp_name.startswith("exp_"):
        exp_name = f"exp_{exp_name}"
    exp_dir = Path("experiments") / exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)
    return exp_dir.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Re‑create plots and metrics for a given experiment timestamp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment timestamp (e.g. 20250528_104233) OR path to the experiment directory.",
    )
    args = parser.parse_args()

    exp_dir = find_experiment_dir(args.exp)
    print(f"🔍 Using experiment directory: {exp_dir}")

    pkl_paths = sorted(exp_dir.glob("*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No *.pkl files found in {exp_dir}")

    # Process every pickle → plots + metrics
    metrics_records = []
    for pkl_path in pkl_paths:
        stem = pkl_path.stem  # e.g. "lgbm_features_corr_var"
        model, features = _infer_model_and_feat(stem)

        # ------------------------------------------------------------------
        # Load predictions
        # ------------------------------------------------------------------
        y_true, y_pred = load_predictions(pkl_path)

        # ------------------------------------------------------------------
        # Plots
        # ------------------------------------------------------------------
        scatter_path = exp_dir / f"{stem}_scatter.png"
        resid_path = exp_dir / f"{stem}_residual.png"
        plot_scatter(y_true, y_pred, f"{model} – {features} (scatter)", scatter_path)
        plot_residuals(y_true, y_pred, f"{model} – {features} (residuals)", resid_path)

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        m = compute_metrics(y_true, y_pred)
        metrics_records.append({"Model": model, "Features": features, **m})
        print(f"✔ {stem}: plots & metrics written")

    # Consolidate all metric rows into a single CSV
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = exp_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    print("\n All done!")
    print(f"   Plots + individual CSVs saved next to each .pkl file")
    print(f"   Combined metrics -> {metrics_csv}")


if __name__ == "__main__":
    main()
