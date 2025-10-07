import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, zscore, rankdata
import statsmodels.api as sm

from utils.functions import *

# ----------------------------- CONSTANTS ------------------------------
TASK_START = 4      # first task index shown on the x-axis
MIX_START = 0       # first mixing %
MIX_STEP  = 10      # step between successive mixing %

EPS = 1e-8          # avoid division by zero in the score

# --------------------------- I/O UTILITIES ----------------------------
def load_datasets(path: Path):
    """Load the (pickle) list-of-lists  datasets[task][mix]  structure."""
    with path.open('rb') as f:
        return pickle.load(f)


# ----------------------------- METRICS --------------------------------
def evaluate_matrix(matrix, start_tasks=TASK_START,
                    start_mixing=MIX_START, by_mixing=MIX_STEP):
    """
    Parameters
    ----------
    matrix : list[list[float]]
        shape = (n_tasks, n_mix_levels)

    Returns
    -------
    dict with keys:
        mean_pearson, mean_spearman,
        mixing_slope_abs, score
    """
    arr = np.asarray(matrix)                            # (T, M)
    n_tasks, n_mix = arr.shape
    tasks = np.arange(start_tasks, start_tasks + n_tasks)

    # --- A. task correlations per mixing level ------------------------
    pearsons, spearmans = [], []
    for j in range(n_mix):
        col = arr[:, j]
        # Pearson on z-scored column
        pearsons.append(pearsonr(tasks, zscore(col))[0])
        # Spearman on ranks
        spearmans.append(spearmanr(tasks, rankdata(col))[0])

    mean_pearson  = float(np.nanmean(pearsons))
    mean_spearman = float(np.nanmean(spearmans))

    # --- B. mixing-slope on normalized means --------------------------
    # raw column means
    col_means = arr.mean(axis=0)
    # normalize to dimensionless (divide by global mean)
    col_means_norm = col_means / col_means.mean()

    mixings = np.asarray(
        [start_mixing + j * by_mixing for j in range(n_mix)],
        dtype=float
    )

    X = sm.add_constant(mixings)                        # [1, mixing%]
    model = sm.OLS(col_means_norm, X).fit()
    slope_norm = model.params[1]                        # coefficient
    mixing_slope_abs = float(abs(slope_norm))

    # --- C. composite score -------------------------------------------
    score = mean_spearman / (mixing_slope_abs + EPS)

    return dict(mean_pearson=mean_pearson,
                mean_spearman=mean_spearman,
                mixing_slope_abs=mixing_slope_abs,
                score=score)


# ------------------------------ PLOTS ---------------------------------
def _regression_line(x, y):
    """Return endpoints for a simple OLS line fit."""
    m, b = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ys = m * xs + b
    return xs, ys


def plot_results(matrix, func_name, metrics, output_dir: Path,
                 start_tasks: int = TASK_START,
                 start_mixing: int = MIX_START,
                 by_mixing: int = MIX_STEP):
    """
    Create a 2-row figure (original & rank plots) with regression lines
    and correlation annotations.  Save to <output_dir>/<func_name>.png
    """
    arr = np.asarray(matrix)
    n_tasks, n_mix = arr.shape
    tasks = np.arange(start_tasks, start_tasks + n_tasks)

    fig, axes = plt.subplots(2, n_mix, figsize=(3.2 * n_mix, 6),
                             sharey='row', sharex='col')

    if n_mix == 1:          # homogenize shape
        axes = np.array([[axes[0]], [axes[1]]])

    for j in range(n_mix):
        mix_pct = start_mixing + j * by_mixing
        y_orig = arr[:, j]
        y_rank = rankdata(y_orig)

        # Original scatter + regression + r
        ax0 = axes[0, j]
        ax0.scatter(tasks, y_orig, s=30)
        xs, ys = _regression_line(tasks, y_orig)
        ax0.plot(xs, ys, ls='--', lw=1)
        r = pearsonr(tasks, zscore(y_orig))[0]
        ax0.text(0.04, 0.9, f"r = {r:.2f}", transform=ax0.transAxes)
        if j == 0:
            ax0.set_ylabel(func_name)
        ax0.grid(True, ls=':')

        # Rank scatter + regression + ρ
        ax1 = axes[1, j]
        ax1.scatter(tasks, y_rank, s=30, color='tab:orange')
        xs, ys = _regression_line(tasks, y_rank)
        ax1.plot(xs, ys, ls='--', lw=1, color='tab:orange')
        rho = spearmanr(tasks, y_rank)[0]
        ax1.text(0.04, 0.9, f"ρ = {rho:.2f}", transform=ax1.transAxes)
        if j == 0:
            ax1.set_ylabel(f"rank({func_name})")
        ax1.set_xlabel("Tasks")
        ax1.grid(True, ls=':')

        ax0.set_title(f"{mix_pct}% mixing")

    # Overall title with updated metrics
    sup = (f"{func_name}   |   "
           f"mean r = {metrics['mean_pearson']:.3f}   •   "
           f"mean ρ = {metrics['mean_spearman']:.3f}   •   "
           f"|slope| = {metrics['mixing_slope_abs']:.4g}   •   "
           f"score = {metrics['score']:.3g}")
    fig.suptitle(sup, fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = output_dir / f"{func_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_results_pro(
    matrix,
    func_name,
    metrics,
    output_dir: Path,
    start_tasks: int = TASK_START,
    start_mixing: int = MIX_START,
    by_mixing: int = MIX_STEP,
):
    """
    2 x 4 grid, with 3 centered top plots and 4 bottom plots.
    Uses task indices shifted by -1, bigger fonts, saves as SVG with “_pro”.
    """

    # ------------------------------------------------------------------ #
    # internal helper: simple linear regression line
    # ------------------------------------------------------------------ #
    def _regression_line(x, y):
        coeffs = np.polyfit(x, y, 1)
        poly_fn = np.poly1d(coeffs)
        return x, poly_fn(x)

    # ------------------------------------------------------------------ #
    # “professional” look — updated colours
    # ------------------------------------------------------------------ #
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "#66706d",         # grid colour
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # ------------------------------------------------------------------ #
    # data prep
    # ------------------------------------------------------------------ #
    arr = np.asarray(matrix)
    n_tasks, n_mix = arr.shape
    tasks = np.arange(start_tasks, start_tasks + n_tasks) - 1

    # x-tick positions: 3, 5, 7, …
    xticks = np.arange(3, tasks[-1] + 1, 2)

    # ------------------------------------------------------------------ #
    # figure and axes
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(3.2 * 4, 6))
    gs = gridspec.GridSpec(2, 8, figure=fig)

    ax_top1 = fig.add_subplot(gs[0, 1:3])
    ax_top2 = fig.add_subplot(gs[0, 3:5], sharey=ax_top1)
    ax_top3 = fig.add_subplot(gs[0, 5:7], sharey=ax_top1)

    ax_bot1 = fig.add_subplot(gs[1, 0:2], sharex=ax_top1, sharey=ax_top1)
    ax_bot2 = fig.add_subplot(gs[1, 2:4], sharex=ax_top1, sharey=ax_top1)
    ax_bot3 = fig.add_subplot(gs[1, 4:6], sharex=ax_top1, sharey=ax_top1)
    ax_bot4 = fig.add_subplot(gs[1, 6:8], sharex=ax_top1, sharey=ax_top1)

    axes_list = [ax_top1, ax_top2, ax_top3, ax_bot1, ax_bot2, ax_bot3, ax_bot4]

    # ------------------------------------------------------------------ #
    # plotting loop
    # ------------------------------------------------------------------ #
    for j in range(n_mix):
        ax = axes_list[j]
        mix_pct = start_mixing + j * by_mixing
        y = arr[:, j]

        # scatter + regression
        ax.scatter(tasks, y, s=30, alpha=0.8, color="#93353a")  # point colour
        xs, ys = _regression_line(tasks, y)
        ax.plot(xs, ys, ls="--", lw=1, color="#c4ab12")         # line colour

        # annotate Pearson r
        r = pearsonr(tasks, zscore(y))[0]
        ax.text(
            0.05,
            0.90,
            f"r = {r:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            fontstyle="italic",
        )

        ax.set_title(f"{mix_pct}% mixing", fontsize=12)

        if ax in [ax_top1, ax_bot1]:
            ax.set_ylabel("Main-IP counts", fontsize=12)
        if ax in [ax_bot1, ax_bot2, ax_bot3, ax_bot4]:
            ax.set_xlabel("Task-switch counts", fontsize=12)

        # apply uniform x-ticks
        ax.set_xticks(xticks)

    fig.tight_layout()

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{func_name}_pro.svg"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved professional plot: {out_path}")


# ------------------------------- MAIN ---------------------------------
def extract_results(datasets, func):
    """Apply `func` to each dataset -> matrix[T][M]."""
    results = []
    for task_level in tqdm(datasets, desc=f"Applying {func.__name__}"):
        level_results = [func(ds) for ds in task_level]
        results.append(level_results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyse mixing correlation & plot results."
    )
    parser.add_argument('-d', '--data-path', type=Path,
                        default=Path('mixed_datasets.pkl'),
                        help='Pickle with list[list[pd.DataFrame]].')
    parser.add_argument('-o', '--output-dir', type=Path,
                        default=Path('./results/mixing_correlation_results'),
                        help='Directory for the plots and CSV.')
    parser.add_argument('-f', '--functions', nargs='+',
                        choices=['unique_ips', 'unique_ports', 'num_S_AS', 'num_rare_triplets',
                                 'local_minima_var_send', 'num_start_matches', 'num_start_matches_cons',
                                 'num_highest_len_ip', 'num_highest_len_ip_mean', 'num_main_ips',
                                 'num_untangled_tasks'],
                        default=['unique_ips', 'unique_ports', 'num_S_AS', 'num_rare_triplets',
                                 'local_minima_var_send', 'num_start_matches', 'num_start_matches_cons',
                                 'num_highest_len_ip', 'num_highest_len_ip_mean', 'num_main_ips',
                                 ],
                        help='Metric functions to evaluate.')
    parser.add_argument('--csv-name', type=str, default='summary_metrics.csv',
                        help='Name of the CSV file to write.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(args.data_path)

    func_map = {'unique_ips': unique_ips,
                'unique_ports': unique_ports,
                'num_S_AS': num_S_AS,
                'num_rare_triplets': num_rare_triplets_tcp,
                'local_minima_var_send': num_local_minima_moving_var_send,
                'num_start_matches': num_start_matches,
                'num_start_matches_cons': num_start_matches_consecutive,
                'num_highest_len_ip': num_highest_frame_len_by_ip,
                'num_highest_len_ip_mean': num_highest_frame_len_by_ip_mean,
                'num_main_ips': num_main_ips,
                'num_untangled_tasks': num_tasks_after_untangle_tasks
                }

    summaries = []

    for name in args.functions:
        func = func_map[name]
        matrix = extract_results(datasets, func)
        metrics = evaluate_matrix(matrix)
        plot_results(matrix, name, metrics, args.output_dir)
        plot_results_pro(matrix, name, metrics, args.output_dir)
        summaries.append(dict(algorithm=name, **metrics))

    # Write CSV sorted by the new score
    df = (pd.DataFrame(summaries)
            .sort_values('score', ascending=False)
            .reset_index(drop=True))
    csv_path = args.output_dir / args.csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary -> {csv_path}\n")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
