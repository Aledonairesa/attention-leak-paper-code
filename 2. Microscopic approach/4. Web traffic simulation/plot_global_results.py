import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_ari_evolution(results_dir, out_path):
    """
    Plot ARI_non_omitted evolution for Base and Optimized models
    across all available (alpha, beta) pairs found in results_dir.
    """
    # adjust font size for one-column usage
    plt.rcParams.update({'font.size': 14})

    metric = "ARI_non_omitted"
    models_to_plot = {"Base", "Hyperopt"}

    # gather series data
    series = {}
    pairs = set()
    for sub in os.listdir(results_dir):
        if "_a" not in sub or "_b" not in sub:
            continue
        model, rest = sub.split("_a", 1)
        if model not in models_to_plot:
            continue
        try:
            alpha_str, beta_str = rest.split("_b")
            alpha, beta = float(alpha_str), float(beta_str)
        except ValueError:
            continue

        csv_path = os.path.join(results_dir, sub, "metrics_progress_new.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            continue

        series[(model, alpha, beta, sub)] = df[["frames", metric]].copy()
        pairs.add((alpha, beta))

    if not series:
        raise RuntimeError(f"No data found for {metric} in {results_dir}")

    # custom line styles for specific alpha/beta pairs
    style_of = {}
    for p in pairs:
        a, b = p
        if (a, b) == (0.01, 0.99) or (a, b) == (0.99, 0.01):
            style_of[p] = '-'
        elif (a, b) == (0.1, 0.9) or (a, b) == (0.9, 0.1):
            style_of[p] = '-.'
        elif (a, b) == (0.2, 0.8) or (a, b) == (0.8, 0.2):
            style_of[p] = '--'
        else:
            style_of[p] = ':'  # fallback

    color_of = {"Hyperopt": "#93353a", "Base": "#299f97"}
    task_color = "#c4ab12"
    grid_color = "#66706d"

    # load task‐change frames
    task_lines = []
    for sub in os.listdir(results_dir):
        if "_a" not in sub or "_b" not in sub:
            continue
        fp = os.path.join(results_dir, sub, "task_first_frames.json")
        if os.path.isfile(fp):
            with open(fp) as f:
                task_lines = json.load(f)
            break

    # adjusted figure size
    plt.figure(figsize=(7, 5))

    # plot each series
    for (model, alpha, beta, sub), df_m in series.items():
        map_label = "Optimized" if model == "Hyperopt" else "Base"
        label = f"{map_label} (α={alpha}, β={beta})"
        plt.plot(
            df_m["frames"],
            df_m[metric],
            label=label,
            color=color_of.get(model),
            linestyle=style_of[(alpha, beta)],
            linewidth=2
        )

    # vertical lines for task changes
    for idx, x in enumerate(sorted(task_lines)):
        plt.axvline(
            x=x,
            color=task_color,
            linestyle=':',  # dotted
            linewidth=2,
            alpha=0.7,
            label="Task change" if idx == 0 else None
        )

    plt.xlabel("Number of packets processed")
    plt.ylabel("Adjusted Rand Index (ARI)")
    plt.grid(color=grid_color, alpha=0.4)
    plt.tight_layout()

    # legend with white background, smaller fontsize
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = plt.legend(by_label.values(), by_label.keys(), fontsize="small", frameon=True)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.7)

    # save as SVG
    plt.savefig(out_path, format='svg', dpi=300)
    plt.close()

def plot_timeline_comparison(truth_file, pred_file, out_path, dpi=300, figsize=(12, 2.5)):
    """
    Draw two horizontal bars—ground‑truth vs. predicted cluster IDs—for a single run.
    """
    def _load(path):
        if path.endswith(".pkl"):
            return pd.read_pickle(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".feather"):
            return pd.read_feather(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    truth_df = _load(truth_file)
    pred_df = _load(pred_file)

    truth_df = truth_df.reset_index().rename(columns={"index": "orig_index"})
    merged = (
        pred_df[["orig_index", "dataset_id"]]
        .merge(
            truth_df[["orig_index", "dataset_id"]]
            .rename(columns={"dataset_id": "true_label"}),
            on="orig_index",
            how="inner"
        )
        .sort_values("orig_index")
    )

    if merged.empty:
        raise RuntimeError("No overlapping orig_index values between truth and pred!")

    gt_labels = merged["true_label"].to_numpy()
    pred_labels = merged.get("dataset_id_x", merged["dataset_id"]).to_numpy()

    unique_ids = sorted(set(gt_labels) | set(pred_labels))
    n_cols = max(3, len(unique_ids))
    cmap = plt.cm.get_cmap("tab20", n_cols)
    idx_of = {lab: i for i, lab in enumerate(unique_ids)}
    gt_idx = [idx_of[x] for x in gt_labels]
    pr_idx = [idx_of[x] for x in pred_labels]

    import numpy as np
    data = np.vstack([gt_idx, pr_idx])

    plt.figure(figsize=figsize)
    plt.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.yticks([0, 1], ["Ground truth", "Predicted"])
    plt.xlabel("Frame index")
    plt.title("Task timeline: truth vs. model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot ARI evolution and timelines for all runs in a results directory"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results/",
        help="Directory containing sub‑folders named like Model_a<alpha>_b<beta>"
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    # ARI evolution
    ari_out = os.path.join(results_dir, "ari_evolution.svg")
    plot_ari_evolution(results_dir, ari_out)
    print(f"Saved ARI_global evolution plot to {ari_out}")

    # default truth file
    truth_file = os.path.join(results_dir, "mixed_dataset.pkl")

    # timeline comparisons for each sub‑folder
    for sub in os.listdir(results_dir):
        if "_a" not in sub or "_b" not in sub:
            continue
        folder = os.path.join(results_dir, sub)
        # find predictions
        pred_file = None
        for ext in ["csv", "pkl", "parquet", "feather"]:
            candidate = os.path.join(folder, f"assignments_pred.{ext}")
            if os.path.isfile(candidate):
                pred_file = candidate
                break
        if not pred_file:
            continue

        tl_out = os.path.join(folder, "timeline_comparison.png")
        plot_timeline_comparison(truth_file, pred_file, tl_out)
        print(f"Saved timeline comparison for {sub} to {tl_out}")

if __name__ == "__main__":
    main()
