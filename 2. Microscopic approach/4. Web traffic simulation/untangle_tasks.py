import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ipaddress
from tqdm import tqdm
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    adjusted_rand_score,
    v_measure_score,
    pair_confusion_matrix
)

# ----------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------
from utils.preprocessing import (
    process_frame_time_column, remove_nan_ip_proto, filter_hosts,
    remove_frame_number, merge_ports, replace_nans_in_tcp, unify_hosts,
    merge_ips_and_create_send_column, add_asn_info
)
from utils.preprocessing_features import normalize_columns
from utils.extract_features_utils import extract_features_single_test_sample
# ----------------------------------------------------------------------


def _safe_pair_f1(y_true, y_pred):
    """Compute pairwise F1 the same way as in the global block."""
    cm = pair_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def compute_progress_metrics(df: pd.DataFrame, step: int = 200) -> pd.DataFrame:
    """
    Build a DataFrame with the same core metrics every <step> frames.

    The input *df* must contain the columns:
        - 'dataset_id'  … predicted label
        - 'true_label'  … ground-truth label
    """
    df_sorted = df.sort_values('frame.time').reset_index(drop=True)
    y_true_full = df_sorted['true_label'].to_numpy()
    y_pred_full = df_sorted['dataset_id'].to_numpy()
    omitted_id = df_sorted['dataset_id'].max()      # last cluster == “omitted”

    checkpoints = list(range(step, len(df_sorted) + 1, step))
    records = []
    for n in checkpoints:
        y_t = y_true_full[:n]
        y_p = y_pred_full[:n]

        rec = {'frames': n}
        rec['ARI_global'] = adjusted_rand_score(y_t, y_p)
        rec['V_measure_global'] = v_measure_score(y_t, y_p)
        rec['pair_f1'] = _safe_pair_f1(y_t, y_p)

        mask_no_om = y_p != omitted_id
        y_t_no = y_t[mask_no_om]
        y_p_no = y_p[mask_no_om]

        rec['accuracy_global'] = (y_p == y_t).mean()
        rec['accuracy_non_omitted'] = accuracy_score(y_t_no, y_p_no) if len(y_t_no) else 0.0
        rec['f1_weighted'] = f1_score(y_t_no, y_p_no, average='weighted') if len(y_t_no) else 0.0
        records.append(rec)

    return pd.DataFrame(records)


def load_config(path):
    """
    Load a JSON configuration file.
    Returns a dict with keys:
      - data_dir: path to filtered datasets
      - output_dir: folder to save results
      - websites: list of domain substrings to select datasets
      - time_shifts: list of floats for timeline mixing
      - models: list of [name, path] tuples for CatBoost .pkl models
      - alpha_beta_pairs: list of [alpha, beta] thresholds
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_trained_model(path):
    """
    Load a CatBoostClassifier model saved via joblib.dump(...).
    """
    return joblib.load(path)


def pre_mixing_superpositions(datasets, time_shifts, save_path):
    """
    Normalize and shift each dataset's 'frame.time' axis to create a single timeline.
    Plots a horizontal bar per dataset to visualize time ranges.

    Args:
        datasets (list of DataFrame): each must have 'frame.time'.
        time_shifts (list of float): length = len(datasets)-1
        save_path (str): where to save the timeline image
    Returns:
        ranges (list of (start, end) tuples): time offsets applied per dataset.
    """
    n = len(datasets)
    assert len(time_shifts) == n - 1, "time_shifts must be length #datasets - 1"

    # Original color palette (one color per possible dataset_id)
    colors = [
        'red', 'blue', 'lime', 'fuchsia', 'black', 'grey', 'gold',
        'pink', 'darkorange', 'green', 'darkorchid', 'darkturquoise',
        'goldenrod', 'wheat', 'lightcoral', 'lightgray', 'turquoise',
        'firebrick', 'navy', 'lightgreen', 'beige', 'deeppink', 'lightblue'
    ]

    # Create a figure: one bar (hlines) per dataset
    fig, ax = plt.subplots(figsize=(10, n * 0.6))
    current_start = 0
    ranges = []

    for i, df in enumerate(datasets):
        # Compute duration and assign start/end on unified timeline
        duration = df['frame.time'].max() - df['frame.time'].min()
        start = current_start
        end = start + duration
        ranges.append((start, end))

        # Plot thick line: y position reversed so first dataset on top
        ax.hlines(
            y=n - 1 - i,
            xmin=start,
            xmax=end,
            color=colors[i % len(colors)],
            linewidth=12
        )

        # Advance start by shift for next dataset
        if i < len(time_shifts):
            current_start += time_shifts[i]

    # Label axes and grid
    ax.set_xlabel('Time (seconds)')
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Dataset {i+1}" for i in range(n)][::-1])
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save and close
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return ranges


def mix_datasets(dfs, time_ranges):
    """
    Apply the computed time offsets to each DataFrame and concatenate them into one.

    Args:
        dfs (list of DataFrame): processed per-website datasets
        time_ranges (list of tuples): start offsets from pre_mixing_superpositions
    Returns:
        merged_df (DataFrame): all rows sorted by new 'frame.time'
    """
    processed = []
    for idx, (df, (start, _)) in enumerate(zip(dfs, time_ranges), start=1):
        df_copy = df.copy()
        df_copy['dataset_id'] = idx

        # Normalize: shift its 'frame.time' so earliest is at 'start'
        offset = df_copy['frame.time'].iloc[0]
        df_copy['frame.time'] = df_copy['frame.time'] - offset + start

        processed.append(df_copy)

    # Concatenate and sort
    merged_df = pd.concat(processed, ignore_index=True)
    return merged_df.sort_values('frame.time').reset_index(drop=True)


def plot_dataset_ids(df, save_path, dpi=300):
    """
    Two-row plot: (1) timeline rectangles per frame, (2) line+points by cluster_id.
    Uses original color palette. Saves only; no pop-up.

    Args:
        df (DataFrame): must contain 'frame.time' and 'dataset_id'
        save_path (str): path to write PNG
    """
    # Validate required columns
    for col in ['frame.time', 'dataset_id']:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column '{col}'")

    cluster_ids = sorted(df['dataset_id'].unique())
    n_clusters  = len(cluster_ids)
    cmap        = plt.cm.get_cmap('tab20', n_clusters)   # any named cmap works

    def pick_colour(cid: int):
        """cid starts at 1 → map to 0‑based index in the colormap"""
        return cmap(cid - 1)

    times = df['frame.time'] - df['frame.time'].iloc[0]
    ids = df['dataset_id'].astype(int)

    colors = [
        'red', 'blue', 'lime', 'fuchsia', 'black', 'grey', 'gold',
        'pink', 'darkorange', 'green', 'darkorchid', 'darkturquoise',
        'goldenrod', 'wheat', 'lightcoral', 'lightgray', 'turquoise',
        'firebrick', 'navy', 'lightgreen', 'beige', 'deeppink', 'lightblue'
    ]

    # Setup subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(17, 4),
        gridspec_kw={'height_ratios': [1, 2]}
    )

    # ---- Row 1: Rectangle timeline ----
    ax1.set_xlim(times.iloc[0], times.iloc[-1])
    ax1.set_ylim(0, 1)

    # Draw one rectangle per frame (width = gap to next)
    for i in range(len(times)-1):
        cid = ids.iloc[i]
        width = times.iloc[i+1] - times.iloc[i]
        rect = mpatches.Rectangle(
            (times.iloc[i], 0),
            width,
            1,
            color=pick_colour(cid)
        )
        ax1.add_patch(rect)

    # Last frame rectangle
    last_cid = ids.iloc[-1]
    last_width = (times.iloc[-1] - times.iloc[-2]) if len(times)>1 else 0
    ax1.add_patch(
        mpatches.Rectangle(
            (times.iloc[-1], 0), last_width, 1,
            color=pick_colour(last_cid)
        )
    )
    ax1.axis('off')  # clean up

    # ---- Row 2: Line + colored points ----
    sorted_idx = np.argsort(times.values)
    ax2.plot(
        times.values[sorted_idx],
        ids.values[sorted_idx],
        '-', color='grey', linewidth=1
    )
    for t, cid in zip(times, ids):
        ax2.plot(
            t, cid, 'o',
            color=pick_colour(cid), markersize=4
        )
    # Y-axis labels
    unique_ids = sorted(ids.unique())
    ax2.set_yticks(unique_ids)
    ax2.set_yticklabels([f"Dataset {i}" for i in unique_ids])
    ax2.set_xlabel('Time (seconds)')

    # Save and close
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def untangle_tasks(
    dataset,
    model,
    alpha,
    beta,
    process_time_column=False,
    process_hosts=True,
    drop_dataset_id=True
):
    """
    Incrementally assign each frame to a 'task' (cluster) based on model probabilities.

    Workflow:
      1. Preprocess raw columns (frame number, TCP ports, NaNs).
      2. Optionally normalize time and unify host columns.
      3. Extract and drop true 'dataset_id' if for evaluation.
      4. Augment features: sender flag + ASN lookup.
      5. For each frame in time order:
         - Filter out non-TCP or known IP ranges.
         - If first valid frame: start first task.
         - Else compute P(belong) for each open task.
           • If all < alpha: open new task.
           • Else if any ≥ beta: assign to best task.
           • Else: mark as omitted.
      6. Combine omitted frames as final cluster.
      7. Build final DataFrame with new 'dataset_id'.
      8. If true labels were present: compute extensive metrics.

    Returns:
      merged_df (DataFrame): all frames with assigned 'dataset_id' and 'orig_index'.
                             If truth was available, 'true_label' is also included.
      metrics (dict): evaluation metrics if truth available, else {}.
    """
    print(f"Starting untangle_tasks with α={alpha}, β={beta}")

    # ----- 1. Preprocessing pipeline -----
    df = dataset.copy()
    df['orig_index'] = df.index
    df = remove_frame_number(df)
    df = merge_ports(df)
    df = replace_nans_in_tcp(df)
    if process_time_column:
        df = process_frame_time_column(df)
    if process_hosts:
        df = unify_hosts(df)

    # ----- 2. Extract and drop ground truth if exists -----
    truth = None
    if drop_dataset_id and 'dataset_id' in df.columns:
        truth = df['dataset_id'].copy()
    df = df.drop(columns=['dataset_id'], errors='ignore')

    # ----- 3. Feature augmentation -----
    df = merge_ips_and_create_send_column(df, "172.17.")
    df = add_asn_info(df)

    # Prepare containers
    tasks = []    # list of DataFrames, one per cluster
    omitted = []  # list of DataFrames for uncertain frames

    # Pre-define IP ranges to exclude
    ip_filter = [
        (ipaddress.IPv4Address('142.250.0.0'), ipaddress.IPv4Address('142.251.255.255'))
    ]

    # ----- 4. Main clustering loop -----
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing frames"):
        proto, ip_str = row['ip.proto'], row['ip']
        ip_obj = ipaddress.IPv4Address(ip_str)

        # a) filter
        if proto != 6 or any(s <= ip_obj <= e for s, e in ip_filter):
            omitted.append(row.to_frame().T)
            continue

        # b) start first cluster if empty
        if not tasks:
            tasks.append(row.to_frame().T)
            continue

        # c) compute membership probabilities for each open task
        probs = []
        for task_df in tasks:
            # build trial dataset to extract features
            trial = pd.concat([task_df, row.to_frame().T], ignore_index=True)
            feats = extract_features_single_test_sample(trial)
            feats_df = pd.DataFrame([feats])
            feats_norm = normalize_columns(
                feats_df,
                ['diff_lenframe_to_last', 'diff_lenframe_to_last_mean']
            )
            probs.append(model.predict_proba(feats_norm)[0][1])
        tqdm.write(f"Frame {idx}: probabilities {probs}")

        # d) decide assignment
        if all(p < alpha for p in probs):
            # open new cluster
            tasks.append(row.to_frame().T)
        elif any(p >= beta for p in probs):
            # assign to cluster with max probability
            best = int(np.argmax(probs))
            tasks[best] = pd.concat([tasks[best], row.to_frame().T], ignore_index=True)
        else:
            # uncertain -> omitted
            omitted.append(row.to_frame().T)

    # ----- 5. Append omitted frames as final cluster -----
    if omitted:
        omitted_df = pd.concat(omitted, ignore_index=True)
    else:
        omitted_df = pd.DataFrame(columns=df.columns)
    tasks.append(omitted_df)
    omitted_cluster_id = len(tasks)

    # ----- 6. Merge all clusters into one DataFrame -----
    merged_list = []
    for cid, tdf in enumerate(tasks, start=1):
        tmp = tdf.copy()
        tmp['dataset_id'] = cid
        merged_list.append(tmp)
    merged_df = pd.concat(merged_list, ignore_index=True)
    merged_df = merged_df.sort_values('frame.time').reset_index(drop=True)

    # ----- 7. Compute metrics if truth was provided -----
    metrics = {}
    if truth is not None:
        # Align predictions to original indexing; omitted=cluster_id for omitted_df rows
        pred_series = pd.Series(
            merged_df['dataset_id'].values,
            index=merged_df['orig_index'].astype(int).values
        )
        y_pred_global = (
            pred_series.reindex(truth.index)
                       .fillna(-1)
                       .astype(int)
                       .values
        )
        y_true_global = truth.values

        # 7.1 Global partition metrics
        metrics['ARI_global'] = adjusted_rand_score(y_true_global, y_pred_global)
        metrics['V_measure_global'] = v_measure_score(y_true_global, y_pred_global)

        # 7.2 Pairwise precision/recall/F1
        cm = pair_confusion_matrix(y_true_global, y_pred_global)
        tn, fp, fn, tp = cm.ravel()
        precision_pw = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall_pw    = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_pw        = (2 * precision_pw * recall_pw / (precision_pw + recall_pw)
                        if precision_pw + recall_pw > 0 else 0.0)
        metrics.update({
            'pair_precision': precision_pw,
            'pair_recall': recall_pw,
            'pair_f1': f1_pw
        })

        # 7.3 Accuracy metrics
        mask_no_om = y_pred_global != omitted_cluster_id
        y_t_no = y_true_global[mask_no_om]
        y_p_no = y_pred_global[mask_no_om]
        total = len(y_true_global)
        omitted_count = total - mask_no_om.sum()

        metrics['accuracy_non_omitted'] = (
            accuracy_score(y_t_no, y_p_no) if len(y_t_no) > 0 else 0.0
        )
        metrics['accuracy_global'] = (
            (y_t_no == y_p_no).sum() / total if total > 0 else 0.0
        )
        metrics['f1_weighted'] = (
            f1_score(y_t_no, y_p_no, average='weighted')
            if len(y_t_no) > 0 else 0.0
        )

        # 7.4 Cluster counts & over/under-clustering
        n_true = len(np.unique(y_t_no))
        n_pred = len(np.unique(y_p_no))
        metrics['n_true_clusters'] = int(n_true)
        metrics['n_pred_clusters'] = int(n_pred)
        metrics['overclustering_ratio'] = (
            n_pred / n_true if n_true > 0 else float('inf')
        )

        # 7.5 Omitted stats
        metrics['n_total_frames'] = int(total)
        metrics['n_omitted_frames'] = int(omitted_count)
        metrics['pct_omitted'] = float(omitted_count / total) if total > 0 else 0.0

        # 7.6 Fragmentation per true task
        # This section adds 'true_label' to merged_df
        merged_df['true_label'] = truth.values[merged_df['orig_index'].astype(int)]
        fragments = []
        for lbl in np.unique(y_true_global):
            sub = merged_df[merged_df['true_label'] == lbl].sort_values('frame.time')
            pred_seq = sub['dataset_id'].values
            if len(pred_seq) == 0:
                fragments.append(0)
            else:
                # count changes between consecutive assignments
                fragments.append(1 + np.sum(pred_seq[1:] != pred_seq[:-1]))
        metrics['fragments_mean_per_true_task'] = float(np.mean(fragments))
        metrics['fragments_max_per_true_task'] = int(np.max(fragments))

        # 7.7 Coverage IoU per true task
        iou_list = []
        for lbl in np.unique(y_true_global):
            true_idx = np.where(y_true_global == lbl)[0]
            # find predicted label covering most of this true task
            choices, counts = np.unique(
                y_pred_global[true_idx], return_counts=True
            )
            best_pred = choices[np.argmax(counts)]
            intersection = counts.max()
            pred_idx = np.where(y_pred_global == best_pred)[0]
            union = len(true_idx) + len(pred_idx) - intersection
            iou_list.append(intersection / union if union > 0 else 0.0)
        metrics['mean_IoU_per_true_task'] = float(np.mean(iou_list))

    # Return both per-frame assignments and computed metrics
    return merged_df, metrics


# ======================================================================
#                               MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Simulate streaming frame clustering with multiple models '
                    'and thresholds, plus optional metrics-evolution output.'
    )
    parser.add_argument('config', help='Path to JSON config file')
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_dir = cfg['data_dir']
    out_root = cfg.get('output_dir', './results')
    os.makedirs(out_root, exist_ok=True)

    # ---- 1. Load and preprocess per-website datasets ----
    datasets = []
    for site in tqdm(cfg['websites'], desc='Loading datasets'):
        matches = [d for d in os.listdir(data_dir) if site in d]
        if not matches:
            raise FileNotFoundError(f"No dataset folder for '{site}' in {data_dir}")
        path = os.path.join(data_dir, matches[0], 'data.csv')

        df = pd.read_csv(path)
        df = process_frame_time_column(df)
        df = remove_nan_ip_proto(df)
        df = filter_hosts(df)
        print(f"Loaded {site}: {df.shape[0]} frames")
        datasets.append(df)

    # ---- 2. Mix datasets along a common timeline ----
    timeline_path = os.path.join(out_root, 'mixed_timeline.png')
    ranges = pre_mixing_superpositions(datasets, cfg['time_shifts'], timeline_path)
    mixed_df = mix_datasets(datasets, ranges)

    mixed_df.to_csv(os.path.join(out_root, 'mixed_dataset.csv'), index=False)
    mixed_df.to_pickle(os.path.join(out_root, 'mixed_dataset.pkl'))

    mixed_plot_path = os.path.join(out_root, 'mixed_data.png')
    plot_dataset_ids(mixed_df, mixed_plot_path) # This uses the 'dataset_id' from mixing (true labels)

    progress_step = int(cfg.get('progress_step', 100))

    # ---- 3. Iterate models and alpha/beta pairs ----
    for model_name, model_path in tqdm(cfg['models'], desc='Models'):
        model = load_trained_model(model_path)

        for alpha, beta in tqdm(cfg['alpha_beta_pairs'],
                                 desc=f"Thresh pairs for {model_name}"):
            print(f"--- Model={model_name}, α={alpha}, β={beta} ---")
            run_dir = os.path.join(out_root, f"{model_name}_a{alpha}_b{beta}")
            os.makedirs(run_dir, exist_ok=True)

            # Perform clustering. 'mixed_df' here contains the true 'dataset_id'
            # which will be used as 'true_label' inside untangle_tasks.
            merged_assignments, metrics = untangle_tasks(
                mixed_df.reset_index(drop=True), # Pass the dataframe with original 'dataset_id'
                model,
                alpha=alpha,
                beta=beta,
                process_time_column=False, # Time already processed during mixing
                process_hosts=True,
                drop_dataset_id=True      # This will use 'dataset_id' as truth
            )

            merged_assignments.to_csv(
                os.path.join(run_dir, 'assignments_pred.csv'), index=False
            )
            merged_assignments.to_pickle(
                os.path.join(run_dir, 'assignments_pred.pkl')
            )

            with open(os.path.join(run_dir, 'metrics.json'), 'w') as fp:
                json.dump(metrics, fp, indent=2)

            # merged_assignments has 'dataset_id' (predicted) and 'true_label' (if truth was available)
            if 'true_label' in merged_assignments.columns:
                prog_df = compute_progress_metrics(merged_assignments, step=progress_step)
                prog_df.to_csv(os.path.join(run_dir, 'metrics_progress.csv'),
                               index=False)

            # plot_dataset_ids expects 'dataset_id' for coloring,
            # here it's the *predicted* cluster ID from merged_assignments
            plot_dataset_ids(
                merged_assignments,
                os.path.join(run_dir, 'clustered.png')
            )

            # record where each true task first appears
            first_frames = (
                merged_assignments.reset_index()               # current row number == frame #
                    .groupby('true_label')['index']
                    .min()                       # first row where that label shows up
                    .sort_values()
                    .add(1)                      # +1 so the first frame is “1”
                    .tolist()
            )
            with open(os.path.join(run_dir, 'task_first_frames.json'), 'w') as fp:
                json.dump(first_frames, fp)

if __name__ == '__main__':
    main()