import argparse
import json
import os
import random
import pickle
import matplotlib.pyplot as plt
plt.use('Agg')

import pandas as pd
from tqdm import tqdm

from utils.preprocessing import *
from utils.extract_features_utils import *

def process_frame_time_column(df):
    """
    Vectorized parse of 'frame.time' for duration correctness:
    - Strips final space + letters (the TZ)
    - Uses pandas to_datetime on the remainder
    - Converts entire column to UNIX seconds in one go
    """
    # Remove trailing timezone (last word) with a regex
    times = df['frame.time'].str.replace(r'\s+\S+$', '', regex=True)
    # Parse all at once
    dt = pd.to_datetime(times, errors='coerce')
    # Convert to UNIX seconds
    df['frame.time'] = dt.astype('int64') / 1e9
    return df

def remove_nan_ip_proto(df):
    """
    Drop rows where ip.proto is NaN, in one vectorized call.
    If column missing, just return df unchanged.
    """
    if 'ip.proto' in df.columns:
        return df[df['ip.proto'].notna()]
    return df

def filter_hosts(df):
    """
    Vectorized IP-range filter:
    - Build a mask for any 172.17.* rows
    - Find the single most common 172.17.* IP
    - Keep all rows outside the range, plus any rows with that IP
    """
    ip_range = '172.17.'
    src = df['ip.src'].str.startswith(ip_range, na=False)
    dst = df['ip.dst'].str.startswith(ip_range, na=False)
    mask = src | dst

    # All IPs in range (from both columns)
    in_range = pd.concat([df.loc[mask, 'ip.src'], df.loc[mask, 'ip.dst']])
    if not in_range.empty:
        top_ip = in_range.value_counts().idxmax()
        # keep: not in range OR contains top_ip
        keep = ~mask | (df['ip.src'] == top_ip) | (df['ip.dst'] == top_ip)
        return df.loc[keep]
    else:
        return df


def preprocess_df(df):
    """Apply frame-time processing, remove NaNs/protocols, and filter hosts."""
    df = process_frame_time_column(df)
    df = remove_nan_ip_proto(df)
    df = filter_hosts(df)
    return df


def fast_calculate_total_times(csv_paths):
    """Duration calculation of each web connection."""
    total_times = []
    for path in tqdm(csv_paths, desc="Calculating total times", unit="file"):
        try:
            # Load full CSV so preprocessing can drop/clean consistently
            df = pd.read_csv(path)
            df = preprocess_df(df)
            duration = df["frame.time"].max() - df["frame.time"].min()
            total_times.append(duration)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(path)} (duration set to 0): {e}")
            total_times.append(0)
    return total_times


def sample_and_preprocess(data_dir, candidates, num_tasks):
    """Randomly sample N dataset directories and preprocess their data into DataFrames."""
    sampled = random.sample(candidates, num_tasks)
    processed = []
    for d in sampled:
        df = pd.read_csv(os.path.join(data_dir, d, 'data.csv'))
        processed.append(preprocess_df(df))
    return processed, sampled


def pre_mixing_superpositions_automatic(datasets, mixing, plot=False, save_plot=False,
                                        filename="plot.png", silent=False):
    """
    Plots and calculates the different start and end times for each dataset,
    with automatic time shifts based on a given mixing percentage. Optionally saves the plot and controls verbosity.

    Args:
        datasets (list of pd.DataFrame): List of dataframes, each with a 'frame.time' column.
        mixing (float): Percentage of overlap between consecutive datasets (0-100).
        plot (bool): If True, displays the plot. Otherwise, only returns the start and end times.
        save_plot (bool): If True, saves the plot to a file.
        filename (str): Name of the file to save the plot. Default is "plot.png".
        silent (bool): If True, suppresses all print messages. Default is False.

    Returns:
        list of tuples: Each tuple contains the adjusted start and end times for each dataset.
    """

    if not (0 <= mixing <= 100):
        raise ValueError("Mixing percentage must be between 0 and 100.")

    num = len(datasets)
    durations = [df['frame.time'].max() - df['frame.time'].min() for df in datasets]
    shifts = [durations[i] * (1 - mixing / 100) for i in range(num - 1)]
    current = 0
    times = []

    if plot or save_plot:
        plt.figure(figsize=(12, num * 0.8))
        colors = [
            'red', 'blue', 'lime', 'fuchsia', 'black',
            'grey', 'gold', 'pink', 'darkorange', 'green',
            'darkorchid', 'darkturquoise', 'goldenrod', 'wheat', 'lightcoral',
            'lightgray', 'turquoise', 'firebrick', 'navy', 'lightgreen',
            'beige', 'deeppink', 'lightblue'
        ]

    for i, df in enumerate(datasets):
        length = df['frame.time'].max() - df['frame.time'].min()
        start = current
        end = start + length
        times.append((start, end))

        if plot or save_plot:
            color = colors[i % len(colors)]  # loop colors if more datasets than colors
            plt.hlines(num - 1 - i, start, end, color=color, lw=15)

        if i < len(shifts):
            current += shifts[i]

    if plot or save_plot:
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel("Time (s)")
        plt.yticks(range(num), [f"Dataset {i+1}" for i in range(num)][::-1])
        plt.title(f"Normalized and Shifted Time Ranges of Datasets with Mixing {mixing}%")
        if save_plot:
            plt.savefig(filename, bbox_inches='tight')
            print(f"Plot saved as '{filename}'")
        if plot:
            plt.show()
        else:
            plt.close()

    return times


def mix_datasets(dfs, time_ranges):
    """Shift each df by its start time, tag, concatenate, and sort by time."""
    aligned = []
    for i, (df, (start, _)) in enumerate(zip(dfs, time_ranges)):
        df2 = df.copy()
        df2['dataset_id'] = i + 1
        init = df2['frame.time'].iloc[0]
        df2['frame.time'] = df2['frame.time'] - init + start
        aligned.append(df2)
    merged = pd.concat(aligned).sort_values('frame.time').reset_index(drop=True)
    return merged


def add_flag_column(df, column_name="tcp.flags.str"):
    """Add a binary flag for SYN/ACK or SYN-only packets."""
    df[column_name] = df[column_name].astype(str)
    substr = ['·······A··S·', '··········S·']
    df['S_AS_flag'] = df[column_name].apply(lambda x: int(any(s in x for s in substr)))
    return df


def post_process_mix(df):
    """Merge IPs, add send flag, and factorize IPs for the mixed DataFrame."""
    df = merge_ips_and_create_send_column(df, start_str="172.17.")
    df = add_flag_column(df)
    df['ip.factorized'] = pd.factorize(df['ip'])[0]
    return df


def generate_mixed_dataset_within_time(data_dir, num_tasks, time_extension,
                                       mixing, total_times, save_path,
                                       max_iter, silent=True, plot_result=True):
    """
    Create a single mixed dataset of num_tasks within time_extension seconds,
    with a given mixing percentage and up to max_iter sampling attempts.
    """
    results_dir = os.path.join(save_path, "results", "individual_mix_plots")
    os.makedirs(results_dir, exist_ok=True)

    # Compute target per-task duration adjusted for overlap
    ATPT = time_extension / (1 + (1 - mixing/100)*(num_tasks - 1))
    margin = 0.06 + 0.04 * ATPT
    mins, maxs = ATPT - margin, ATPT + margin
    candidates = [i for i, t in enumerate(total_times) if mins <= t <= maxs]
    if len(candidates) < num_tasks:
        print(f"[WARN] Skipping {num_tasks} tasks @ {mixing}%: only {len(candidates)} candidates in range [{mins:.2f}, {maxs:.2f}]")
        return

    # Map indices to actual folder names
    all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    sel_dirs = [all_dirs[i] for i in candidates]

    # Iteratively sample until the mix duration meets the target or we reach max_iter
    iteration, end_time = 0, 0
    while end_time < time_extension and iteration <= max_iter:
        dfs, _ = sample_and_preprocess(data_dir, sel_dirs, num_tasks)
        times = pre_mixing_superpositions_automatic(dfs, mixing, plot=False, save_plot=False, silent=True)
        end_time = times[-1][1]
        iteration += 1
        if not silent:
            print(f"Iter {iteration}: end_time={end_time:.2f}")
    if iteration > max_iter:
        print(f"[WARN] Max iterations ({max_iter}) reached for {num_tasks} tasks @ {mixing}%.")
        return

    # Final mix & save
    dfs, _ = sample_and_preprocess(data_dir, sel_dirs, num_tasks)
    plot_path = os.path.join(results_dir, f"{num_tasks}t_{time_extension}s_{mixing}m.jpg")
    times = pre_mixing_superpositions_automatic(
        dfs, mixing, plot=plot_result, save_plot=True,
        filename=plot_path, silent=True
    )
    mix = mix_datasets(dfs, times)
    mix = post_process_mix(mix)
    mix = mix[mix['frame.time'] <= time_extension]

    # Log plot path
    print(f"[INFO] Saved mix plot at: {plot_path}")

    # Save individual mix
    individual_dir = os.path.join(save_path, "results", "individual_mixings")
    os.makedirs(individual_dir, exist_ok=True)

    pkl_file = os.path.join(individual_dir, f"{num_tasks}t_{time_extension}s_{mixing}m.pkl")
    with open(pkl_file, 'wb') as f:
        pickle.dump(mix, f)

    return mix


def generate_all_mixed_datasets(raw_data_dir, config, save_path):
    """
    Compute dataset durations and generate all mixed datasets per configuration.
    """
    # Find all CSV file paths under each subdirectory
    csv_paths = [os.path.join(raw_data_dir, d, 'data.csv')
                 for d in os.listdir(raw_data_dir)
                 if os.path.isfile(os.path.join(raw_data_dir, d, 'data.csv'))]

    # Fast duration calculation
    total_times = fast_calculate_total_times(csv_paths)

    # Loop over number of tasks and mixing levels
    all_mixed = []
    for n in tqdm(range(config['min_tasks'], config['max_tasks'] + 1),
                  desc="Generating mixed datasets", unit="task-count"):
        level_mixed = []
        for mix in range(config['min_mixing'], config['max_mixing'] + config['by_mixing'], config['by_mixing']):
            ds = generate_mixed_dataset_within_time(
                data_dir=raw_data_dir,
                num_tasks=n,
                time_extension=config['time_extension'],
                mixing=mix,
                total_times=total_times,
                save_path=save_path,
                max_iter=config['max_iter'],
                silent=True
            )
            level_mixed.append(ds)
        all_mixed.append(level_mixed)

    # Persist overall results
    with open(os.path.join(save_path, "mixed_datasets.pkl"), 'wb') as f:
        pickle.dump(all_mixed, f)
    return all_mixed


def main():
    parser = argparse.ArgumentParser(
        description="Generate mixed datasets within a specified time extension.")
    parser.add_argument('--raw-data-dir', required=True,
                        help='Directory with filtered raw data folders.')
    parser.add_argument('--save-path', required=True,
                        help='Directory to save mixed datasets and outputs.')
    parser.add_argument('--config', required=True,
                        help='Path to JSON config file with generation parameters.')
    args = parser.parse_args()

    with open(args.config) as cf:
        config = json.load(cf)

    # Seed for reproducibility
    seed = config.get('seed', 1)
    random.seed(seed)

    os.makedirs(args.save_path, exist_ok=True)

    # Run full generation pipeline
    generate_all_mixed_datasets(
        raw_data_dir=args.raw_data_dir,
        config=config,
        save_path=args.save_path
    )
    print(f"Mixed datasets generated under {args.save_path}")


if __name__ == '__main__':
    main()
