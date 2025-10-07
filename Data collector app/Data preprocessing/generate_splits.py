#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge activity CSVs and split into fixed-duration segments, saving splits organized by interval."
    )
    parser.add_argument(
        '--users',
        nargs='+',
        required=True,
        help='List of user folder names to process (e.g. User1 User2)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        required=True,
        help='Interval size used in activity folder naming (in seconds, e.g. 0.5)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        required=True,
        help='Duration of each split segment (in minutes)'
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    users_dir = base_dir / 'Users'

    duration_sec = args.duration * 60.0
    interval_str = f"{args.interval}s"

    for user in args.users:
        activity_dir = users_dir / user / 'Activity' / interval_str
        if not activity_dir.exists():
            print(f"[SKIP] {user} -- Activity folder not found for interval {interval_str}")
            continue

        # Gather and sort activity files by index
        files = sorted(
            activity_dir.glob('activity_*.csv'),
            key=lambda p: int(p.stem.split('_')[1])
        )
        if not files:
            print(f"[SKIP] {user} -- no CSVs in {activity_dir}")
            continue

        # Merge with continuous time
        merged = []
        offset = 0.0
        for file_path in files:
            df = pd.read_csv(file_path)
            # shift time
            df['time'] = df['time'] + offset
            merged.append(df)
            # update offset: last time + interval
            last_time = df['time'].iloc[-1]
            offset = last_time + args.interval
        merged_df = pd.concat(merged, ignore_index=True)

        # Create splits directory for this interval
        splits_dir = users_dir / user / 'Splits' / interval_str
        splits_dir.mkdir(parents=True, exist_ok=True)

        # Determine number of splits
        max_time = merged_df['time'].max()
        n_splits = int(np.ceil(max_time / duration_sec))

        # Write each split
        for i in range(n_splits):
            start_t = i * duration_sec
            end_t = (i + 1) * duration_sec
            if i < n_splits - 1:
                split_df = merged_df[(merged_df['time'] >= start_t) & (merged_df['time'] < end_t)]
            else:
                split_df = merged_df[merged_df['time'] >= start_t]

            out_path = splits_dir / f'split_{i+1}.csv'
            split_df.to_csv(out_path, index=False)
            print(f"[SAVED] {user}: {out_path}")

if __name__ == '__main__':
    main()
