import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.utils_activity import build_activity_df

def main():
    parser = argparse.ArgumentParser(
        description="Build activity tables from preprocessed traces and timestamps CSVs."
    )
    parser.add_argument(
        '--users',
        nargs='+',
        required=True,
        help='List of user folder names to process (e.g. User1 User2)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        type=int,
        required=True,
        help='List of file indices to process (e.g. 1 2 3)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        required=True,
        help='Time interval (in seconds) for activity bins (e.g. 0.5)'
    )
    args = parser.parse_args()

    # Base directory assumed to contain 'Users' folder
    base_dir = Path(__file__).resolve().parent
    users_dir = base_dir / 'Users'

    for user in args.users:
        preproc_dir = users_dir / user / 'Preprocessed'
        if not preproc_dir.exists():
            print(f"[SKIP] {user} -- Preprocessed folder not found")
            continue

        # Create output folder: Users/<user>/Activity/<interval>s/
        interval_str = f"{args.interval}s"
        activity_dir = users_dir / user / 'Activity' / interval_str
        activity_dir.mkdir(parents=True, exist_ok=True)

        for idx in args.files:
            traces_file = preproc_dir / f'traces_{idx}.csv'
            timestamps_file = preproc_dir / f'timestamps_{idx}.csv'

            if not traces_file.exists() or not timestamps_file.exists():
                print(f"[SKIP] {user} - missing preprocessed traces_{idx}.csv or timestamps_{idx}.csv")
                continue

            print(f"[PROCESS] {user} - index {idx} @ interval {args.interval}s")
            # Read preprocessed data
            frames_df = pd.read_csv(traces_file)
            ts_df = pd.read_csv(timestamps_file)

            # Determine start and end times
            start_time = 0.0
            last_frame = frames_df['frame.time'].iloc[-1]
            last_ts = ts_df['time'].iloc[-1]
            end_time = float(min(last_frame, last_ts))

            # Build activity DataFrame
            activity_df = build_activity_df(
                timestamps_df=ts_df,
                frames_df=frames_df,
                start_time=start_time,
                end_time=end_time,
                time_interval=args.interval
            )

            # Save
            out_file = activity_dir / f'activity_{idx}.csv'
            activity_df.to_csv(out_file, index=False)
            print(f"[SAVED] {out_file}")

if __name__ == '__main__':
    main()
