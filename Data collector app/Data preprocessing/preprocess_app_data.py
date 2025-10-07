import argparse
from pathlib import Path

from utils.utils_preprocessing import *

def read_and_preprocess_data(
        frames_path: str,
        timestamps_path: str,
        time_diff_threshold: float = 0.2
    ):
    """
    Load and preprocess network frames and window-focus timestamps.

    Parameters:
        frames_path (str): Path to the frames CSV file.
        timestamps_path (str): Path to the timestamps CSV file.
        time_diff_threshold (float): Minimum time delta to keep consecutive timestamp rows.

    Returns:
        tuple: (preprocessed_frames_df, preprocessed_timestamps_df)
    """

    # Frames preprocessing:
    # 1. Read and clean CSV
    # 2. Parse time and convert to UTC
    # 3. Infer host IP and mark outgoing packets
    # 4. Extract peer IPs
    # 5. Drop rows with missing or multicast IPs
    # 6. Merge TCP/UDP ports
    print("FRAMES: loading and preprocessing...")
    raw_frames = read_frames_csv(frames_path)
    frames_tz   = infer_frames_tz(raw_frames)
    frames_df = read_and_preprocess_frames(frames_path)
    print(f"Frames done. Final shape: {frames_df.shape}")

    # Timestamps preprocessing:
    # 1. Parse lines and clean empty messages
    # 2. Convert time to UTC
    # 3. Remove spurious messages
    # 4. Filter small time differences and duplicates
    print("\nTIMESTAMPS: loading and preprocessing...")
    ts_df = read_and_preprocess_timestamps(
        timestamps_path,
        timezone=frames_tz,
        time_diff_threshold=time_diff_threshold
    )
    print(f"Timestamps done. Final shape: {ts_df.shape}")

    return frames_df, ts_df

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess network frame and timestamp CSV files for selected users and file indices."
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
        help='List of file indices to process (e.g. 1 2 3 for traces_1.csv, timestamps_1.csv, etc.)'
    )
    args = parser.parse_args()

    # Assume script is run from the directory containing the 'Users' folder
    base_dir = Path(__file__).resolve().parent
    users_dir = base_dir / 'Users'

    for user in args.users:
        raw_dir = users_dir / user / 'Raw'
        preproc_dir = users_dir / user / 'Preprocessed'
        preproc_dir.mkdir(parents=True, exist_ok=True)

        for idx in args.files:
            traces_path = raw_dir / f'traces_{idx}.csv'
            timestamps_path = raw_dir / f'timestamps_{idx}.csv'

            if not traces_path.exists() or not timestamps_path.exists():
                print(f"[SKIP] {user} - missing traces_{idx}.csv or timestamps_{idx}.csv")
                continue

            print(f"[PROCESS] {user} - file index {idx}")
            # Load and preprocess
            frames_df, ts_df = read_and_preprocess_data(
                str(traces_path),
                str(timestamps_path)
            )

            # Normalize time columns across both dataframes
            min_frame_time = frames_df['frame.time'].min()
            min_ts_time = ts_df['time'].min()
            global_min = min(min_frame_time, min_ts_time)

            frames_df['frame.time'] = frames_df['frame.time'] - global_min
            ts_df['time'] = ts_df['time'] - global_min

            # Save preprocessed CSVs
            out_traces = preproc_dir / f'traces_{idx}.csv'
            out_timestamps = preproc_dir / f'timestamps_{idx}.csv'

            frames_df.to_csv(out_traces, index=False)
            ts_df.to_csv(out_timestamps, index=False)

            print(f"[SAVED] {out_traces} and {out_timestamps}")


if __name__ == '__main__':
    main()
