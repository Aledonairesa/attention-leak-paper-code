import argparse
import os
import re
import shutil
from pathlib import Path
import pandas as pd
from collections import Counter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter dataset folders by CSV row count and copy qualifying folders."
    )
    parser.add_argument(
        "datasets_path",
        type=Path,
        help="Path to the root directory containing dataset subfolders."
    )
    parser.add_argument(
        "rows_threshold",
        type=int,
        help="Minimum number of rows in data.csv to consider a dataset good."
    )
    parser.add_argument(
        "min_num_good_datasets",
        type=int,
        help="Minimum number of good datasets per website to include that website."
    )
    parser.add_argument(
        "joined_new_path",
        type=Path,
        help="Output path where filtered folders will be copied."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    datasets_path = args.datasets_path
    rows_threshold = args.rows_threshold
    min_num_good_datasets = args.min_num_good_datasets
    output_dir = args.joined_new_path

    # Gather CSV paths
    csv_paths = [
        datasets_path / run_folder / "data.csv"
        for run_folder in os.listdir(datasets_path)
    ]

    # Filter datasets by row count
    good_csv_paths = []
    for csv_path in tqdm(csv_paths, desc="Checking CSVs", unit="dataset"):
        if not csv_path.exists():
            tqdm.write(f"Warning: {csv_path} does not exist.")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            tqdm.write(f"Warning: failed to read {csv_path}: {e}")
            continue
        if len(df) > rows_threshold:
            good_csv_paths.append(csv_path)

    if not good_csv_paths:
        print("No datasets exceed the row threshold. Exiting.")
        return

    # Extract website identifiers
    websites = []
    pattern = re.compile(r"www\.([^-\\/]+)")
    for path in good_csv_paths:
        match = pattern.search(str(path))
        if match:
            websites.append(match.group(1))

    freq = Counter(websites)
    good_websites = [
        site
        for site, count in freq.items()
        if count > min_num_good_datasets
    ]

    if not good_websites:
        print("No websites have more than the minimum number of good datasets. Exiting.")
        return

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = set()

    # Copy qualifying folders
    for csv_path in tqdm(good_csv_paths, desc="Copying folders", unit="dataset"):
        parent = csv_path.parent
        if any(site in str(parent) for site in good_websites):
            if parent in copied:
                continue
            dest = output_dir / parent.name
            if dest.exists():
                tqdm.write(f"Skipping existing folder: {dest}")
            else:
                shutil.copytree(parent, dest)
            copied.add(parent)

    print(f"Copied {len(copied)} folders to {output_dir}")

if __name__ == "__main__":
    main()
