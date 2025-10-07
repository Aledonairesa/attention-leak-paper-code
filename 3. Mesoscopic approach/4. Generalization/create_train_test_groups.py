import argparse
import random
import math
from pathlib import Path

def compress_indices(indices):
    """
    Turn a sorted list of ints into a string like "1-3,5,7-9".
    """
    if not indices:
        return ""
    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            # close off previous run
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = idx
    # last run
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")
    return ",".join(ranges)

def main():
    parser = argparse.ArgumentParser(
        description="Split CSV file indices into 70/30 groups, excluding holdouts."
    )
    parser.add_argument(
        "--total",
        type=int,
        default=50,
        help="Total number of CSV files (default: 50)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.7,
        help="Fraction for the first split (default: 0.7 for 70%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (optional, for reproducibility)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_test_groups.txt"),
        help="Path to write the split indices (default: splits.txt)"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # 1) build available indices
    all_idxs = set(range(1, args.total + 1))
    available = sorted(all_idxs)
    n = len(available)

    # 2) determine split sizes
    first_size = int(math.floor(n * args.ratio))

    # 3) sample
    first_group = sorted(random.sample(available, first_size))
    second_group = sorted(set(available) - set(first_group))

    # 4) compress to ranges
    first_str = compress_indices(first_group)
    second_str = compress_indices(second_group)

    # 5) write out
    with open(args.output, 'w') as f:
        f.write(f"{int(args.ratio*100)}: {first_str}\n")
        f.write(f"{100-int(args.ratio*100)}: {second_str}\n")

    print(f"Wrote splits to {args.output}")

if __name__ == "__main__":
    main()
