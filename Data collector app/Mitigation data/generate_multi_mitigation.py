#!/usr/bin/env python3
"""
generate_multi_injected.py

Builds `Users/User_Injected_Mix/` by sampling each split_K.csv (K = 1-50)
from one of the existing injected folders specified in SOURCE_NS (e.g., N = 1, 100, 300).
Also writes a decision log `generate_multi_injected.txt`.
"""

import os
import random
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# CONFIGURE THESE IF YOUR LAYOUT DIFFERS
BASE_DIR = Path("Users")                       # Folder that holds everything
SOURCE_NS = [1, 600, 1000]                      # Allowed N values (adjust as needed)
TARGET_DIR = BASE_DIR / "User_Injected_Mix" / "Splits" / "5.0s"
LOG_FILE   = TARGET_DIR.parent / "generate_multi_injected.txt"
K_RANGE    = range(1, 51)                      # K = 1 … 50
# --------------------------------------------------------------------------- #

def main(seed: int | None = None) -> None:
    """Copy each split_K.csv into the Injected Mix folder and record the mapping."""
    if seed is not None:
        random.seed(seed)                      # Optional reproducibility

    # 1. Ensure target folders exist
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Process every K
    decisions: list[str] = []
    for k in K_RANGE:
        n = random.choice(SOURCE_NS)           # Pick the source N

        src = BASE_DIR / f"User_Injected_{n}" / "Splits" / "5.0s" / f"split_{k}.csv"
        dst = TARGET_DIR / f"split_{k}.csv"

        if not src.exists():
            print(f"[WARN] Missing source file: {src}", file=sys.stderr)
            continue                           # or `raise` if you want it to stop

        shutil.copy2(src, dst)
        decisions.append(f"Split {k} -> {n} injections (per 5.0s)")

    # 3. Write log file
    LOG_FILE.write_text("\n".join(decisions), encoding="utf-8")
    print(f"Done! Mixed splits in: {TARGET_DIR}")
    print(f"Decision log:          {LOG_FILE}")

if __name__ == "__main__":
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed_arg)
