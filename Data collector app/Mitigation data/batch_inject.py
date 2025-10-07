#!/usr/bin/env python3
"""
batch_inject.py

Wrapper to apply inject_frames. For a given user folder Users/<user>/Preprocessed,
reads all traces_*.csv, applies augment_trace (in memory) and writes results to
Users/<user>_Injected_<injections>/Preprocessed/traces_*.csv

Usage example:
  python batch_inject.py User --users-dir ./Users --injections 5 --interval 3

Requirements:
  - inject_frames.py accessible (same folder or pass its path via --injector)
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Callable

import pandas as pd


def load_augment_function(injector_path: Path) -> Callable:
    """Dynamically import augment_trace from the provided inject_frames.py path."""
    spec = importlib.util.spec_from_file_location("inject_frames_module", str(injector_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {injector_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "augment_trace"):
        raise AttributeError(f"Module {injector_path} does not expose 'augment_trace'")
    return getattr(mod, "augment_trace")


def find_trace_files(preprocessed_dir: Path) -> list[Path]:
    """Return a sorted list of files matching traces_*.csv inside preprocessed_dir."""
    return sorted(preprocessed_dir.glob("traces_*.csv"))


def process_user(
    user_name: str,
    users_dir: Path,
    inject_fn: Callable,
    injections: int,
    interval_min: int,
    injector_path: Path,
    dry_run: bool = False,
):
    user_dir = users_dir / user_name
    preprocessed_dir = user_dir / "Preprocessed"
    if not preprocessed_dir.exists() or not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

    trace_files = find_trace_files(preprocessed_dir)
    if not trace_files:
        print(f"[!] No traces_*.csv files found in {preprocessed_dir}")
        return

    out_user_dir = users_dir / f"{user_name}_Injected_{injections}"
    out_preprocessed_dir = out_user_dir / "Preprocessed"

    print(f"[+] Found {len(trace_files)} files to process for user '{user_name}'.")
    print(f"[+] Injector module: {injector_path}")
    print(f"[+] injections per interval: {injections}, interval_min: {interval_min}")
    print(f"[+] Output directory will be: {out_preprocessed_dir}")
    if dry_run:
        print("[DRY RUN] No files will be written.")
    else:
        out_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    for in_path in trace_files:
        print(f"    -> Processing {in_path.name} ...", end=" ", flush=True)
        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            print(f"SKIP (read error: {e})")
            continue

        # Call the augment function (it returns a DataFrame)
        try:
            df_aug = inject_fn(df, interval_min=interval_min, injections=injections)
        except Exception as e:
            print(f"SKIP (augment error: {e})")
            continue

        out_path = out_preprocessed_dir / in_path.name
        if dry_run:
            print(f"WOULD WRITE -> {out_path}")
        else:
            try:
                df_aug.to_csv(out_path, index=False)
                print(f"OK -> written to {out_path}")
            except Exception as e:
                print(f"ERROR writing: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch inject handshakes into traces for a user.")
    parser.add_argument("user", help="User folder name (e.g., Alice)")
    parser.add_argument("--users-dir", type=Path, default=Path("./Users"), help="Base Users directory (default: ./Users)")
    parser.add_argument("--injector", type=Path, default=Path("./inject_frames.py"), help="Path to inject_frames.py (default: ./inject_frames.py)")
    parser.add_argument("--injections", "-n", type=int, default=1, help="Number of handshake groups to inject per interval (default: 1)")
    parser.add_argument("--interval", "-t", type=int, default=3, help="Interval length in minutes (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files; show what would be done")
    args = parser.parse_args()

    injector_path = args.injector.resolve()
    if not injector_path.exists():
        print(f"[!] injector script not found: {injector_path}", file=sys.stderr)
        sys.exit(2)

    try:
        augment_fn = load_augment_function(injector_path)
    except Exception as e:
        print(f"[!] Failed to import augment_trace from {injector_path}: {e}", file=sys.stderr)
        sys.exit(3)

    try:
        process_user(
            user_name=args.user,
            users_dir=args.users_dir.resolve(),
            inject_fn=augment_fn,
            injections=args.injections,
            interval_min=args.interval,
            injector_path=injector_path,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[!] Error: {exc}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
