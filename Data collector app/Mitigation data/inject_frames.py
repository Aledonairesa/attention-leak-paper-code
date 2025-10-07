#!/usr/bin/env python3
"""inject_frames.py – Augment a Wi-Fi/PCAP-derived CSV with artificial TCP 3-way-handshake frames

This utility adds groups of statistically plausible synthetic frames (SYN, SYN-ACK, ACK)
to subsequent intervals. Each group is sampled from statistics of the previous interval.
Modifications:
- Exclude DNS resolver IPs and port 53 from sampling.
- Frame 2 uses swapped ports from Frame 1 (no new port sampling).
- Add CLI option `--injections` to control how many handshake groups (each = 3 frames)
  to inject per processed interval.
- Informative prints for intervals processed and frames injected.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ────────────────────────────────────────────────────────────────────────────────
FLAG_SYN = "··········S·"
FLAG_SYN_ACK = "·······A··S·"
FLAG_ACK = "·······A····"
IP_PROTO_TCP = 6  # stored as int, will be converted to float if dataframe uses


# ----------------------------------------------------------------------------
# Core utility functions
# ----------------------------------------------------------------------------

def _discrete_sample(series: pd.Series) -> any:
    """Sample a value proportional to its frequency in *series*.

    Returns
    -------
    any
        One element from ``series`` (even if the series is already categorical).
    """
    if series.empty:
        raise ValueError("Cannot sample from empty series")
    counts = series.value_counts(normalize=True)
    return np.random.choice(counts.index, p=counts.values)


def _identify_resolvers(df_interval: pd.DataFrame) -> set[str]:
    """Return a set of IPs acting as DNS resolvers in the interval (port 53 or dns.a)."""
    resolvers = set()
    # If either port.src or port.dst is 53, that endpoint is a resolver
    if "port.src" in df_interval and "ip.src" in df_interval:
        resolvers.update(df_interval.loc[df_interval["port.src"] == 53, "ip.src"].astype(str).unique())
    if "port.dst" in df_interval and "ip.dst" in df_interval:
        resolvers.update(df_interval.loc[df_interval["port.dst"] == 53, "ip.dst"].astype(str).unique())
    # If dns.a is not null, ip.src is resolver
    if "dns.a" in df_interval:
        ip_srcs = df_interval.loc[df_interval["dns.a"].notna(), "ip.src"].astype(str).unique()
        resolvers.update(ip_srcs)
    return resolvers


def _sample_four_tuple(df_interval: pd.DataFrame, resolvers: set[str]) -> tuple[str, str, int, str]:
    """Sample (ip.src, ip.dst, send, ip) tuple excluding resolver IPs and port 53.
    """
    s_frames = df_interval[df_interval["tcp.flags.str"] == FLAG_SYN]
    if s_frames.empty:
        raise ValueError("Interval has no SYN frames; cannot build distribution")

    # Filter out any rows where ip.src or ip.dst is a resolver
    s_frames = s_frames[~s_frames["ip.src"].astype(str).isin(resolvers)]
    s_frames = s_frames[~s_frames["ip.dst"].astype(str).isin(resolvers)]

    if s_frames.empty:
        # Fallback to original SYN frames if all filtered out
        s_frames = df_interval[df_interval["tcp.flags.str"] == FLAG_SYN]

    # Build counts of unique 4-tuples
    tuples = s_frames[["ip.src", "ip.dst", "send", "ip"]].astype(str)
    tuple_counts = tuples.value_counts()

    # Exclude the most frequent (likely resolver tuple) if possible
    if len(tuple_counts) > 1:
        tuple_counts = tuple_counts.iloc[1:]

    if tuple_counts.empty:
        tuple_counts = tuples.value_counts()

    probs = tuple_counts / tuple_counts.sum()
    choice = np.random.choice(probs.index, p=probs.values)
    return tuple(choice)


def _sample_frame_length(df_interval: pd.DataFrame, flag: str) -> int:
    """Sample a frame length conditioned on TCP flag *flag* in the interval."""
    subset = df_interval[df_interval["tcp.flags.str"] == flag]
    if subset.empty:
        subset = df_interval  # graceful fallback
    return int(_discrete_sample(subset["frame.len"]))


def _sample_port(df_interval: pd.DataFrame, flag: str, port_col: str, send_val: int, resolvers: set[str]) -> int:
    """Sample source/destination port for *send_val* given *flag*, excluding port 53 if possible."""
    subset = df_interval[
        (df_interval["tcp.flags.str"] == flag) & (df_interval["send"] == send_val)
    ]
    # Exclude port 53 from sampling
    if port_col in subset.columns:
        subset = subset[subset[port_col] != 53]
    if subset.empty:
        # Relax send condition, but still exclude 53 if possible
        subset = df_interval[df_interval["tcp.flags.str"] == flag]
        if port_col in subset.columns:
            subset = subset[subset[port_col] != 53]
    if subset.empty:
        # Fallback to all rows excluding port 53 if possible
        if port_col in df_interval.columns:
            subset = df_interval[df_interval[port_col] != 53]
        else:
            subset = df_interval
    if subset.empty:
        # Finally fallback to anything
        subset = df_interval
    # pick the port column if present, else raise if absent
    if port_col not in subset.columns:
        # As a last resort, try port.src or port.dst generically
        for alt in ("port.src", "port.dst"):
            if alt in subset.columns:
                return int(_discrete_sample(subset[alt]))
        raise ValueError(f"Port column {port_col} not found in dataframe")
    return int(_discrete_sample(subset[port_col]))


def _generate_handshake(df_interval: pd.DataFrame) -> list[dict]:
    """Generate three synthetic frames (SYN, SYN-ACK, ACK) from statistics of *df_interval*."""
    # Identify resolver IPs in this interval
    resolvers = _identify_resolvers(df_interval)

    ip_src, ip_dst, send_val, ip_version = _sample_four_tuple(df_interval, resolvers)

    # Frame 1 – SYN
    port_src_1 = _sample_port(df_interval, FLAG_SYN, "port.src", int(send_val), resolvers)
    port_dst_1 = _sample_port(df_interval, FLAG_SYN, "port.dst", int(send_val), resolvers)
    frame1 = {
        "ip.src": ip_src,
        "ip.dst": ip_dst,
        "send": int(send_val),
        "ip": ip_version,
        "ip.proto": IP_PROTO_TCP,
        "tcp.flags.str": FLAG_SYN,
        "frame.len": _sample_frame_length(df_interval, FLAG_SYN),
        "dns.qry.name": np.nan,
        "dns.a": np.nan,
        "port.src": port_src_1,
        "port.dst": port_dst_1,
    }

    # Frame 2 – SYN-ACK (flip directions and swap ports from frame1)
    frame2 = frame1.copy()
    frame2.update({
        "ip.src": frame1["ip.dst"],
        "ip.dst": frame1["ip.src"],
        "send": 1 - frame1["send"],
        "tcp.flags.str": FLAG_SYN_ACK,
        "frame.len": _sample_frame_length(df_interval, FLAG_SYN_ACK),
        # Swap ports from frame1
        "port.src": frame1["port.dst"],
        "port.dst": frame1["port.src"],
    })

    # Frame 3 – ACK (same direction as frame1, same ports)
    frame3 = frame1.copy()
    frame3.update({
        "tcp.flags.str": FLAG_ACK,
        "frame.len": _sample_frame_length(df_interval, FLAG_ACK),
    })

    return [frame1, frame2, frame3]


def _choose_insertion_times(
    start_ts: float, T_sec: float, frame_diffs: np.ndarray
) -> list[float]:
    """Return 3 ascending time-stamps inside the interval [start_ts, start_ts + T_sec).

    We pick a random base such that all three frames fit.
    """
    total_span = frame_diffs.sum()
    if total_span >= T_sec:
        # unrealistically large diffs; shrink proportionally
        frame_diffs = frame_diffs * (T_sec * 0.8 / total_span)
        total_span = frame_diffs.sum()

    base_offset = random.uniform(0, T_sec - total_span)
    t1 = start_ts + base_offset
    t2 = t1 + frame_diffs[0]
    t3 = t2 + frame_diffs[1]
    return [t1, t2, t3]


# ────────────────────────────────────────────────────────────────────────────────
# Main augmentation routine
# ────────────────────────────────────────────────────────────────────────────────

def augment_trace(df: pd.DataFrame, interval_min: int = 3, injections: int = 1) -> pd.DataFrame:
    """Return a new DataFrame with synthetic handshake frames inserted.

    Parameters
    ----------
    df : pd.DataFrame
        The original trace dataframe.
    interval_min : int
        Interval length in minutes used to compute intervals.
    injections : int
        Number of handshake groups (each group = 3 frames) to inject per processed interval.
    """
    print(f"[*] Starting augmentation with {interval_min}-minute intervals, injections per interval = {injections}...")
    T_sec = interval_min * 60

    # Prepare interval mapping
    df_sorted = df.sort_values("frame.time").reset_index(drop=True)
    df_sorted["interval_idx"] = (df_sorted["frame.time"] // T_sec).astype(int)

    # Pre-compute global distribution of frame-time diffs (excluding zeros)
    global_diffs = df_sorted["frame.time"].diff().dropna()
    global_diffs = global_diffs[global_diffs > 0].to_numpy()

    new_rows: list[dict] = []
    intervals_processed = 0
    frames_injected = 0
    attempted_injections_total = 0
    successful_injections_total = 0

    max_interval = int(df_sorted["interval_idx"].max())
    for i in range(1, max_interval):  # skip first and last
        df_i = df_sorted[df_sorted["interval_idx"] == i]
        if df_i.empty:
            continue

        intervals_processed_here = 0
        successful_injections_here = 0

        # For each requested injection, try to build + insert a handshake
        for inj_idx in range(injections):
            attempted_injections_total += 1
            try:
                synthetic_frames = _generate_handshake(df_i)
            except Exception:
                # skip this injection if generation fails for any reason
                continue

            if len(synthetic_frames) != 3:
                continue

            if len(global_diffs) == 0:
                # cannot sample time diffs; skip insertion
                continue
            sampled_diffs = np.random.choice(global_diffs, size=3, replace=True)

            start_ts = (i + 1) * T_sec
            times = _choose_insertion_times(start_ts, T_sec, sampled_diffs)

            for frame_dict, ts in zip(synthetic_frames, times):
                frame_dict["frame.time"] = ts
                new_rows.append(frame_dict)

            successful_injections_here += 1
            successful_injections_total += 1
            frames_injected += 3
            intervals_processed_here = 1  # mark that this interval yielded at least one injection

        if successful_injections_here > 0:
            print(f"    [+] Interval {i} → injected {3 * successful_injections_here} frames ({successful_injections_here} handshakes) into interval {i+1}")
            intervals_processed += 1

    print(f"[*] Completed intervals processed: {intervals_processed}")
    print(f"[*] Attempted injections: {attempted_injections_total}, successful injections: {successful_injections_total}, total fake frames added: {frames_injected}")

    if not new_rows:
        return df_sorted.drop(columns=["interval_idx"])

    df_new = pd.DataFrame(new_rows)

    combined = pd.concat([df_sorted.drop(columns=["interval_idx"]), df_new], ignore_index=True)
    combined = combined.sort_values("frame.time").reset_index(drop=True)

    # Renumber frames starting at 1
    combined["frame.number"] = np.arange(1, len(combined) + 1)

    return combined


# ────────────────────────────────────────────────────────────────────────────────
# CLI wrapper
# ────────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Augment trace CSV with synthetic TCP handshake frames.")
    parser.add_argument("input", type=Path, help="Input CSV file (e.g., traces_1.csv)")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV path (default: <input>_aug.csv)")
    parser.add_argument("-t", "--interval", type=int, default=5, help="Interval length in minutes")
    parser.add_argument("-n", "--injections", type=int, default=1, help="Number of handshake groups (each = 3 frames) to inject per processed interval")

    args = parser.parse_args()

    output_path = args.output if args.output else args.input.with_name(args.input.stem + "_aug.csv")

    print(f"[+] Loading {args.input} …")
    df = pd.read_csv(args.input)

    print("[+] Generating synthetic frames …")
    df_aug = augment_trace(df, interval_min=args.interval, injections=args.injections)

    print(f"[+] Writing augmented trace to {output_path} …")
    df_aug.to_csv(output_path, index=False)
    print("[✓] Done.")


if __name__ == "__main__":
    _cli()
