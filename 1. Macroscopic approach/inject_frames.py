from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from dataclasses import dataclass

import ipaddress
import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------------------------------
# TCP flag constants
# -------------------------------------------------------------------------------
FLAG_SYN     = "··········S·"
FLAG_SYN_ACK = "·······A··S·"
FLAG_ACK     = "·······A····"
IP_PROTO_TCP = 6


# -------------------------------------------------------------------------------
# Precomputed distributions (built once from the global pool)
# -------------------------------------------------------------------------------

@dataclass
class PoolDistributions:
    """All distributions needed for handshake generation, precomputed as numpy arrays."""
    # (ip, send) tuples sampled from SYN frames
    syn_tuple_values: np.ndarray   # shape (N, 2) — each row is [ip, send]
    syn_tuple_probs:  np.ndarray   # shape (N,)

    # frame.len Gaussian params per flag: (mean, std)
    len_syn_mean:    float
    len_syn_std:     float
    len_synack_mean: float
    len_synack_std:  float
    len_ack_mean:    float
    len_ack_std:     float
    len_min:         int    # global minimum observed frame length
    len_max:         int    # global maximum observed frame length

    # port distributions: srcport and dstport from SYN frames
    srcport_values: np.ndarray
    srcport_probs:  np.ndarray
    dstport_values: np.ndarray
    dstport_probs:  np.ndarray

    # inter-frame time diffs (for timestamp placement)
    global_diffs: np.ndarray


def _build_distribution(series: pd.Series):
    """Return (values_array, probs_array) from a Series, excluding NaNs."""
    counts = series.dropna().value_counts(normalize=True)
    return counts.index.to_numpy(), counts.values.astype(float)


def precompute_distributions(pool: pd.DataFrame) -> PoolDistributions:
    """Compute all sampling distributions once from the global pool."""
    print("[+] Precomputing distributions from global pool ...")

    # --- Identify resolver IPs (port 53) ---
    resolvers = set()
    for col in ("tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"):
        if col in pool.columns:
            resolvers.update(pool.loc[pool[col] == 53, "ip"].astype(str).unique())
    print(f"    Resolver IPs found: {len(resolvers)}")

    # --- Google IP range to exclude ---
    google_start = int(ipaddress.IPv4Address("142.250.0.0"))
    google_end   = int(ipaddress.IPv4Address("142.251.255.255"))

    def _is_google(ip_series: pd.Series) -> pd.Series:
        """Return boolean mask: True where IP falls in 142.250.0.0-142.251.255.255."""
        try:
            ip_ints = ip_series.apply(lambda ip: int(ipaddress.IPv4Address(str(ip))))
            return ip_ints.between(google_start, google_end)
        except Exception:
            return pd.Series(False, index=ip_series.index)

    # --- SYN tuple distribution (ip, send) ---
    syn = pool[pool["tcp.flags.str"] == FLAG_SYN].copy()
    if syn.empty:
        raise ValueError("Global pool has no SYN frames.")
    # Exclude resolver IPs and Google IPs
    google_mask = _is_google(syn["ip"])
    syn_clean = syn[~syn["ip"].astype(str).isin(resolvers) & ~google_mask]
    if syn_clean.empty:
        syn_clean = syn[~google_mask]  # drop Google but keep resolvers as fallback
    if syn_clean.empty:
        syn_clean = syn  # last resort: no filtering
    print(f"    Google IPs excluded from SYN pool: {google_mask.sum()}")
    tuples = syn_clean[["ip", "send"]].astype(str)
    tuple_counts = tuples.value_counts(normalize=True)
    # skip the most frequent tuple if possible
    if len(tuple_counts) > 1:
        tuple_counts = tuple_counts.iloc[1:]
    syn_tuple_values = np.array(tuple_counts.index.tolist())  # (N, 2)
    syn_tuple_probs  = tuple_counts.values.astype(float)
    syn_tuple_probs /= syn_tuple_probs.sum()  # renormalize after possible slice

    # --- Frame length Gaussian params per flag ---
    all_lens = pool["frame.len"].dropna()
    len_min  = int(all_lens.min())
    len_max  = int(all_lens.max())

    def _len_gauss(flag):
        sub = pool[pool["tcp.flags.str"] == flag]["frame.len"].dropna()
        if sub.empty:
            sub = all_lens
        return float(sub.mean()), float(sub.std(ddof=1))

    len_syn_mean,    len_syn_std    = _len_gauss(FLAG_SYN)
    len_synack_mean, len_synack_std = _len_gauss(FLAG_SYN_ACK)
    len_ack_mean,    len_ack_std    = _len_gauss(FLAG_ACK)

    # --- Port distributions from SYN frames, excluding port 53 ---
    def _port_dist(col):
        sub = syn_clean[col].dropna() if col in syn_clean.columns else pd.Series(dtype=float)
        sub = sub[sub != 53]
        if sub.empty:
            # fallback: all SYN frames
            sub = syn[col].dropna() if col in syn.columns else pd.Series(dtype=float)
            sub = sub[sub != 53]
        if sub.empty:
            sub = pool[col].dropna() if col in pool.columns else pd.Series(dtype=float)
        return _build_distribution(sub)

    srcport_v, srcport_p = _port_dist("tcp.srcport")
    dstport_v, dstport_p = _port_dist("tcp.dstport")

    # --- Global inter-frame time diffs ---
    pool_sorted = pool.sort_values("frame.time")
    diffs = pool_sorted["frame.time"].diff().dropna()
    global_diffs = diffs[diffs > 0].to_numpy()

    print(f"    SYN tuple choices:  {len(syn_tuple_values)}")
    print(f"    frame.len SYN:      mean={len_syn_mean:.1f}, std={len_syn_std:.1f}")
    print(f"    frame.len SYN-ACK:  mean={len_synack_mean:.1f}, std={len_synack_std:.1f}")
    print(f"    frame.len ACK:      mean={len_ack_mean:.1f}, std={len_ack_std:.1f}")
    print(f"    frame.len bounds:   [{len_min}, {len_max}]")
    print(f"    srcport choices:    {len(srcport_v)}")
    print(f"    dstport choices:    {len(dstport_v)}")
    print(f"    inter-frame diffs:  {len(global_diffs):,} samples")

    return PoolDistributions(
        syn_tuple_values=syn_tuple_values,
        syn_tuple_probs=syn_tuple_probs,
        len_syn_mean=len_syn_mean,       len_syn_std=len_syn_std,
        len_synack_mean=len_synack_mean, len_synack_std=len_synack_std,
        len_ack_mean=len_ack_mean,       len_ack_std=len_ack_std,
        len_min=len_min,                 len_max=len_max,
        srcport_values=srcport_v,        srcport_probs=srcport_p,
        dstport_values=dstport_v,        dstport_probs=dstport_p,
        global_diffs=global_diffs,
    )


# -------------------------------------------------------------------------------
# Fast handshake generation
# -------------------------------------------------------------------------------

def _sample_len(mean: float, std: float, len_min: int, len_max: int,
                len_spread: float) -> int:
    """Sample a frame length from N(mean, std * len_spread), clamped to [len_min, len_max].

    Parameters
    ----------
    len_spread : float
        Multiplier on the empirical std. 1.0 = original distribution.
        Values > 1.0 produce more extreme (both small and large) frame lengths.
    """
    val = np.random.normal(mean, std * len_spread)
    return int(max(round(val), len_min))


def _generate_handshake(dist: PoolDistributions, len_spread: float = 1.0,
                        fixed_ip: str = None, fixed_send: int = None) -> list:
    """Generate one synthetic TCP handshake using precomputed distributions.

    Parameters
    ----------
    len_spread : float
        Std multiplier for frame length sampling. 1.0 = empirical distribution.
        Higher values generate more extreme frame lengths.
    fixed_ip : str, optional
        If provided, use this IP instead of sampling from the distribution.
    fixed_send : int, optional
        If provided, use this send value instead of sampling.
    """
    if fixed_ip is not None and fixed_send is not None:
        ip_val, send_val = fixed_ip, fixed_send
    else:
        # Sample (ip, send) tuple
        idx = np.random.choice(len(dist.syn_tuple_values), p=dist.syn_tuple_probs)
        ip_val, send_val = dist.syn_tuple_values[idx]
        send_val = int(send_val)

    src_port = int(np.random.choice(dist.srcport_values, p=dist.srcport_probs))
    dst_port = int(np.random.choice(dist.dstport_values, p=dist.dstport_probs))

    len_syn    = _sample_len(dist.len_syn_mean,    dist.len_syn_std,    dist.len_min, dist.len_max, len_spread)
    len_synack = _sample_len(dist.len_synack_mean, dist.len_synack_std, dist.len_min, dist.len_max, len_spread)
    len_ack    = _sample_len(dist.len_ack_mean,    dist.len_ack_std,    dist.len_min, dist.len_max, len_spread)

    base = {
        "ip":            ip_val,
        "ip.proto":      IP_PROTO_TCP,
        "udp.srcport":   np.nan,
        "udp.dstport":   np.nan,
        "dataset_id":    np.nan,
        "ip.factorized": np.nan,
    }

    frame1 = {**base, "send": send_val,     "tcp.flags.str": FLAG_SYN,
              "frame.len": len_syn,    "tcp.srcport": src_port, "tcp.dstport": dst_port,
              "S_AS_flag": 1}

    frame2 = {**base, "send": 1 - send_val, "tcp.flags.str": FLAG_SYN_ACK,
              "frame.len": len_synack, "tcp.srcport": dst_port, "tcp.dstport": src_port,
              "S_AS_flag": 1}

    frame3 = {**base, "send": 0,           "tcp.flags.str": FLAG_ACK,
              "frame.len": len_ack,    "tcp.srcport": src_port, "tcp.dstport": dst_port,
              "S_AS_flag": 0}

    return [frame1, frame2, frame3]


def _choose_insertion_times(t_min: float, t_max: float,
                             frame_diffs: np.ndarray) -> list:
    """Return 3 ascending timestamps placed randomly within [t_min, t_max)."""
    span = t_max - t_min
    total_diff = frame_diffs.sum()
    if total_diff >= span:
        frame_diffs = frame_diffs * (span * 0.8 / total_diff)
        total_diff = frame_diffs.sum()
    base = random.uniform(t_min, t_max - total_diff)
    return [base, base + frame_diffs[0], base + frame_diffs[0] + frame_diffs[1]]


# -------------------------------------------------------------------------------
# Injection distribution across target IPs
# -------------------------------------------------------------------------------

def _distribute_injections(n_injections: int, n_ips: int) -> list:
    """Randomly partition n_injections across n_ips buckets (all buckets >= 1)."""
    proportions = np.random.dirichlet(np.ones(n_ips))
    counts = np.round(proportions * n_injections).astype(int)
    # Fix rounding so counts sum exactly to n_injections
    diff = n_injections - counts.sum()
    for i in range(abs(diff)):
        counts[i % n_ips] += int(np.sign(diff))
    return counts.tolist()


# -------------------------------------------------------------------------------
# Per-dataset augmentation
# -------------------------------------------------------------------------------

def augment_trace(df: pd.DataFrame,
                  dist: PoolDistributions,
                  injections: int = 1,
                  len_spread: float = 1.0,
                  silent: bool = False) -> pd.DataFrame:
    """Inject `injections` synthetic handshakes into df using precomputed distributions.

    Parameters
    ----------
    df : pd.DataFrame
        One all_mixed[i][j] DataFrame.
    dist : PoolDistributions
        Precomputed distributions from the global pool.
    injections : int
        Number of handshake groups (3 frames each) to inject.
    len_spread : float
        Std multiplier for frame length sampling. 1.0 = empirical distribution.
        Higher values generate more extreme (both small and large) frame lengths.
    silent : bool
        Suppress prints.
    """
    t_min = df["frame.time"].min()
    t_max = df["frame.time"].max()

    if len(dist.global_diffs) == 0:
        if not silent:
            print("    [!] No inter-frame diffs available; skipping.")
        return df.reset_index(drop=True)

    # --- Sample 2-3 target IPs for this dataset ---
    n_ips = random.randint(2, 3)
    all_ips = dist.syn_tuple_values[:, 0]
    unique_ips, unique_idx = np.unique(all_ips, return_index=True)
    n_ips = min(n_ips, len(unique_ips))
    ip_probs = np.array([dist.syn_tuple_probs[unique_idx[k]] for k in range(len(unique_ips))])
    ip_probs /= ip_probs.sum()
    chosen_idx = np.random.choice(len(unique_ips), size=n_ips, replace=False, p=ip_probs)
    chosen_ips = unique_ips[chosen_idx]

    # --- Distribute injections across chosen IPs ---
    ip_injection_counts = _distribute_injections(injections, n_ips)

    if not silent:
        for ip, cnt in zip(chosen_ips, ip_injection_counts):
            print(f"    [*] IP {ip}: {cnt} injections")

    # --- Inject per IP ---
    new_rows = []
    for ip, count in zip(chosen_ips, ip_injection_counts):
        # Find send value for this IP from the SYN tuple distribution
        ip_mask = dist.syn_tuple_values[:, 0] == ip
        send_val = int(dist.syn_tuple_values[ip_mask][0][1]) if ip_mask.any() else 1

        for _ in range(count):
            frames = _generate_handshake(dist, len_spread=len_spread,
                                         fixed_ip=ip, fixed_send=send_val)
            diffs  = np.random.choice(dist.global_diffs, size=3, replace=True)
            times  = _choose_insertion_times(t_min, t_max, diffs)
            for frame_dict, ts in zip(frames, times):
                frame_dict["frame.time"] = ts
                new_rows.append(frame_dict)

    if not silent:
        print(f"    [*] Injected {injections} handshakes ({injections * 3} frames) "
              f"across {n_ips} IPs: {list(zip(chosen_ips, ip_injection_counts))}")

    df_new   = pd.DataFrame(new_rows)
    combined = pd.concat([df, df_new], ignore_index=True)
    combined = combined.sort_values("frame.time").reset_index(drop=True)
    combined["frame.number"] = np.arange(1, len(combined) + 1)
    return combined


# -------------------------------------------------------------------------------
# Batch processing
# -------------------------------------------------------------------------------

def inject_all_mixed(all_mixed: list, injections: int,
                     len_spread: float = 1.0, silent: bool = False) -> list:
    """Precompute distributions once, then inject into every all_mixed[i][j]."""

    # --- 1. Build global pool ---
    frames = [ds for level in all_mixed for ds in level if ds is not None]
    if not frames:
        raise ValueError("No valid DataFrames found in all_mixed.")
    pool = pd.concat(frames, ignore_index=True)
    print(f"[+] Global pool: {len(pool):,} rows from {len(frames)} datasets.")

    # --- 2. Precompute all distributions once ---
    dist = precompute_distributions(pool)
    del pool  # free memory

    # --- 3. Inject ---
    all_mixed_injected = []
    total = sum(len(level) for level in all_mixed)

    with tqdm(total=total, desc="Injecting into all_mixed[i][j]", unit="mix") as pbar:
        for i, level in enumerate(all_mixed):
            level_injected = []
            for j, ds in enumerate(level):
                pbar.set_postfix(i=i, j=j)
                if ds is None:
                    level_injected.append(None)
                    pbar.update(1)
                    continue
                try:
                    ds_injected = augment_trace(ds, dist, injections=injections,
                                              len_spread=len_spread, silent=silent)
                    level_injected.append(ds_injected)
                except Exception as e:
                    print(f"\n    [!] Failed for all_mixed[{i}][{j}]: {e} -- keeping original")
                    level_injected.append(ds)
                pbar.update(1)
            all_mixed_injected.append(level_injected)

    return all_mixed_injected


# -------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Inject synthetic TCP handshakes (pool-based) into every "
                    "all_mixed[i][j] DataFrame in a mixed_datasets.pkl file."
    )
    parser.add_argument("--input",  type=Path, required=True,
                        help="Path to mixed_datasets.pkl")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write mixed_datasets_injected.pkl")
    parser.add_argument("--injections", type=int, default=1,
                        help="Number of handshake groups (3 frames each) to inject "
                             "per dataset (default: 1)")
    parser.add_argument("--len-spread", type=float, default=1.0,
                        help="Std multiplier for frame length sampling. "
                             "1.0 = empirical distribution (default). "
                             "Higher values generate more extreme frame lengths.")
    parser.add_argument("--silent", action="store_true", default=False,
                        help="Suppress per-dataset injection details (default: verbose)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"[+] Loading {args.input} ...")
    with open(args.input, "rb") as f:
        all_mixed = pickle.load(f)

    n_i = len(all_mixed)
    n_j = max((len(level) for level in all_mixed), default=0)
    print(f"[+] Loaded all_mixed with shape ~({n_i}, {n_j})")
    print(f"[+] Settings: injections={args.injections} per dataset, "
          f"len-spread={args.len_spread}\n")

    all_mixed_injected = inject_all_mixed(all_mixed, injections=args.injections,
                                          len_spread=args.len_spread, silent=args.silent)

    print(f"\n[+] Writing output to {args.output} ...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_mixed_injected, f)
    print("[OK] All done.")


if __name__ == "__main__":
    _cli()