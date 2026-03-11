from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from dataclasses import dataclass, field

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

# dataset_id value assigned to all injected frames.
INJECTED_DATASET_ID = 0


# -------------------------------------------------------------------------------
# ASN lookup table  (built once from the pool)
# -------------------------------------------------------------------------------

def _build_asn_lookup(pool: pd.DataFrame) -> dict[str, dict]:
    """Return {ip_str: {col: value, ...}} from the first real row per IP.

    Covers all columns that the feature extractor may call str.contains() on:
    - any column whose name contains 'asn' (case-insensitive), and
    - any column whose dtype is object/string (e.g. network_name, org_name).

    NaN values in string-typed columns are replaced with "" so that
    str.contains() never receives a non-string argument and raises TypeError.
    """
    str_cols = [
        c for c in pool.columns
        if "asn" in c.lower() or pool[c].dtype == object
    ]
    # Always exclude the core structural columns — we never want to override these
    exclude = {"ip", "tcp.flags.str", "frame.time"}
    str_cols = [c for c in str_cols if c not in exclude]

    if not str_cols:
        return {}

    lookup: dict[str, dict] = {}
    for ip_str, grp in pool.groupby("ip"):
        first = grp[str_cols].dropna(how="all")
        if first.empty:
            # All NaN: use "" for string cols, NaN for everything else
            row_vals = {c: "" for c in str_cols}
        else:
            row_vals = first.iloc[0][str_cols].to_dict()
            # Replace any remaining NaN in string columns with ""
            for c in str_cols:
                v = row_vals.get(c, "")
                if not isinstance(v, str) and (v is None or (isinstance(v, float) and np.isnan(v))):
                    row_vals[c] = ""
        lookup[str(ip_str)] = row_vals
    return lookup


# -------------------------------------------------------------------------------
# Precomputed distributions  (identical logic to inject_frames.py)
# -------------------------------------------------------------------------------

@dataclass
class PoolDistributions:
    """All distributions needed for handshake generation, precomputed as numpy arrays."""
    syn_tuple_values: np.ndarray   # shape (N, 2) — each row is [ip, send]
    syn_tuple_probs:  np.ndarray   # shape (N,)

    len_syn_mean:    float
    len_syn_std:     float
    len_synack_mean: float
    len_synack_std:  float
    len_ack_mean:    float
    len_ack_std:     float
    len_min:         int
    len_max:         int

    srcport_values: np.ndarray | None   # None when column absent (e.g. after merge_ports)
    srcport_probs:  np.ndarray | None
    dstport_values: np.ndarray | None
    dstport_probs:  np.ndarray | None

    global_diffs: np.ndarray

    # ASN lookup: ip_str -> {col: value}  (post-init, not part of constructor)
    asn_lookup: dict = field(default_factory=dict)


def _build_distribution(series: pd.Series):
    """Return (values_array, probs_array) from a Series, excluding NaNs."""
    counts = series.dropna().value_counts(normalize=True)
    return counts.index.to_numpy(), counts.values.astype(float)


def precompute_distributions(pool: pd.DataFrame) -> PoolDistributions:
    """Compute all sampling distributions once from the global pool (mixed_df)."""
    print("[+] Precomputing distributions from mixed_df pool ...")

    # --- Resolver IPs---
    resolvers: set[str] = set()
    for col in ("tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"):
        if col in pool.columns:
            resolvers.update(pool.loc[pool[col] == 53, "ip"].astype(str).unique())
    print(f"    Resolver IPs found: {len(resolvers)}")

    # --- Google IP range to exclude ---
    google_start = int(ipaddress.IPv4Address("142.250.0.0"))
    google_end   = int(ipaddress.IPv4Address("142.251.255.255"))

    def _is_google(ip_series: pd.Series) -> pd.Series:
        try:
            ip_ints = ip_series.apply(lambda ip: int(ipaddress.IPv4Address(str(ip))))
            return ip_ints.between(google_start, google_end)
        except Exception:
            return pd.Series(False, index=ip_series.index)

    # --- SYN tuple distribution (ip, send) ---
    syn = pool[pool["tcp.flags.str"] == FLAG_SYN].copy()
    if syn.empty:
        raise ValueError("Pool has no SYN frames — cannot build distributions.")
    google_mask = _is_google(syn["ip"])
    syn_clean = syn[~syn["ip"].astype(str).isin(resolvers) & ~google_mask]
    if syn_clean.empty:
        syn_clean = syn[~google_mask]
    if syn_clean.empty:
        syn_clean = syn
    print(f"    Google IPs excluded from SYN pool: {google_mask.sum()}")

    tuples = syn_clean[["ip", "send"]].astype(str)
    tuple_counts = tuples.value_counts(normalize=True)
    if len(tuple_counts) > 1:
        tuple_counts = tuple_counts.iloc[1:]
    syn_tuple_values = np.array(tuple_counts.index.tolist())
    syn_tuple_probs  = tuple_counts.values.astype(float)
    syn_tuple_probs /= syn_tuple_probs.sum()

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
            sub = syn[col].dropna() if col in syn.columns else pd.Series(dtype=float)
            sub = sub[sub != 53]
        if sub.empty:
            sub = pool[col].dropna() if col in pool.columns else pd.Series(dtype=float)
            sub = sub[sub != 53]
        if sub.empty:
            return None, None   # signal: column absent or all-53
        return _build_distribution(sub)

    srcport_v, srcport_p = _port_dist("tcp.srcport")
    dstport_v, dstport_p = _port_dist("tcp.dstport")

    # --- Global inter-frame time diffs ---
    pool_sorted  = pool.sort_values("frame.time")
    diffs        = pool_sorted["frame.time"].diff().dropna()
    global_diffs = diffs[diffs > 0].to_numpy()

    # --- ASN lookup table ---
    asn_lookup = _build_asn_lookup(pool)

    print(f"    SYN tuple choices:  {len(syn_tuple_values)}")
    print(f"    frame.len SYN:      mean={len_syn_mean:.1f}, std={len_syn_std:.1f}")
    print(f"    frame.len SYN-ACK:  mean={len_synack_mean:.1f}, std={len_synack_std:.1f}")
    print(f"    frame.len ACK:      mean={len_ack_mean:.1f}, std={len_ack_std:.1f}")
    print(f"    frame.len bounds:   [{len_min}, {len_max}]")
    print(f"    srcport choices:    {len(srcport_v) if srcport_v is not None else 'none (fallback: random ephemeral)'}")
    print(f"    dstport choices:    {len(dstport_v) if dstport_v is not None else 'none (fallback: 443)'}")
    print(f"    inter-frame diffs:  {len(global_diffs):,} samples")
    print(f"    ASN lookup entries: {len(asn_lookup)}")

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
        asn_lookup=asn_lookup,
    )


# -------------------------------------------------------------------------------
# Handshake generation
# -------------------------------------------------------------------------------

def _sample_len(mean: float, std: float, len_min: int, len_max: int,
                len_spread: float) -> int:
    val = np.random.normal(mean, std * len_spread)
    return int(max(round(val), len_min))


def _generate_handshake(dist: PoolDistributions, len_spread: float = 1.0,
                        fixed_ip: str = None, fixed_send: int = None) -> list[dict]:
    """Generate one synthetic TCP handshake (SYN / SYN-ACK / ACK).

    ASN and all other string-typed columns are copied from the lookup table
    for the chosen IP so that downstream feature extraction (e.g. calls to
    str.contains() on network_name) never receives a NaN instead of a string.
    """
    if fixed_ip is not None and fixed_send is not None:
        ip_val, send_val = fixed_ip, int(fixed_send)
    else:
        idx = np.random.choice(len(dist.syn_tuple_values), p=dist.syn_tuple_probs)
        ip_val, send_val = dist.syn_tuple_values[idx]
        send_val = int(send_val)

    # Port sampling: fall back to sensible defaults when the TCP port columns
    # were merged away by merge_ports() before injection was called.
    if dist.srcport_values is not None:
        src_port = int(np.random.choice(dist.srcport_values, p=dist.srcport_probs))
    else:
        src_port = int(np.random.randint(1024, 65535))  # random ephemeral port

    if dist.dstport_values is not None:
        dst_port = int(np.random.choice(dist.dstport_values, p=dist.dstport_probs))
    else:
        dst_port = 443

    len_syn    = _sample_len(dist.len_syn_mean,    dist.len_syn_std,    dist.len_min, dist.len_max, len_spread)
    len_synack = _sample_len(dist.len_synack_mean, dist.len_synack_std, dist.len_min, dist.len_max, len_spread)
    len_ack    = _sample_len(dist.len_ack_mean,    dist.len_ack_std,    dist.len_min, dist.len_max, len_spread)

    # ASN info: copy from the real IP row, fall back to empty dict
    asn_fields = dist.asn_lookup.get(str(ip_val), {})

    base = {
        "ip":            ip_val,
        "ip.proto":      IP_PROTO_TCP,
        "udp.srcport":   np.nan,
        "udp.dstport":   np.nan,
        "dataset_id":    INJECTED_DATASET_ID,
        "ip.factorized": np.nan,
        **asn_fields,   # populate all ASN columns from the real IP's data
    }

    frame1 = {**base, "send": send_val,     "tcp.flags.str": FLAG_SYN,
              "frame.len": len_syn,    "tcp.srcport": src_port, "tcp.dstport": dst_port,
              "S_AS_flag": 1}

    frame2 = {**base, "send": 1 - send_val, "tcp.flags.str": FLAG_SYN_ACK,
              "frame.len": len_synack, "tcp.srcport": dst_port, "tcp.dstport": src_port,
              "S_AS_flag": 1}

    frame3 = {**base, "send": 0,            "tcp.flags.str": FLAG_ACK,
              "frame.len": len_ack,    "tcp.srcport": src_port, "tcp.dstport": dst_port,
              "S_AS_flag": 0}

    return [frame1, frame2, frame3]


def _choose_insertion_times(t_min: float, t_max: float,
                             frame_diffs: np.ndarray) -> list[float]:
    """Return 3 ascending timestamps placed randomly within [t_min, t_max)."""
    span       = t_max - t_min
    total_diff = frame_diffs.sum()
    if total_diff >= span:
        frame_diffs = frame_diffs * (span * 0.8 / total_diff)
        total_diff  = frame_diffs.sum()
    base = random.uniform(t_min, t_max - total_diff)
    return [base, base + frame_diffs[0], base + frame_diffs[0] + frame_diffs[1]]


def _distribute_injections(n_injections: int, n_ips: int) -> list[int]:
    """Randomly partition n_injections across n_ips buckets (all buckets >= 1)."""
    proportions = np.random.dirichlet(np.ones(n_ips))
    counts = np.round(proportions * n_injections).astype(int)
    diff = n_injections - counts.sum()
    for i in range(abs(diff)):
        counts[i % n_ips] += int(np.sign(diff))
    return counts.tolist()


# -------------------------------------------------------------------------------
# Main injection entry point
# -------------------------------------------------------------------------------

def inject_mixed_df(mixed_df: pd.DataFrame,
                    injections: int,
                    len_spread: float = 1.0,
                    silent: bool = False) -> pd.DataFrame:
    """Inject `injections` synthetic TCP handshakes into the post-mix DataFrame.

    Parameters
    ----------
    mixed_df : pd.DataFrame
        The fully mixed DataFrame produced by untangle_tasks.mix_datasets().
        Must contain at minimum: frame.time, frame.len, tcp.flags.str, ip,
        send, tcp.srcport, tcp.dstport, ip.proto, dataset_id.
    injections : int
        Total number of handshake groups (3 frames each) to inject.
    len_spread : float
        Std multiplier for frame length sampling (default 1.0 = empirical).
    silent : bool
        Suppress per-IP injection details.

    Returns
    -------
    pd.DataFrame
        mixed_df with injected frames interleaved, sorted by frame.time, and
        with frame.number reassigned.  Injected rows carry dataset_id = 0.
    """
    print(f"[+] Pool size: {len(mixed_df):,} rows")

    # 1. Build distributions from the full mixed_df
    dist = precompute_distributions(mixed_df)

    t_min = mixed_df["frame.time"].min()
    t_max = mixed_df["frame.time"].max()

    if len(dist.global_diffs) == 0:
        print("    [!] No inter-frame diffs available; returning original DataFrame.")
        return mixed_df.reset_index(drop=True)

    # 2. Sample 2-3 target IPs and distribute injections across them
    n_ips     = random.randint(2, 3)
    all_ips   = dist.syn_tuple_values[:, 0]
    unique_ips, unique_idx = np.unique(all_ips, return_index=True)
    n_ips     = min(n_ips, len(unique_ips))

    ip_probs  = np.array([dist.syn_tuple_probs[unique_idx[k]]
                          for k in range(len(unique_ips))])
    ip_probs /= ip_probs.sum()
    chosen_idx = np.random.choice(len(unique_ips), size=n_ips, replace=False, p=ip_probs)
    chosen_ips = unique_ips[chosen_idx]

    ip_injection_counts = _distribute_injections(injections, n_ips)

    if not silent:
        for ip, cnt in zip(chosen_ips, ip_injection_counts):
            print(f"    [*] IP {ip}: {cnt} injections")

    # 3. Generate synthetic frames
    new_rows: list[dict] = []
    for ip, count in zip(chosen_ips, ip_injection_counts):
        ip_mask  = dist.syn_tuple_values[:, 0] == ip
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

    # 4. Merge, sort, renumber
    df_new = pd.DataFrame(new_rows)
    # Assign unique negative orig_index values to injected rows so that:
    #   1. .astype(int) never hits NaN (avoids ValueError on conversion), and
    #   2. each row has a distinct label (avoids duplicate-label ValueError in reindex).
    # Negative values are never in truth.index (which spans 0..N-1), so
    # reindex silently drops them from all metric computations.
    df_new["orig_index"] = np.arange(-1, -len(df_new) - 1, -1)
    combined = pd.concat([mixed_df, df_new], ignore_index=True)
    combined = combined.sort_values("frame.time").reset_index(drop=True)
    combined["frame.number"] = np.arange(1, len(combined) + 1)

    print(f"[+] Done. {len(mixed_df):,} original + {len(df_new):,} injected "
          f"= {len(combined):,} total frames.")
    return combined


# -------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description=(
            "Inject synthetic TCP handshakes into a post-mix DataFrame "
            "(mixed_dataset.pkl) produced by untangle_tasks.py. "
            "Injected frames carry dataset_id=0 and are invisible to metrics."
        )
    )
    parser.add_argument("--input",  type=Path, required=True,
                        help="Path to mixed_dataset.pkl")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write the injected mixed_dataset.pkl")
    parser.add_argument("--injections", type=int, default=1,
                        help="Total number of handshake groups (3 frames each) "
                             "to inject into mixed_df (default: 1)")
    parser.add_argument("--len-spread", type=float, default=1.0,
                        help="Std multiplier for frame length sampling. "
                             "1.0 = empirical distribution (default). "
                             "Higher values generate more extreme frame lengths.")
    parser.add_argument("--silent", action="store_true", default=False,
                        help="Suppress per-IP injection details (default: verbose)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"[+] Loading {args.input} ...")
    with open(args.input, "rb") as f:
        mixed_df = pickle.load(f)

    print(f"[+] Loaded mixed_df: {len(mixed_df):,} rows, "
          f"{mixed_df['dataset_id'].nunique()} real datasets")
    print(f"[+] Settings: injections={args.injections}, "
          f"len-spread={args.len_spread}, seed={args.seed}\n")

    mixed_injected = inject_mixed_df(
        mixed_df,
        injections=args.injections,
        len_spread=args.len_spread,
        silent=args.silent,
    )

    print(f"\n[+] Writing output to {args.output} ...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(mixed_injected, f)
    print("[OK] All done.")


if __name__ == "__main__":
    _cli()