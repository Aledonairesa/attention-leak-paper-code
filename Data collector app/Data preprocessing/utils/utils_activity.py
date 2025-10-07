import pandas as pd
import numpy as np
from functools import reduce

# ==========================================================================
# Helper functions
# ==========================================================================

def count_rows_in_time_intervals(df, time_col, start, end, interval, count_col):
    bins = np.arange(start, end + interval, interval)
    df = df.copy()
    df['time_bin'] = pd.cut(df[time_col], bins=bins, labels=bins[:-1], right=False)
    return (
        df.groupby('time_bin', observed=True)
        .size()
        .reindex(bins[:-1], fill_value=0)
        .reset_index()
        .rename(columns={'time_bin': 'time', 0: count_col})
    )

def count_browser_rows_in_time_intervals(df, time_col, start, end, interval, count_col):
    df = df.copy()
    bins = np.arange(start, end + interval, interval)
    df = df[df['message'].str.contains('Edge|Chrome', na=False, case=False)]
    df['time_bin'] = pd.cut(df[time_col], bins=bins, labels=bins[:-1], right=False)
    return (
        df.groupby('time_bin', observed=True)
        .size()
        .reindex(bins[:-1], fill_value=0)
        .reset_index()
        .rename(columns={'time_bin': 'time', 0: count_col})
    )

def count_match_rows_in_time_intervals(df, time_col, start_time, end_time, interval, new_col_name, categorical_column, col_value):
    df = df[df[categorical_column] == col_value].copy()
    bins = np.arange(start_time, end_time + interval, interval)
    df['time_bin'] = pd.cut(df[time_col], bins=bins, labels=bins[:-1], right=False)
    return (
        df.groupby('time_bin', observed=True)
        .size()
        .reindex(bins[:-1], fill_value=0)
        .reset_index()
        .rename(columns={'time_bin': 'time', 0: new_col_name})
    )

def shannon_entropy(elements):
    elements = pd.Series(elements).dropna()
    if len(elements) == 0:
        return 0

    # Ensure uniform type
    elements = elements.astype(str)
    values, counts = np.unique(elements, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def aggregate_in_time_intervals(dataframe, time_column, numerical_column,
                                start_time, end_time, time_interval, measure):
    df = dataframe.copy()
    bins = np.arange(start_time, end_time + time_interval, time_interval)
    df['time_bin'] = pd.cut(df[time_column], bins=bins, labels=bins[:-1], right=False)

    def normalized_raggedness(x):
        return np.sum(np.abs(np.diff(x))) / (len(x) - 1) if len(x) > 1 else 0

    def iqr_fast(x):
        return np.percentile(x, 75) - np.percentile(x, 25) if len(x) > 0 else 0

    aggregation_funcs = {
        'sum': 'sum',
        'max': 'max',
        'min': 'min',
        'mean': 'mean',
        'range': lambda x: x.max() - x.min() if len(x) > 0 else 0,
        'variance': lambda x: x.var(ddof=0) if len(x) > 1 else 0,
        'std': lambda x: x.std(ddof=0) if len(x) > 1 else 0,
        'iqr': iqr_fast,
        'raggedness': normalized_raggedness,
        'entropy': shannon_entropy
    }

    if measure not in aggregation_funcs:
        raise ValueError(f"Invalid measure: {measure}")

    result = (
        df.groupby('time_bin', observed=True)[numerical_column]
        .agg(aggregation_funcs[measure])
        .reindex(bins[:-1], fill_value=0)
        .reset_index()
        .rename(columns={'time_bin': 'time', numerical_column: f"{numerical_column}_{measure}"})
    )

    return result

def count_unique_categories_in_time_intervals(df, time_col, start_time, end_time, interval,
                                              new_col_name, categorical_column):
    # Copy to avoid mutating original
    df = df.copy()

    # Define bin edges and labels
    bins = np.arange(start_time, end_time + interval, interval)
    labels = bins[:-1]

    # Assign each row to a bin
    df['time_bin'] = pd.cut(
        df[time_col],
        bins=bins,
        labels=labels,
        right=False
    )

    # Group and count unique categories
    result = (
        df.groupby('time_bin', observed=True)[categorical_column]
          .nunique()
          .reindex(labels, fill_value=0)
          .reset_index()
          .rename(columns={'time_bin': 'time', categorical_column: new_col_name})
    )

    return result

def percentage_udp_in_time_intervals(
    dataframe,
    time_column,
    proto_column,
    start_time,
    end_time,
    time_interval
):
    bins = np.arange(start_time, end_time + time_interval, time_interval)
    df = dataframe.copy()
    df['time_bin'] = pd.cut(df[time_column], bins=bins, labels=bins[:-1], right=False)

    df['is_udp'] = (df[proto_column] == 17).astype(int)

    grouped = (
        df.groupby('time_bin', observed=True)
          .agg(total_count=(proto_column, 'count'), udp_count=('is_udp', 'sum'))
          .reindex(bins[:-1], fill_value=0)
          .reset_index()
    )

    grouped['percentage_udp'] = np.where(
        grouped['total_count'] == 0,
        0,
        100 * grouped['udp_count'] / grouped['total_count']
    )

    return grouped[['time_bin', 'percentage_udp']].rename(columns={'time_bin': 'time'})


def count_non_nan_in_time_intervals(dataframe, time_column, start_time, end_time,
                                    time_interval, new_col_name, categorical_column,
):
    """
    For each [start_time, end_time) bin of width time_interval in dataframe[time_column],
    count rows where categorical_column is non-NaN.
    """
    # make a local copy so we don’t clobber the caller’s DataFrame
    df = dataframe.copy()

    # build interval edges and assign each timestamp into a left‐inclusive bin
    bins = np.arange(start_time, end_time + time_interval, time_interval)
    df['time_bin'] = pd.cut(
        df[time_column],
        bins=bins,
        labels=bins[:-1],
        right=False
    )

    # count non-null values of categorical_column per bin,
    # reindex to ensure every interval appears
    result = (
        df.groupby('time_bin', observed=True)[categorical_column]
          .count()
          .reindex(bins[:-1], fill_value=0)
          .reset_index()
    )

    # rename to match build_activity_df’s “time” + our new column name
    result.rename(
        columns={'time_bin': 'time', categorical_column: new_col_name},
        inplace=True
    )

    return result

def count_unique_tuples_in_time_intervals(
    df, time_col, start_time, end_time, interval, new_col_name
):
    """
    Count the number of unique (ip.src, ip.dst, port.src, port.dst) tuples
    in each [start_time, end_time) bin of width `interval` in df[time_col].
    """
    df = df.copy()

    # ensure required columns are present
    required_cols = ['ip.src', 'ip.dst', 'port.src', 'port.dst']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # define bins and corresponding labels (left‐edges)
    bins = np.arange(start_time, end_time + interval, interval)
    labels = bins[:-1]

    # assign each timestamp into a left‐inclusive bin
    df['time_bin'] = pd.cut(df[time_col], bins=bins, labels=labels, right=False)

    # build the 4‐tuple for each row
    df['tuple'] = list(zip(
        df['ip.src'], df['ip.dst'],
        df['port.src'], df['port.dst']
    ))

    # group by bin and count unique tuples, ensuring all bins appear
    result = (
        df.groupby('time_bin', observed=True)['tuple']
          .nunique()
          .reindex(labels, fill_value=0)
          .reset_index()
          .rename(columns={'time_bin': 'time', 'tuple': new_col_name})
    )

    return result


# ==========================================================================
# Activity generation pipeline function
# ==========================================================================
def build_activity_df(timestamps_df, frames_df, start_time, end_time, time_interval):
    """
    Build a per-interval activity table combining task counts, traffic statistics,
    protocol behaviour, DNS activity, and several derived transformations.
    """
    # ------------------------------------------------------------------
    # Task counters (from the separate timestamps table)
    # ------------------------------------------------------------------
    activity_df = count_rows_in_time_intervals(
        timestamps_df, "time",
        start_time, end_time, time_interval,
        count_col="num_tasks",
    )

    browser_df = count_browser_rows_in_time_intervals(
        timestamps_df, "time",
        start_time, end_time, time_interval,
        count_col="num_browser_tasks",
    )
    activity_df = pd.merge(activity_df, browser_df, on="time")
    activity_df["num_non_browser_tasks"] = (
        activity_df["num_tasks"] - activity_df["num_browser_tasks"]
    )

    # ------------------------------------------------------------------
    # Frame counters and size aggregates
    # ------------------------------------------------------------------
    num_frames_df = count_rows_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        count_col="num_frames",
    )

    frame_len_measures = [
        "sum", "max", "min", "mean", "range",
        "variance", "std", "iqr", "raggedness",
    ]
    frame_len_dfs = [
        aggregate_in_time_intervals(
            frames_df, "frame.time", "frame.len",
            start_time, end_time, time_interval, m,
        )
        for m in frame_len_measures
    ]

    send_measures = ["sum", "variance", "raggedness"]
    send_dfs = [
        aggregate_in_time_intervals(
            frames_df, "frame.time", "send",
            start_time, end_time, time_interval, m,
        )
        for m in send_measures
    ]

    # ------------------------------------------------------------------
    # TCP-flag statistics
    # ------------------------------------------------------------------
    tcp_flag_dfs = [
        count_match_rows_in_time_intervals(
            frames_df, "frame.time",
            start_time, end_time, time_interval,
            new_col_name=f"num_{flag.replace('·', '')}_flag",
            categorical_column="tcp.flags.str",
            col_value=flag,
        )
        for flag in frames_df["tcp.flags.str"].dropna().unique()
        if isinstance(flag, str)
    ]

    tcp_entropy_df = aggregate_in_time_intervals(
        frames_df, "frame.time", "tcp.flags.str",
        start_time, end_time, time_interval, "entropy",
    )

    # ------------------------------------------------------------------
    # Unique-count and entropy features for IPs and ports
    # ------------------------------------------------------------------
    num_unique_ips_df = count_unique_categories_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        new_col_name="num_unique_IPs",
        categorical_column="ip",
    )
    ip_entropy_df = aggregate_in_time_intervals(
        frames_df, "frame.time", "ip",
        start_time, end_time, time_interval, "entropy",
    )

    num_unique_dst_ports_df = count_unique_categories_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        new_col_name="num_unique_dst_ports",
        categorical_column="port.dst",
    )
    dst_port_entropy_df = aggregate_in_time_intervals(
        frames_df, "frame.time", "port.dst",
        start_time, end_time, time_interval, "entropy",
    )

    num_unique_src_ports_df = count_unique_categories_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        new_col_name="num_unique_src_ports",
        categorical_column="port.src",
    )
    src_port_entropy_df = aggregate_in_time_intervals(
        frames_df, "frame.time", "port.src",
        start_time, end_time, time_interval, "entropy",
    )

    # ------------------------------------------------------------------
    # Connection-level features
    # ------------------------------------------------------------------
    socket_pairs_df = count_unique_tuples_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        new_col_name="num_unique_socket_pairs",
    )

    percentage_udp_df = percentage_udp_in_time_intervals(
        frames_df, "frame.time", "ip.proto",
        start_time, end_time, time_interval,
    )

    proto_dfs = [
        count_match_rows_in_time_intervals(
            frames_df, "frame.time",
            start_time, end_time, time_interval,
            new_col_name=f"num_proto_{proto}",
            categorical_column="ip.proto",
            col_value=proto,
        )
        for proto in (6, 17)   # 6 = TCP, 17 = UDP
    ]

    # ------------------------------------------------------------------
    # DNS-related features
    # ------------------------------------------------------------------
    num_unique_dns_names_df = count_unique_categories_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        new_col_name="num_unique_dns_names",
        categorical_column="dns.qry.name",
    )
    num_dns_names_df = count_non_nan_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        new_col_name="num_dns_names",
        categorical_column="dns.qry.name",
    )

    num_unique_dns_ips_df = count_unique_categories_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        new_col_name="num_unique_dns_IPs",
        categorical_column="dns.a",
    )
    num_dns_ips_df = count_non_nan_in_time_intervals(
        frames_df, "frame.time",
        start_time, end_time, time_interval,
        new_col_name="num_dns_IPs",
        categorical_column="dns.a",
    )

    dns_query_ips = frames_df.loc[
        frames_df["dns.qry.name"].notna(), "ip"
    ]
    if not dns_query_ips.empty:
        dns_ip = dns_query_ips.mode().iat[0]
        num_dns_frames_df = count_match_rows_in_time_intervals(
            frames_df, "frame.time",
            start_time, end_time, time_interval,
            new_col_name="num_DNS_frames",
            categorical_column="ip",
            col_value=dns_ip,
        )
    else:
        bins = np.arange(start_time, end_time + time_interval, time_interval)
        num_dns_frames_df = pd.DataFrame(
            {"time": bins[:-1], "num_DNS_frames": 0}
        )

    # ------------------------------------------------------------------
    # Consolidate everything in one merge
    # ------------------------------------------------------------------
    all_feature_dfs = (
        [activity_df, num_frames_df] +
        frame_len_dfs +
        send_dfs +
        tcp_flag_dfs + [tcp_entropy_df] +
        [
            num_unique_ips_df, ip_entropy_df,
            num_unique_dst_ports_df, dst_port_entropy_df,
            num_unique_src_ports_df, src_port_entropy_df,
            socket_pairs_df,
            percentage_udp_df,
            *proto_dfs,
            num_unique_dns_names_df, num_dns_names_df,
            num_unique_dns_ips_df, num_dns_ips_df,
            num_dns_frames_df,
        ]
    )

    activity_df = reduce(
        lambda left, right: pd.merge(left, right, on="time"),
        all_feature_dfs,
    )

    # ------------------------------------------------------------------
    # Derived ratios and transformations
    # ------------------------------------------------------------------
    activity_df["receive_frac"] = (
        (1 - activity_df["send_sum"] / activity_df["num_frames"])
        .where(activity_df["num_frames"] != 0, 0)
    )
    activity_df.drop(columns=["send_sum"], inplace=True)

    base_columns = [
        c for c in activity_df.columns
        if c not in (
            "time",
            "num_tasks",
            "num_browser_tasks",
            "num_non_browser_tasks",
        )
    ]

    diff_df = activity_df[base_columns].diff().fillna(0)
    lad_df = np.log1p(diff_df.abs()).add_suffix("_LAD")
    log_df = np.log1p(activity_df[base_columns]).add_suffix("_Log")
    diff_df_named = diff_df.add_suffix("_Diff")
    absdiff_df = diff_df.abs().add_suffix("_AbsDiff")

    activity_df = pd.concat(
        [activity_df, lad_df, log_df, diff_df_named, absdiff_df],
        axis=1,
    )

    return activity_df