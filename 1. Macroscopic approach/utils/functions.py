import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import ipaddress
import pickle
import numpy as np
import pandas as pd

from utils.preprocessing import *
from utils.extract_features_utils import *
from utils.preprocessing_features import normalize_columns

def unique_ips(dataset):
    return dataset['ip'].nunique()

def unique_ports(dataset):
    src = dataset['tcp.srcport'].unique()
    dst = dataset['tcp.dstport'].unique()
    return len(set(src).union(dst))

def num_S_AS(dataset):
    flags = dataset['tcp.flags.str']
    return int(((flags == "··········S·") | (flags == "·······A··S·")).sum())

def num_rare_triplets_tcp(dataset):
    # only keep the flags we care about, mapped to short labels
    mapping = {
        '··········S·':  'S',
        '·······A··S·': 'AS',
        '·······A····':  'A',
        '·······AP···': 'AP'
    }
    # the specific rare triplets we want to count
    rare = {
        ('A', 'AS', 'AP'),
        ('A', 'AS', 'AS'),
        ('A', 'AS', 'S'),
        ('AP', 'AS', 'AP'),
        ('AP', 'AS', 'AS'),
        ('AP', 'AS', 'S'),
        ('AS', 'A', 'S'),
        ('AS', 'AP', 'A'),
        ('AS', 'AP', 'AP'),
        ('AS', 'AP', 'AS'),
        ('AS', 'AP', 'S'),
        ('AS', 'AS', 'A'),
        ('AS', 'AS', 'AP'),
        ('AS', 'AS', 'AS'),
        ('AS', 'AS', 'S'),
        ('AS', 'S', 'A'),
        ('AS', 'S', 'AP'),
        ('AS', 'S', 'AS'),
        ('AS', 'S', 'S'),
        ('S', 'AS', 'AP'),
        ('S', 'AS', 'AS'),
        ('S', 'AS', 'S'),
    }

    # map and filter the series in one go
    flags = (
        dataset['tcp.flags.str']
        .map(mapping)    # non-mapped values become NaN
        .dropna()        # drop anything not in mapping
        .tolist()        # go to a plain list
    )

    # sliding‐window count via zip over three offsets
    return sum(
        1
        for a, b, c in zip(flags, flags[1:], flags[2:])
        if (a, b, c) in rare
    )

def num_local_minima_moving_var_send(dataset):
    # compute 250-point moving variance of the “send” column
    mv = dataset['send'].rolling(window=250).var()
    # detect local minima over a 251-point centered window
    local_min = mv == mv.rolling(window=251, center=True).min()
    # count how many local minima
    return int(local_min.sum())

def num_start_matches(dataset):
    """
    Count how many 10-row windows contain, in order,
      1) send=1, ip.proto=17, tcp.flags.str=nan
      2) send=0, ip.proto=17, tcp.flags.str=nan
      3) send=1, ip.proto=6,  tcp.flags.str='··········S·'
      4) send=0, ip.proto=6,  tcp.flags.str='·······A··S·'
   —and also satisfy the “basic” window-level constraints.
    """
    # pull columns into numpy arrays for fast slicing/counting
    proto = dataset['ip.proto'].to_numpy()
    send  = dataset['send'].to_numpy()
    # make sure missing‐flag rows show up as literal 'nan'
    flags = dataset['tcp.flags.str'].fillna('nan').to_numpy()

    window = 10
    matches = 0
    N = len(dataset) - window + 1

    for i in range(N):
        p = proto[i:i+window]
        s = send[i:i+window]
        f = flags[i:i+window]

        # quick window-level filters (in order of rarity)
        if (p == 17).sum() < 2: continue
        if (p == 6) .sum() < 2: continue
        if not ('··········S·' in f):    continue
        if not ('·······A··S·' in f):   continue
        if s.sum()      < 2:            continue
        if (window - s.sum()) < 2:      continue

        # precompute index arrays for each pattern
        idx1 = np.where((s == 1) & (p == 17) & (f == 'nan'))[0]
        idx2 = np.where((s == 0) & (p == 17) & (f == 'nan'))[0]
        idx3 = np.where((s == 1) & (p == 6)  & (f == '··········S·'))[0]
        idx4 = np.where((s == 0) & (p == 6)  & (f == '·······A··S·'))[0]

        # look for an increasing sequence j<k<l<m
        found = False
        for j in idx1:
            k_candidates = idx2[idx2 > j]
            if k_candidates.size == 0: continue
            k = k_candidates[0]

            l_candidates = idx3[idx3 > k]
            if l_candidates.size == 0: continue
            l = l_candidates[0]

            if np.any(idx4 > l):
                matches += 1
                break

    return int(matches)

def num_start_matches_consecutive(df):
    """
    Count occurrences of the 4-step start sequence:
      send:       [1, 0, 1, 0]
      ip.proto:  [17,17, 6, 6]
      tcp.flags: ['nan','nan','··········S·','·······A··S·']
    """
    # pull columns once
    s = df['send'].to_numpy()
    p = df['ip.proto'].to_numpy()
    f = df['tcp.flags.str'].fillna('nan').to_numpy()

    # build all length-4 sliding windows
    Ws = sliding_window_view(s, 4)  # shape = (N-3, 4)
    Wp = sliding_window_view(p, 4)
    Wf = sliding_window_view(f, 4)

    # define the target patterns
    pat_s = np.array([1, 0, 1, 0])
    pat_p = np.array([17, 17, 6, 6])
    pat_f = np.array(['nan', 'nan',
                      '··········S·',
                      '·······A··S·'], dtype=object)

    # compare and count complete matches
    matches = (
        (Ws == pat_s).all(axis=1) &
        (Wp == pat_p).all(axis=1) &
        (Wf == pat_f).all(axis=1)
    )
    return int(matches.sum())

def num_highest_frame_len_by_ip(dataset):
    """
    Count how many IPs (for send==0) have total frame.len above the 90th percentile.
    """
    # sum frame lengths per IP where send == 0
    sums = dataset.loc[dataset['send'] == 0].groupby('ip')['frame.len'].sum()
    # compute the 90th‐percentile threshold
    thresh = sums.quantile(0.9)
    # count IPs exceeding it
    return int((sums > thresh).sum())

def num_highest_frame_len_by_ip_mean(dataset):
    """
    Count how many IPs (for send==0) have total frame.len above the mean.
    """
    # sum frame lengths per IP where send == 0
    sums = dataset.loc[dataset['send'] == 0].groupby('ip')['frame.len'].sum()
    # compute the mean threshold
    mean_thresh = sums.mean()
    # count IPs exceeding it
    return int((sums > mean_thresh).sum())

def num_main_ips(dataset):
    """
    Count "main" IPs by:
      - filtering send==0 & specific TCP flags
      - excluding Google IPs 142.250.0.0-142.251.255.255
      - computing per-IP features: max frame.len & max rolling-mean (window=150; fallback to mean)
      - loading a pre-trained model and summing its binary predictions
    """
    # 1. filter rows
    flags = dataset['tcp.flags.str']
    df = dataset.loc[
        (dataset['send'] == 0) &
        flags.isin(['·······A····', '·······AP···'])
    ]

    # 2. exclude Google IP range by converting to integers
    ip_ints = df['ip'].apply(lambda ip: int(ipaddress.IPv4Address(ip)))
    start, end = (
        int(ipaddress.IPv4Address('142.250.0.0')),
        int(ipaddress.IPv4Address('142.251.255.255'))
    )
    df = df[(ip_ints < start) | (ip_ints > end)]

    # 3. group and compute features
    grp = df.groupby('ip')['frame.len']
    max_vals = grp.max()
    
    window = 150
    # rolling mean per IP → MultiIndex Series; drop the ip index level to regroup
    roll_mean = grp.rolling(window).mean().reset_index(level=0, drop=True)
    max_roll = roll_mean.groupby(df['ip']).max()
    # where rolling didn't produce any value (<window rows), fallback to simple mean
    mean_vals = grp.mean()
    max_mean_vals = max_roll.fillna(mean_vals)

    features = max_vals.to_frame(name='max')
    features['max_mean'] = max_mean_vals

    # 4. load model, predict, sum positives
    with open('main_ip_model.pkl', 'rb') as f:
        model = pickle.load(f)
    preds = model.predict(features)
    return int(preds.sum())

def num_tasks_after_untangle_tasks(
    dataset,
    alpha: float = 0.0001,
    beta: float = 0.9999
):
    """
    Untangle frames into “tasks” using a pre-trained CatBoost model,
    with dual-threshold (alpha/beta) assignment rules.
    Returns the number of tasks (excluding the final 'omitted' cluster).
    """

    # Helper function
    def add_dataset_id_and_merge(dataframes):
        # List to store (start_time, end_time) tuples for each dataframe
        time_ranges = []
        
        # Add 'dataset_id' to each dataframe and calculate time ranges
        for idx, df in enumerate(dataframes, start=1):
            # Extract start and end times for the dataset
            start_time = df['frame.time'].min()
            end_time = df['frame.time'].max()
            time_ranges.append((start_time, end_time))
            
            # Add 'dataset_id' column to the dataframe
            df['dataset_id'] = idx
        
        # Concatenate all dataframes while maintaining the index ordering
        merged_df = pd.concat(dataframes).sort_index()
        
        return merged_df, time_ranges

    # 1. Load classifier once
    with open("./../Approach 1/3. Model optimization/results/hyperopt/model.pkl", "rb") as f:
        model = pickle.load(f)

    # ----- 1. Preprocessing pipeline -----
    df = dataset.copy()
    df['orig_index'] = df.index

    df = remove_frame_number(df)
    df = merge_ports(df)
    df = replace_nans_in_tcp(df)
    df = add_asn_info(df)

    # ----- 2. Exclude non-TCP or Google IPs -----
    ip_int = df['ip'].apply(lambda ip: int(ipaddress.IPv4Address(ip)))
    g_start = int(ipaddress.IPv4Address("142.250.0.0"))
    g_end   = int(ipaddress.IPv4Address("142.251.255.255"))
    valid_mask = (df['ip.proto'] == 6) & ~ip_int.between(g_start, g_end)

    df_valid = df[valid_mask].reset_index(drop=True)
    df_omit  = df[~valid_mask].reset_index(drop=True)

    # ----- 3. Main clustering loop with alpha/beta -----
    tasks = []
    for _, row in df_valid.iterrows():
        new_row = row.to_frame().T

        if not tasks:
            tasks.append(new_row)
            continue

        # compute membership probabilities
        probs = []
        for task_df in tasks:
            trial = pd.concat([task_df, new_row], ignore_index=True)
            feat_dict = extract_features_single_test_sample(trial)
            feat_df = pd.DataFrame([feat_dict])
            feat_df = normalize_columns(
                feat_df,
                ['diff_lenframe_to_last', 'diff_lenframe_to_last_mean']
            )
            probs.append(model.predict_proba(feat_df)[0, 1])

        # assignment rules
        if all(p < alpha for p in probs):
            # start a new task
            tasks.append(new_row)
        elif any(p >= beta for p in probs):
            # assign to the highest‐probability task
            best = int(np.argmax(probs))
            tasks[best] = pd.concat([tasks[best], new_row], ignore_index=True)
        else:
            # uncertain -> omit
            df_omit = pd.concat([df_omit, new_row], ignore_index=True)

    # ----- 4. Append omitted as final cluster -----
    tasks.append(df_omit)

    # ----- 5. Merge tasks & return count (minus omitted) -----
    merged_tasks, start_end_times = add_dataset_id_and_merge(tasks)
    return len(start_end_times) - 1
