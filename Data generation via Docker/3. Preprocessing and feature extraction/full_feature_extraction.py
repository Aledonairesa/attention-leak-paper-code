import os
import sys
import argparse
import json
import pandas as pd
import subprocess
import pickle
import random
import ipaddress
from ipwhois import IPWhois
from tqdm import tqdm

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def generate_datasets(datasets_path, output_folder, cfg):
    """
    Runs generate_dataset.py over all parameter combinations.
    Returns the list of raw .pkl files that were produced.
    """
    combos = []
    for num_webs, num_ds in zip(cfg['num_webs_list'], cfg['num_datasets_per_web_list']):
        for levels in cfg['levels_list']:
            combos.append({
                'num_samples': cfg['num_samples'],
                'num_webs': num_webs,
                'num_datasets_per_web': num_ds,
                'levels': levels,
                'level_weights': cfg['level_weights'],
                'time_between_tasks': cfg['time_between_tasks'],
                'seed': cfg['seed'],
            })

    os.makedirs(output_folder, exist_ok=True)
    print("Generating raw datasets…")
    for combo in tqdm(combos, desc="  generate_dataset", unit="run"):
        cmd = [
            sys.executable, "generate_dataset.py",
            "--datasets_path", datasets_path,
            "--num_samples", str(combo['num_samples']),
            "--num_webs", str(combo['num_webs']),
            "--num_datasets_per_web", str(combo['num_datasets_per_web']),
            "--levels", json.dumps(combo['levels']),
            "--level_weights", json.dumps(combo['level_weights']),
            "--time_between_tasks", str(combo['time_between_tasks']),
            "--seed", str(combo['seed']),
            "--output_folder", output_folder,
        ]
        subprocess.run(cmd, check=True)

    raw_files = [
        f for f in os.listdir(output_folder)
        if f.endswith('.pkl') and not f.startswith(('equalized_', 'asn_', 'nogoogle_'))
    ]
    return raw_files

def equalize_datasets(output_folder, input_files, cfg):
    """
    For each raw .pkl in input_files, produce an equalized_*.pkl.
    Returns the list of equalized files.
    """
    def is_ip_change(frame_dict):
        df = pd.DataFrame(frame_dict)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        return last['ip'] != prev['ip']

    def equalize_file(in_path, out_path, size):
        with open(in_path, 'rb') as f:
            data = pickle.load(f)

        pos_ch, pos_no, neg = [], [], []

        for frame_dict, flag in data:
            if flag:
                if is_ip_change(frame_dict):
                    pos_ch.append((frame_dict, flag))
                else:
                    pos_no.append((frame_dict, flag))
            else:
                neg.append((frame_dict, flag))

        subset = (
            random.sample(pos_ch, size // 4) +
            random.sample(pos_no, size // 4) +
            random.sample(neg, size // 2)
        )

        with open(out_path, 'wb') as f:
            pickle.dump(subset, f)

    print("Equalizing datasets…")
    equalized = []
    for fname in tqdm(input_files, desc="  equalize", unit="file"):
        in_fp = os.path.join(output_folder, fname)
        out_name = f"equalized_{cfg['equalize_num_size']}_{fname}"
        out_fp = os.path.join(output_folder, out_name)
        equalize_file(in_fp, out_fp, cfg['equalize_num_size'])
        equalized.append(out_name)

    return equalized

def enrich_asn(output_folder, input_files):
    """
    For each equalized .pkl in input_files, add ASN columns and write asn_<orig>.pkl.
    Returns the list of asn_ files.
    """
    cache = {}
    def lookup(ip):
        if ip in cache:
            return cache[ip]
        try:
            r = IPWhois(ip).lookup_rdap()
            c = r.get('asn_country_code', 'Unknown')
            d = r.get('asn_description', 'Unknown').split(',')[0].split('-')[0]
            n = r.get('network', {}).get('name', 'Unknown').split('-')[0]
        except:
            c, d, n = 'Unknown', 'Unknown', 'Unknown'
        cache[ip] = (c, d, n)
        return cache[ip]

    print("Adding ASN info…")
    asn_files = []
    for fname in tqdm(input_files, desc="  asn lookup", unit="file"):
        path = os.path.join(output_folder, fname)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        for idx, (df, flag) in enumerate(data):
            for key, ip in df['ip'].items():
                c, d, n = lookup(ip)
                df.setdefault('asn_country', {})[key] = c
                df.setdefault('asn_description', {})[key] = d
                df.setdefault('network_name', {})[key] = n
            data[idx] = (df, flag)

        out_name = f"asn_{fname}"
        out_fp = os.path.join(output_folder, out_name)
        with open(out_fp, 'wb') as f:
            pickle.dump(data, f)
        asn_files.append(out_name)

    return asn_files

def filter_nogoogle(output_folder, input_files, cfg):
    """
    For each asn_ .pkl in input_files, remove Google IPs and write nogoogle_<orig>.pkl.
    Returns the list of nogoogle_ files.
    """
    ranges = [
        (ipaddress.IPv4Address(s), ipaddress.IPv4Address(e))
        for s, e in cfg['to_filter_range_list']
    ]

    def keep_row(df):
        to_del = {
            rid for rid, ip in df['ip'].items()
            if any(lo <= ipaddress.IPv4Address(ip) <= hi for lo, hi in ranges)
        }
        new_df = {
            k: {i: v for i, v in vals.items() if i not in to_del}
            for k, vals in df.items()
        }
        if (len(next(iter(new_df.values()))) > 1 and
            max(df['ip'].keys()) not in to_del):
            return new_df
        return None

    print("Filtering out Google IP ranges…")
    nogoogle = []
    for fname in tqdm(input_files, desc="  filter nogoogle", unit="file"):
        path = os.path.join(output_folder, fname)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        filtered = []
        for df, flag in data:
            kept = keep_row(df)
            if kept is not None:
                filtered.append((kept, flag))

        out_name = f"nogoogle_{fname}"
        out_fp = os.path.join(output_folder, out_name)
        with open(out_fp, 'wb') as f:
            pickle.dump(filtered, f)
        nogoogle.append(out_name)

    return nogoogle

def extract_features_for(files, output_folder, asn_flag):
    """
    Runs extract_features.py on each .pkl in files, then renames
    its <basename>.csv -> feat_<basename>.csv, passing the ASN flag.
    """
    print(f"Extracting features (ASN={'True' if asn_flag else 'False'}) for {len(files)} files…")
    for fname in tqdm(files, desc="  extract_features", unit="file"):
        in_fp = os.path.join(output_folder, fname)
        cmd = [
            sys.executable, "extract_features.py",
            "--dataset_path", in_fp,
            "--output_folder", output_folder,
            "--asn", "True" if asn_flag else "False"
        ]
        subprocess.run(cmd, check=True)

        base = os.path.splitext(fname)[0]
        src_csv = os.path.join(output_folder, f"{base}.csv")
        dst_csv = os.path.join(output_folder, f"feat_{base}.csv")
        if os.path.exists(src_csv):
            os.replace(src_csv, dst_csv)
        else:
            print(f"Warning: expected {src_csv} not found — skipping rename.")

def main():
    p = argparse.ArgumentParser(
        description="Full pipeline with features after each stage"
    )
    p.add_argument("--datasets_path", required=True,
                   help="Folder with raw data for generate_dataset.py")
    p.add_argument("--output_folder", required=True,
                   help="Directory for all .pkl and .csv outputs")
    p.add_argument("--config_file", default="config.json",
                   help="JSON file with pipeline parameters")
    args = p.parse_args()

    cfg = load_config(args.config_file)

    # Stage 1: raw
    raw = generate_datasets(args.datasets_path, args.output_folder, cfg)
    extract_features_for(raw, args.output_folder, asn_flag=False)

    # Stage 2: equalized
    eq = equalize_datasets(args.output_folder, raw, cfg)
    extract_features_for(eq, args.output_folder, asn_flag=False)

    # Stage 3: asn enrichment
    asn = enrich_asn(args.output_folder, eq)
    extract_features_for(asn, args.output_folder, asn_flag=True)

    # Stage 4: filter nogoogle
    ng = filter_nogoogle(args.output_folder, asn, cfg)
    extract_features_for(ng, args.output_folder, asn_flag=True)

if __name__ == "__main__":
    main()
