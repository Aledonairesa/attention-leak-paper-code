import os
import random
import pickle
import argparse
import pandas as pd
import numpy as np
import ipaddress
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import List, Tuple, Dict

from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from utils.preprocessing import *
from utils.extract_features_utils import *

# --------------------------------------------------------------------------- #
# 1. DATA LOADING AND PRE-PROCESSING
# --------------------------------------------------------------------------- #

def build_clean_datasets(
    filtered_raw_data_dir: Path,
    *,
    sample_size: int = 100,
    seed: int = 1,
    internal_host_prefix: str = "172.17."
) -> List[pd.DataFrame]:
    """Build cleaned datasets following the refined notebook pipeline.

    1. Enumerate subdirectories inside filtered_raw_data_dir.
    2. Collect each <subdir>/data.csv.
    3. Randomly sample sample_size files (seed for reproducibility).
    4. Apply process_frame_time_column, remove_nan_ip_proto,
       filter_hosts, merge_ips_and_create_send_column.
    """
    random.seed(seed)
    subdirs = [filtered_raw_data_dir / d for d in os.listdir(filtered_raw_data_dir)]
    csv_paths = [d / "data.csv" for d in subdirs if (d / "data.csv").is_file()]
    if not csv_paths:
        raise FileNotFoundError(f"No <folder>/data.csv files found in '{filtered_raw_data_dir}'")
    csv_sample = random.sample(csv_paths, min(sample_size, len(csv_paths)))

    clean_dsets: List[pd.DataFrame] = []
    for csv_file in tqdm(csv_sample, desc="Reading & cleaning captures", unit="capture"):
        df = pd.read_csv(csv_file)
        df = process_frame_time_column(df)
        df = remove_nan_ip_proto(df)
        df = filter_hosts(df)
        df = merge_ips_and_create_send_column(df, internal_host_prefix)
        clean_dsets.append(df.reset_index(drop=True))
    return clean_dsets


def in_ip_ranges(ip_str: str, ranges: List[Tuple[str, str]]) -> bool:
    ip_obj = ipaddress.IPv4Address(ip_str)
    return any(
        ipaddress.IPv4Address(start) <= ip_obj <= ipaddress.IPv4Address(end)
        for start, end in ranges
    )


def preprocess_datasets(
    dsets: List[pd.DataFrame],
    ip_ranges_to_filter: List[Tuple[str, str]],
    tcp_flag_values: Tuple[str, ...] = ("·······A····", "·······AP···"),
) -> pd.DataFrame:
    """Concatenate and filter datasets:
       - client→server (send == 0)
       - tcp.flags.str in tcp_flag_values
       - ip not in any ip_ranges_to_filter
    """
    filtered: List[pd.DataFrame] = []
    for d in dsets:
        mask_flags = d["tcp.flags.str"].isin(tcp_flag_values)
        mask_direction = d["send"] == 0
        d2 = d[mask_direction & mask_flags].copy()
        d2 = d2[~d2["ip"].apply(lambda ip: in_ip_ranges(ip, ip_ranges_to_filter))]
        filtered.append(d2.reset_index(drop=True))
    return pd.concat(filtered, ignore_index=True)

# --------------------------------------------------------------------------- #
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #

def build_feature_tables(
    df: pd.DataFrame,
    main_ips: List[str],
    window_sizes: List[int]
) -> Dict[int, pd.DataFrame]:
    """Generate per-IP feature tables for each window size."""
    tables: Dict[int, pd.DataFrame] = {}
    for w in window_sizes:
        rows: List[Dict[str, float]] = []
        for ip, grp in df.groupby("ip")["frame.len"]:
            maximum = grp.max()
            max_mean = grp.rolling(window=w).mean().max() if len(grp) >= w else grp.mean()
            rows.append({
                "ip": ip,
                "max": maximum,
                "max_mean": max_mean,
                "target": 1 if ip in main_ips else 0,
            })
        tables[w] = pd.DataFrame(rows)
    return tables

# --------------------------------------------------------------------------- #
# 3. MODEL SELECTION
# --------------------------------------------------------------------------- #

def tune_models(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[Pipeline, Dict[str, Dict[str, float]]]:
    """Run grid searches for LogisticRegression and RBF-SVM."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search_space = {
        "logreg": {
            "estimator": LogisticRegression(solver="liblinear", class_weight="balanced"),
            "params": {"estimator__C": [0.01, 0.1, 1, 10, 100],
                        "estimator__penalty": ["l1", "l2"]},
        },
        "svm_rbf": {
            "estimator": SVC(kernel="rbf", class_weight="balanced", probability=True),
            "params": {"estimator__C": [0.01, 0.1, 1, 10, 100],
                        "estimator__gamma": ["scale", "auto", 0.01, 0.1, 1, 10]},
        },
    }
    best_model, best_f1 = None, -np.inf
    metrics: Dict[str, Dict[str, float]] = {}
    for name, cfg in search_space.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("estimator", cfg["estimator"])])
        gs = GridSearchCV(pipe, cfg["params"], scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        metrics[name] = {"f1": f1, "accuracy": acc, "best_params": gs.best_params_}
        if f1 > best_f1:
            best_f1, best_model = f1, gs.best_estimator_
    return best_model, metrics

# --------------------------------------------------------------------------- #
# 4. VISUALISATION HELPERS
# --------------------------------------------------------------------------- #

IMAGES_DIR = Path("results") / "main_IPs_images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def plot_frame_len_info(
    df: pd.DataFrame,
    ip_sample: str,
    *,
    filename: str = None
) -> None:
    """Plot and save frame length evolution for a single IP."""
    fig, ax = plt.subplots(figsize=(10, 4))
    series = df[df["ip"] == ip_sample]["frame.len"]
    ax.plot(series.index, series.values, linewidth=1)
    ax.set_title(f"Frame length evolution - IP {ip_sample}")
    ax.set_xlabel("Packet #")
    ax.set_ylabel("frame.len")
    ax.grid(True)

    if filename is None:
        filename = f"frame_len_{ip_sample}.png"
    fig.savefig(IMAGES_DIR / filename)
    plt.close(fig)


def scatter_features(df: pd.DataFrame, *, filename: str = None) -> None:
    """Plot and save scatter of feature space coloured by target, with legend."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot with color mapping
    colors = df["target"].map({0: "tab:blue", 1: "tab:orange"})
    ax.scatter(df["max"], df["max_mean"], c=colors, alpha=0.6, s=30)

    # Add legend manually
    legend_elements = [
        Patch(facecolor="tab:orange", edgecolor="k", label="Main IP"),
        Patch(facecolor="tab:blue", edgecolor="k", label="Non-main IP")
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel("max(frame.len)")
    ax.set_ylabel("max rolling-mean(frame.len)")
    ax.set_title("Feature space - main vs non-main IPs")
    ax.grid(True)

    if filename is None:
        filename = "scatter_features.png"
    fig.savefig(IMAGES_DIR / filename)
    plt.close(fig)


def decision_boundary(model: Pipeline, df: pd.DataFrame, *, filename: str = None) -> None:
    """Plot and save model decision boundary over 2D feature space with legend."""
    fig, ax = plt.subplots(figsize=(6, 5))
    X = df[["max", "max_mean"]].values

    # Background scatter
    colors = df["target"].map({0: "tab:blue", 1: "tab:orange"})
    ax.scatter(df["max"], df["max_mean"], c=colors, alpha=0.6, s=30)

    # Decision boundary
    x_min, x_max = X[:, 0].min() * 0.9, X[:, 0].max() * 1.05
    y_min, y_max = X[:, 1].min() * 0.9, X[:, 1].max() * 1.05
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["max", "max_mean"])
    Z = model.predict(grid).reshape(xx.shape)

    background_cmap = ListedColormap([
        plt.get_cmap('tab10')(0),  # tab:blue for class 0
        plt.get_cmap('tab10')(1)   # tab:orange for class 1
    ])
    ax.contourf(xx, yy, Z, levels=[-0.1, 0.5, 1.1], cmap=background_cmap, alpha=0.15)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Model decision boundary")
    ax.set_xlabel("max(frame.len)")
    ax.set_ylabel("max rolling-mean(frame.len)")
    ax.grid(True)

    # Add legend
    legend_elements = [
        Patch(facecolor="tab:orange", edgecolor="k", label="Main IP"),
        Patch(facecolor="tab:blue", edgecolor="k", label="Non-main IP")
    ]
    ax.legend(handles=legend_elements)

    if filename is None:
        filename = "decision_boundary.png"
    fig.savefig(IMAGES_DIR / filename)
    plt.close(fig)


def barplot_metrics(
    metrics: Dict[str, Dict[str, float]],
    *,
    filename: str = None
) -> None:
    """Plot and save bar chart comparing F1 and accuracy for each model config."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(metrics.keys())
    f1_scores = [metrics[n]["f1"] for n in names]
    accuracy_scores = [metrics[n]["accuracy"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, f1_scores, width, label="F1")
    ax.bar(x + width/2, accuracy_scores, width, label="Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend()
    ax.grid(axis="y", linestyle=":")
    fig.tight_layout()

    if filename is None:
        filename = "model_comparison.png"
    fig.savefig(IMAGES_DIR / filename)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# 5. MAIN
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train IP main/non-main classifier"
    )
    parser.add_argument(
        "main_ips_pkl",
        type=Path,
        help="Path to main_IPs.pkl"
    )
    parser.add_argument(
        "raw_data_dir",
        type=Path,
        help="Path to filtered raw data folder"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of captures to sample"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for sampling"
    )
    args = parser.parse_args()

    # Load main IP list
    with open(args.main_ips_pkl, "rb") as f:
        main_ip_list = pickle.load(f)

    # Build cleaned captures
    clean_dsets = build_clean_datasets(
        args.raw_data_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        internal_host_prefix="172.17."
    )

    if not clean_dsets:
        raise SystemExit("No cleaned datasets; aborting.")

    # Preprocess (client->server, flags, IP range filter)
    google_range = [("142.250.0.0", "142.251.255.255")]
    clean_df = preprocess_datasets(clean_dsets, google_range)

    # Feature extraction
    WINDOW_SIZES = [25, 50, 100, 200, 300]
    features_by_w = build_feature_tables(clean_df, main_ip_list, WINDOW_SIZES)

    # Model training & selection
    best_global_model, metrics_all = None, {}
    best_w, best_f1 = None, -np.inf

    for w, ftab in features_by_w.items():
        model, met = tune_models(ftab[["max", "max_mean"]], ftab["target"])
        for name, vals in met.items():
            key = f"w{w}:{name}"
            metrics_all[key] = vals
            if vals["f1"] > best_f1:
                best_f1, best_global_model, best_w = vals["f1"], model, w

    print(f"Best model: {best_global_model.named_steps['estimator']}")
    print(f"Best window size: {best_w}")

    print("\nBest model full configuration:")
    print(best_global_model.named_steps['estimator'].get_params())

    with open("main_ip_model.pkl", "wb") as f:
        pickle.dump(best_global_model, f)

    # Visualizations
    plot_frame_len_info(clean_df, ip_sample=clean_df["ip"].iloc[0])
    scatter_features(features_by_w[best_w])
    decision_boundary(best_global_model, features_by_w[best_w])
    barplot_metrics(metrics_all)
    plt.show()