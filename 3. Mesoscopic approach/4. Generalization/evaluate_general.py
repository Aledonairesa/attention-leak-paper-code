#!/usr/bin/env python
"""
Single-train / single-test experiment script
--------------------------------------------
• Train once on *arbitrary* collections of user–split CSVs  
• Test once, still storing **per-split** y_true / y_pred vectors in a pkl  
• Same data layout, models, and feature pickles as the original script

Example
-------
python evaluate_general.py \
    --train "Alice:1-3,5-10" "Bob:3-10" \
    --test  "Charlie:2-8" \
    --model lgbm \
    --features features_corr_var \
    --interval 5.0 \
    --target num_tasks
"""
import argparse
import datetime as dt
import pickle
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------
DATA_ROOT = Path("./../../Data generation - App/Users")
FEATURE_DIR = Path("./../../Data generation - App/select_features_results")
FEATURE_PKL_FILES = [
    "features_all.pkl",
    "features_corr_var.pkl",
    "features_RFE.pkl",
]
PCA_FILE = FEATURE_DIR / "pca_40_components.pkl"
TARGET_COL_DEFAULT = "num_tasks"

# Import models **after** constants so script is standalone-runnable
from models import MODEL_REGISTRY  # noqa: E402

# Build feature-label → filename map
FEATURE_MAP: Dict[str, str] = {Path(fn).stem: fn for fn in FEATURE_PKL_FILES}
DEFAULT_MODELS = list(MODEL_REGISTRY.keys())
DEFAULT_FEATURES = list(FEATURE_MAP.keys())

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_spec(specs: Sequence[str], interval_str: str) -> List[Tuple[str, List[int]]]:
    """
    Turn ["Alice:1-3,5-10", "Bob:all"] into
    [("Alice", [1,2,3,5,6,7,8,9,10]), ("Bob", [... all detected splits ...])]
    """
    users_splits: List[Tuple[str, List[int]]] = []

    for spec in specs:
        try:
            user, rng = spec.split(":", 1)
        except ValueError:
            raise ValueError(f'Bad spec "{spec}" – must be "User:split_desc"')
        splits_dir = DATA_ROOT / user / "Splits" / interval_str
        if rng.strip().lower() == "all":
            if not splits_dir.exists():
                raise FileNotFoundError(f"Directory not found: {splits_dir}")
            split_ids = []
            for fp in splits_dir.glob('split_*.csv'):
                sid = int(fp.stem.split('_')[1])
                split_ids.append(sid)
            split_ids = sorted(split_ids)
        else:
            split_ids: List[int] = []
            for part in rng.split(','):
                part = part.strip()
                if '-' in part:
                    a, b = map(int, part.split('-'))
                    split_ids.extend(range(a, b + 1))
                else:
                    split_ids.append(int(part))
            split_ids = sorted(set(split_ids))
        users_splits.append((user, split_ids))
    return users_splits


def load_split(user: str, split_id: int, interval_str: str) -> pd.DataFrame:
    path = DATA_ROOT / user / "Splits" / interval_str / f"split_{split_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_csv(path)


def clip_extremes(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Winsorise at 1 % / 99 % to tame outliers."""
    lower = train.quantile(0.01)
    upper = train.quantile(0.99)
    return (
        train.clip(lower=lower, upper=upper, axis="columns"),
        test.clip(lower=lower, upper=upper, axis="columns"),
    )

# ---------------------------------------------------------------------
# Training / inference routine
# ---------------------------------------------------------------------
def train_and_test(
    model_name: str,
    feature_names: List[str],
    target_col: str,
    train_specs: List[Tuple[str, List[int]]],
    test_specs: List[Tuple[str, List[int]]],
    interval: float,
    pca=None,
) -> Dict[str, Dict[str, List]]:
    """
    • Merge *all* train splits, train once
    • Predict each test split separately, returning a per-split dict
    """
    interval_str = f"{interval}s"

    # --- assemble training data ---------------------------------------
    train_frames = [
        load_split(u, sid, interval_str)
        for u, ids in train_specs for sid in ids
    ]
    train_df = pd.concat(train_frames, ignore_index=True)
    X_train = train_df[feature_names].copy()
    y_train = train_df[target_col].to_numpy()

    # --- pre-process: clip & scale -----------------------------------
    X_train_clip, _ = clip_extremes(X_train, X_train)
    scaler = StandardScaler().fit(X_train_clip)
    X_train_scaled = scaler.transform(X_train_clip)

    # --- optional PCA -------------------------------------------------
    if pca is not None:
        X_train_final = pca.transform(pd.DataFrame(X_train_scaled, columns=feature_names))
    else:
        X_train_final = X_train_scaled

    # --- fit model ----------------------------------------------------
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train_final, y_train)

    # --- evaluate test splits ----------------------------------------
    results: Dict[str, Dict[str, List]] = {}
    for user, ids in tqdm(test_specs, desc=f"Testing {model_name}"):
        for sid in ids:
            df_test = load_split(user, sid, interval_str)
            X_test = df_test[feature_names].copy()
            y_test = df_test[target_col].to_numpy()

            # clip & scale using train extremes & scaler
            _, X_test_clip = clip_extremes(X_train, X_test)
            X_test_scaled = scaler.transform(X_test_clip)
            if pca is not None:
                X_test_final = pca.transform(pd.DataFrame(X_test_scaled, columns=feature_names))
            else:
                X_test_final = X_test_scaled

            y_pred = model.predict(X_test_final)
            results[f"{user}_split_{sid}"] = {
                "true": y_test.tolist(),
                "pred": y_pred.tolist(),
            }

    return results

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single train/test experiment and save predictions."
    )
    parser.add_argument(
        "--train", nargs="+", required=True,
        help='User/split spec, e.g. "Alice:1-3,5" "Bob:all"'
    )
    parser.add_argument(
        "--test", nargs="+", required=True,
        help='User/split spec, e.g. "Charlie:2-8"'
    )
    parser.add_argument(
        "--model", choices=DEFAULT_MODELS, required=True,
        help="Model name"
    )
    parser.add_argument(
        "--features", choices=DEFAULT_FEATURES, required=True,
        help="Feature set label"
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Split interval (e.g. 2.0 or 5.0)"
    )
    parser.add_argument(
        "--target", default=TARGET_COL_DEFAULT,
        help=f'Target column (default "{TARGET_COL_DEFAULT}")'
    )
    args = parser.parse_args()

    # prepare output directory
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_DIR = Path("experiments") / f"exp_{ts}"
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # load feature names
    feat_file = FEATURE_MAP[args.features]
    with open(FEATURE_DIR / feat_file, "rb") as f:
        feature_names = pickle.load(f)

    # optional PCA
    pca_model = None
    if args.features == "features_corr_var" and PCA_FILE.exists():
        with open(PCA_FILE, "rb") as f:
            pca_model = pickle.load(f)

    # parse specs, now using actual folder contents for 'all'
    interval_str = f"{args.interval}s"
    train_specs = parse_spec(args.train, interval_str)
    test_specs = parse_spec(args.test, interval_str)

    # run experiment
    preds = train_and_test(
        args.model,
        feature_names,
        args.target,
        train_specs,
        test_specs,
        interval=args.interval,
        pca=pca_model,
    )

    # save predictions
    pkl_path = EXP_DIR / f"{args.model}_{args.features}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(preds, f)

    # record args
    (EXP_DIR / "args_used.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in vars(args).items())
    )

    # final message
    print(f" Results saved to {pkl_path}")


if __name__ == "__main__":
    main()
