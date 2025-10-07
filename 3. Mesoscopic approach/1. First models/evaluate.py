import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------
# Fixed config (before interval override)
# ---------------------------------------------------------------------
FEATURE_DIR       = Path("./../../Data generation - App/select_features_results")
FEATURE_PKL_FILES = [
    "features_all.pkl",
    "features_corr_var.pkl",
    "features_RFE.pkl",
]
PCA_FILE    = FEATURE_DIR / "pca_40_components.pkl"

# Will be overridden in main()
DATA_SPLITS_DIR = Path("")
PKL_DIR         = Path("")

TARGET_COL = "num_tasks"

# Import models after path constants so this can run standalone
from models import MODEL_REGISTRY

# ---------------------------------------------------------------------
# Feature maps and defaults
# ---------------------------------------------------------------------
FEATURE_MAP = {Path(fn).stem: fn for fn in FEATURE_PKL_FILES}

# ★ add a synthetic entry so the PCA experiment can be requested directly
FEATURE_MAP["features_corr_var_PCA40"] = "features_corr_var.pkl"

DEFAULT_MODELS   = list(MODEL_REGISTRY.keys())
DEFAULT_FEATURES = list(FEATURE_MAP.keys())

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_split(split_id: int) -> pd.DataFrame:
    return pd.read_csv(DATA_SPLITS_DIR / f"split_{split_id}.csv")

def valid_splits(interval: float) -> List[int]:
    # UNAVAILABLE_SPLITS logic stays unchanged
    return sorted(set(range(1, 51)) - UNAVAILABLE_SPLITS)

def neighbouring_ids(k: int) -> set:
    return {
        i for i in (k - 1, k + 1)
        if 1 <= i <= 50 and i not in UNAVAILABLE_SPLITS
    }

def clip_extremes(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lower = train.quantile(0.01)
    upper = train.quantile(0.99)
    return (
        train.clip(lower=lower, upper=upper, axis="columns"),
        test.clip(lower=lower,  upper=upper,  axis="columns"),
    )

def evaluate_model(
    model_name: str,
    model_ctor,
    feature_names: List[str],
    feature_label: str,
    interval: float,
    pca: Optional[object] = None,
):
    """
    For each test split, fit on the remaining splits and
    save raw y_true / y_pred vectors into a dict.
    """
    split_results = {}

    for k in tqdm(valid_splits(interval), desc=f"{model_name}-{feature_label}"):
        excl = neighbouring_ids(k)
        train_ids = [i for i in valid_splits(interval) if i not in {k, *excl}]

        # Load data
        test_df  = load_split(k)
        train_df = pd.concat([load_split(i) for i in train_ids],
                             ignore_index=True)

        X_tr = train_df[feature_names].copy()
        y_tr = train_df[TARGET_COL].to_numpy()
        X_te = test_df[feature_names].copy()
        y_te = test_df[TARGET_COL].to_numpy()

        # Clip & scale
        X_tr_c, X_te_c = clip_extremes(X_tr, X_te)
        scaler = StandardScaler().fit(X_tr_c)
        X_tr_s = scaler.transform(X_tr_c)
        X_te_s = scaler.transform(X_te_c)

        # PCA if provided
        if pca is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "X does not have valid feature names.*",
                    category=UserWarning
                )
                X_tr_f = pca.transform(pd.DataFrame(X_tr_s, columns=feature_names))
                X_te_f = pca.transform(pd.DataFrame(X_te_s, columns=feature_names))
        else:
            X_tr_f, X_te_f = X_tr_s, X_te_s

        # Fit & predict
        model = model_ctor()
        model.fit(X_tr_f, y_tr)
        y_pr = model.predict(X_te_f)

        # Save true / pred lists
        split_results[f"test_split_{k}"] = {
            "true": y_te.tolist(),
            "pred": y_pr.tolist()
        }

    # Dump into interval-specific folder
    pkl_path = PKL_DIR / f"{model_name}_{feature_label}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(split_results, f)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save per-split predictions for selected models, features, and interval"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
        help="List of model names to run"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=DEFAULT_FEATURES,
        default=DEFAULT_FEATURES,
        help="List of feature labels to use"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Data-split interval (e.g. 2.0 or 5.0), will look in Splits/{interval}s"
    )
    args = parser.parse_args()

    # Override globals
    interval_str = f"{args.interval}s"
    DATA_SPLITS_DIR = Path(f"./../../../Splits/{interval_str}")

    # Prepare output directory
    PKL_DIR = Path("prediction_pkls") / interval_str
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    # Load holdout splits only once
    with open("./../../Data generation - App/holdout_splits.txt") as f:
        UNAVAILABLE_SPLITS = {
            int(line.strip().split("_")[1].split(".")[0])
            for line in f if line.strip()
        }
    UNAVAILABLE_SPLITS.add(50)

    # -----------------------------------------------------------------
    # Unified evaluation loop
    # -----------------------------------------------------------------
    for feature_label in args.features:
        feat_file = FEATURE_MAP[feature_label]
        with open(FEATURE_DIR / feat_file, "rb") as f:
            feat_names = pickle.load(f)

        # Decide if PCA should be applied based on the feature label itself
        pca_model = None
        if feature_label.endswith("_PCA40"):
            with open(PCA_FILE, "rb") as f:
                pca_model = pickle.load(f)

        for model_name in args.models:
            print(f" Saving preds for {model_name} + {feature_label} @ {interval_str}")
            evaluate_model(
                model_name,
                MODEL_REGISTRY[model_name],
                feat_names,
                feature_label,
                args.interval,
                pca=pca_model
            )

    print("Done.")
