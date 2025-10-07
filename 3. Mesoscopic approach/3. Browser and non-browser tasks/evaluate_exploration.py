# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
import random
SEED = 42
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

# ---------------------------------------------------------------------
# Standard library & deps
# ---------------------------------------------------------------------
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

# ---------------------------------------------------------------------
# Fixed config  (edit here if you like)
# ---------------------------------------------------------------------
# Paths & files
FEATURE_DIR   = Path("./../../Data generation - App/select_features_results")
HOLDOUT_FILE  = Path("./../../Data generation - App/holdout_splits.txt")

# Model / features / interval
MODEL_NAME     = "TCN_hidden24"
FEATURE_LABEL  = "features_all"
INTERVAL       = 5.0                        # seconds
INTERVAL_STR   = f"{INTERVAL}s"

# Targets
TARGET_COL            = "num_tasks"
BROWSER_TARGET_COL    = "num_browser_tasks"
NON_BROWSER_TARGET_COL = "num_non_browser_tasks"

# Numbers of splits to draw
N_TRAIN_SPLITS = 5
N_TEST_SPLITS  = 1

# Output
PKL_DIR  = Path("holdout_prediction_pkls") / INTERVAL_STR
PKL_DIR.mkdir(parents=True, exist_ok=True)
PKL_PATH = PKL_DIR / f"{MODEL_NAME}_{FEATURE_LABEL}_holdout.pkl"

# Import models after path constants so this can run standalone
from models import MODEL_REGISTRY            # pylint: disable=wrong-import-position

# ---------------------------------------------------------------------
# Build a map from feature-set label → filename
# ---------------------------------------------------------------------
FEATURE_PKL_FILES = ["features_all.pkl",
                     "features_corr_var.pkl",
                     "features_RFE.pkl"]
FEATURE_MAP: Dict[str, str] = {Path(fn).stem: fn for fn in FEATURE_PKL_FILES}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
DATA_SPLITS_DIR = (
    Path(f"./../../../Splits/{INTERVAL_STR}")
)

def load_split(split_id: int) -> pd.DataFrame:
    """Read one split CSV into a DataFrame."""
    return pd.read_csv(DATA_SPLITS_DIR / f"split_{split_id}.csv")

def clip_extremes(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Winsorise each numeric column to the 1st–99th percentile of the train set."""
    lower = train.quantile(0.01)
    upper = train.quantile(0.99)
    return (
        train.clip(lower=lower, upper=upper, axis="columns"),
        test.clip(lower=lower,  upper=upper,  axis="columns"),
    )

def choose_train_test_splits() -> Tuple[List[int], List[int]]:
    """Randomly sample 4 train + 2 test splits out of the hold-out list."""
    with open(HOLDOUT_FILE) as f:
        holdout_ids = [
            int(line.strip().split("_")[1].split(".")[0])
            for line in f if line.strip()
        ]
    assert len(holdout_ids) >= N_TRAIN_SPLITS + N_TEST_SPLITS, \
        "Not enough hold-out splits for the requested train/test sizes."
    test_ids  = random.sample(holdout_ids, N_TEST_SPLITS)
    remaining = [sid for sid in holdout_ids if sid not in test_ids]
    train_ids = random.sample(remaining, N_TRAIN_SPLITS)
    return train_ids, test_ids

# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main() -> None:
    # -----------------------------------------------------------------
    # Feature list
    # -----------------------------------------------------------------
    feat_file = FEATURE_MAP[FEATURE_LABEL]
    with open(FEATURE_DIR / feat_file, "rb") as f:
        feature_names: List[str] = pickle.load(f)

    # -----------------------------------------------------------------
    # Pick train / test splits
    # -----------------------------------------------------------------
    train_ids, test_ids = choose_train_test_splits()
    print(f"Training on splits  : {train_ids}")
    print(f"Testing  on splits  : {test_ids}")

    # -----------------------------------------------------------------
    # Assemble data
    # -----------------------------------------------------------------
    train_df = pd.concat([load_split(i) for i in train_ids],
                         ignore_index=True)
    test_df  = pd.concat([load_split(i) for i in test_ids],
                         ignore_index=True)

    X_tr = train_df[feature_names].copy()
    y_tr = train_df[TARGET_COL].to_numpy()
    X_te = test_df[feature_names].copy()

    # Keep the three target columns from the test set for later output
    y_te_total        = test_df[TARGET_COL].to_numpy()
    y_te_browser      = test_df[BROWSER_TARGET_COL].to_numpy()
    y_te_non_browser  = test_df[NON_BROWSER_TARGET_COL].to_numpy()

    # -----------------------------------------------------------------
    # Pre-processing: clip → scale
    # -----------------------------------------------------------------
    X_tr_c, X_te_c = clip_extremes(X_tr, X_te)
    scaler = StandardScaler().fit(X_tr_c)
    X_tr_s = scaler.transform(X_tr_c)
    X_te_s = scaler.transform(X_te_c)

    # -----------------------------------------------------------------
    # Fit the model
    # -----------------------------------------------------------------
    model_ctor = MODEL_REGISTRY[MODEL_NAME]
    model = model_ctor()
    print(f"Fitting {MODEL_NAME} ...")
    model.fit(X_tr_s, y_tr)

    print("Running predictions on test sequences ...")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "X does not have valid feature names.*",
            category=UserWarning
        )
        y_pred = model.predict(X_te_s)

    # -----------------------------------------------------------------
    # Gather results & dump
    # -----------------------------------------------------------------
    results = {
        "meta": {
            "train_splits": train_ids,
            "test_splits":  test_ids,
            "model":        MODEL_NAME,
            "features":     FEATURE_LABEL,
            "interval":     INTERVAL,
        },
        "data": {
            # Each is a plain Python list so the pkl stays lightweight
            "pred":                    y_pred.tolist(),
            "num_tasks_true":          y_te_total.tolist(),
            "num_browser_tasks_true":  y_te_browser.tolist(),
            "num_non_browser_tasks_true": y_te_non_browser.tolist(),
        }
    }

    with open(PKL_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f" Saved prediction bundle to: {PKL_PATH.resolve()}")
    print("Done.")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
