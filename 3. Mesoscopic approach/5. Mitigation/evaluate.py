import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import sys

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
TRAIN_SPLITS_DIR: Path = Path("")
PKL_DIR:          Path = Path("")

TARGET_COL = "num_tasks"

# Import models after path constants so this can run standalone
from models import MODEL_REGISTRY

# Build a map from feature label → filename
FEATURE_MAP = { Path(fn).stem: fn for fn in FEATURE_PKL_FILES }

# Default lists
DEFAULT_MODELS   = list(MODEL_REGISTRY.keys())
DEFAULT_FEATURES = list(FEATURE_MAP.keys())

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_split(split_id: int, base_dir: Path) -> pd.DataFrame:
    return pd.read_csv(base_dir / f"split_{split_id}.csv")

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

def evaluate_model_multi(
    model_name: str,
    model_ctor,
    feature_names: List[str],
    feature_label: str,
    interval: float,
    train_dir: Path,
    test_targets: List[Tuple[str, Path]],  # list of (suffix, path)
    pca: Optional[object] = None,
):
    """
    Para cada split de test k:
      - Entrena con los splits normales (excluyendo k y vecinos)
      - Predice en cada test_target (varias carpetas de test)
    Genera un .pkl por test_target (suffix).
    """
    # Prepara un diccionario de resultados por variante
    results_by_variant: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        suffix: {} for suffix, _ in test_targets
    }

    for k in tqdm(valid_splits(interval), desc=f"{model_name}-{feature_label}"):
        excl = neighbouring_ids(k)
        train_ids = [i for i in valid_splits(interval) if i not in {k, *excl}]

        # Carga y concatena entrenamiento (normal)
        train_df = pd.concat(
            [load_split(i, train_dir) for i in train_ids],
            ignore_index=True
        )

        X_tr = train_df[feature_names].copy()
        y_tr = train_df[TARGET_COL].to_numpy()

        # Ajuste de clipping y scaler solo con train
        # (el test de cada variante se transformará con estos)
        # Para cada variante, cargaremos su test_df y aplicaremos las mismas trafo.
        scaler = None
        X_tr_c = None
        X_tr_s = None

        # Entrenamos una única vez por k
        # (aplicamos PCA si procede sobre train y se reutiliza la trafo)
        # NOTA: para el clipping necesitamos un test para calcular X_te_c
        #       pero los umbrales se derivan de train, así que podemos preparar
        #       X_tr_c/X_tr_s aquí y luego procesar cada test por separado.
        X_tr_c = X_tr.clip(
            lower=X_tr.quantile(0.01),
            upper=X_tr.quantile(0.99),
            axis="columns"
        )
        scaler = StandardScaler().fit(X_tr_c)
        X_tr_s = scaler.transform(X_tr_c)

        if pca is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "X does not have valid feature names.*",
                    category=UserWarning
                )
                X_tr_f = pca.transform(pd.DataFrame(X_tr_s, columns=feature_names))
        else:
            X_tr_f = X_tr_s

        model = model_ctor()
        model.fit(X_tr_f, y_tr)

        # Ahora iteramos por cada variante de test
        for suffix, test_dir in test_targets:
            test_df = load_split(k, test_dir)

            X_te = test_df[feature_names].copy()
            y_te = test_df[TARGET_COL].to_numpy()

            # Aplica mismos límites de clipping (1-99% basados en train)
            lower = X_tr.quantile(0.01)
            upper = X_tr.quantile(0.99)
            X_te_c = X_te.clip(lower=lower, upper=upper, axis="columns")

            # Misma normalización
            X_te_s = scaler.transform(X_te_c)

            # Mismo PCA si procede
            if pca is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "X does not have valid feature names.*",
                        category=UserWarning
                    )
                    X_te_f = pca.transform(pd.DataFrame(X_te_s, columns=feature_names))
            else:
                X_te_f = X_te_s

            y_pr = model.predict(X_te_f)

            results_by_variant[suffix][f"test_split_{k}"] = {
                "true": y_te.tolist(),
                "pred": y_pr.tolist()
            }

    # Guardado: un pkl por variante
    for suffix, variant_results in results_by_variant.items():
        pkl_path = PKL_DIR / f"{model_name}_{feature_label}{suffix}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(variant_results, f)

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
    parser.add_argument(
        "--injected",
        nargs="+",   # <-- múltiples variantes: 100 200 Mix ...
        type=str,
        default=None,
        help=("If set, test on one or more injected variants instead of normal. "
              "Examples: 1 100 200 Mix → uses Users/User_Injected_<value>/Splits/{interval}s")
    )
    args = parser.parse_args()

    interval_str = f"{args.interval}s"

    # Train (siempre normal)
    TRAIN_SPLITS_DIR = Path(f"./../../.../Splits/{interval_str}")
    if not TRAIN_SPLITS_DIR.exists():
        print(f"[ERROR] Train splits dir not found: {TRAIN_SPLITS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Construye los objetivos de test:
    # - Si NO hay --injected → solo test normal ("")
    # - Si HAY --injected → una entrada por cada variante (suffix = _inj<var>)
    test_targets: List[Tuple[str, Path]] = []
    if args.injected:
        for inj_label in args.injected:
            test_dir = Path(f"./../../User_Injected_{inj_label}/Splits/{interval_str}")
            if not test_dir.exists():
                print(f"[WARN] Test splits dir not found for injected '{inj_label}': {test_dir} (skipping)", file=sys.stderr)
                continue
            test_targets.append((f"_inj{inj_label}", test_dir))
        if not test_targets:
            print("[ERROR] No valid injected test folders found. Exiting.", file=sys.stderr)
            sys.exit(1)
    else:
        # Test normal
        test_targets.append(("", TRAIN_SPLITS_DIR))

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

    # Run for requested features & models
    for feature_label in args.features:
        feat_file = FEATURE_MAP[feature_label]
        with open(FEATURE_DIR / feat_file, "rb") as f:
            feat_names = pickle.load(f)

        # Standard (no PCA)
        for model_name in args.models:
            where_str = ", ".join([("Normal" if suf == "" else f"Injected({suf[4:]})")
                                   for suf, _ in test_targets])
            print(f"→ Saving preds for {model_name} + {feature_label} @ {interval_str} | Test targets: {where_str}")
            evaluate_model_multi(
                model_name,
                MODEL_REGISTRY[model_name],
                feat_names,
                feature_label,
                args.interval,
                train_dir=TRAIN_SPLITS_DIR,
                test_targets=test_targets,
                pca=None,
            )

        # PCA-40 on corr_var, if requested
        if feature_label == "features_corr_var":
            with open(PCA_FILE, "rb") as f:
                pca_model = pickle.load(f)
            pca_label = f"{feature_label}_PCA40"
            for model_name in args.models:
                where_str = ", ".join([("Normal" if suf == "" else f"Injected({suf[4:]})")
                                       for suf, _ in test_targets])
                print(f"→ Saving preds for {model_name} + {pca_label} @ {interval_str} | Test targets: {where_str}")
                evaluate_model_multi(
                    model_name,
                    MODEL_REGISTRY[model_name],
                    feat_names,
                    pca_label,
                    args.interval,
                    train_dir=TRAIN_SPLITS_DIR,
                    test_targets=test_targets,
                    pca=pca_model,
                )

    print("Done.")
