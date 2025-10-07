import os
import glob
import argparse
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Silence warnings
warnings.filterwarnings('ignore')

# Reproducibility
import tensorflow as tf

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

# Sklearn and other model imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# User utilities
from utils.preprocessing_features import normalize_columns


def get_csv_paths(datasets_dir: str) -> list:
    """
    Return sorted list of CSV file paths matching 'feat_dataset_*.csv' in the directory.
    """
    pattern = os.path.join(datasets_dir, "feat_dataset_*.csv")
    return sorted(glob.glob(pattern))


def sort_paths_by_difficulty(csv_paths: list) -> list:
    """
    Reorder csv paths by a predefined difficulty index.
    """
    order_idxs = [18, 17, 19, 16, 10, 9, 11, 8, 2, 1, 3, 0, 14, 13, 15, 12, 6, 5, 7, 4]
    return [csv_paths[i] for i in order_idxs]


def get_parameter_combinations(csv_paths: list) -> list:
    """
    Extract parameter combination names from filenames: remove prefix and extension.
    """
    names = []
    for path in csv_paths:
        base = os.path.basename(path)
        name = os.path.splitext(base)[0].replace("feat_dataset_", "")
        names.append(name)
    return names


def load_datasets(csv_paths: list) -> list:
    """
    Load CSV datasets into pandas DataFrames with progress bar.
    """
    return [pd.read_csv(p) for p in tqdm(csv_paths, desc="Loading datasets", unit="file")]


def split_datasets_in_train_test(datasets: list, test_size: float = 0.2, random_state: int = 42) -> list:
    """
    Perform train-test split on each DataFrame, returning a list of dicts with X_train, X_test, y_train, y_test.
    """
    split = []
    for df in tqdm(datasets, desc="Splitting datasets", unit="dataset"):
        X = df.drop(columns=["target"])
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        split.append({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test})
    return split


def calculate_classification_metrics(y_true, y_pred):
    """Calculate and return accuracy, precision, recall, and F1 score."""
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )

# Model functions (train: returns model; eval: returns metrics + importances/data)
def logistic_regression(X_train, X_test, y_train, y_test, train=True, model=None):
    cols = ['diff_lenframe_to_last', 'diff_lenframe_to_last_mean']
    X_train = normalize_columns(X_train, cols)
    X_test  = normalize_columns(X_test,  cols)
    # drop only if present
    drop_cols = ['time_to_last_frame', 'length_previous']
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_test  = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    if train:
        m = LogisticRegression(max_iter=1000, random_state=42)
        m.fit(X_train, y_train)
        return m
    y_pred = model.predict(X_test)
    return *calculate_classification_metrics(y_test, y_pred), None


def random_forest(X_train, X_test, y_train, y_test, train=True, model=None):
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),    columns=X_test.columns)
    if train:
        m = RandomForestClassifier(random_state=42)
        m.fit(X_train, y_train)
        return m
    y_pred = model.predict(X_test)
    acc, prec, rec, f1 = calculate_classification_metrics(y_test, y_pred)
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    imp_df = imp.sort_values(ascending=False).reset_index()
    imp_df.columns = ['feature', 'importance']
    return acc, prec, rec, f1, imp_df


def svm_factory(kernel):
    def _svm(X_train, X_test, y_train, y_test, train=True, model=None):
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),    columns=X_test.columns)
        if train:
            m = SVC(kernel=kernel, probability=True, random_state=42)
            m.fit(X_train, y_train)
            return m
        y_pred = model.predict(X_test)
        return *calculate_classification_metrics(y_test, y_pred), None
    return _svm


def xgboost_model(X_train, X_test, y_train, y_test, train=True, model=None):
    if train:
        m = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42)
        m.fit(X_train, y_train)
        return m
    y_pred = model.predict(X_test)
    acc, prec, rec, f1 = calculate_classification_metrics(y_test, y_pred)
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    df = imp.sort_values(ascending=False).reset_index()
    df.columns = ['feature', 'importance']
    return acc, prec, rec, f1, df


def lightgbm_model(X_train, X_test, y_train, y_test, train=True, model=None):
    if train:
        m = LGBMClassifier(verbose=-1, random_state=42)
        m.fit(X_train, y_train)
        return m
    y_pred = model.predict(X_test)
    acc, prec, rec, f1 = calculate_classification_metrics(y_test, y_pred)
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    df = imp.sort_values(ascending=False).reset_index()
    df.columns = ['feature', 'importance']
    return acc, prec, rec, f1, df


def catboost_model(X_train, X_test, y_train, y_test, train=True, model=None):
    if train:
        m = CatBoostClassifier(verbose=False, random_seed=42)
        m.fit(X_train, y_train)
        return m
    y_pred = model.predict(X_test)
    acc, prec, rec, f1 = calculate_classification_metrics(y_test, y_pred)
    imp = pd.Series(model.get_feature_importance(), index=X_train.columns)
    df = imp.sort_values(ascending=False).reset_index()
    df.columns = ['feature', 'importance']
    return acc, prec, rec, f1, df


def bernoulli_nb(X_train, X_test, y_train, y_test, train=True, model=None):
    if train:
        m = BernoulliNB()
        m.fit(X_train, y_train)
        return m
    y_pred = model.predict(X_test)
    return *calculate_classification_metrics(y_test, y_pred), None


def neural_network(X_train, X_test, y_train, y_test, train=True, model=None):
    if train:
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        y_cat = to_categorical(y_train, num_classes)
        m = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        m.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        m.fit(X_train, y_cat, epochs=20, batch_size=32, verbose=0)
        return m
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return *calculate_classification_metrics(y_test, y_pred), None


def evaluate_models(split_data: list, model_funcs: dict) -> dict:
    """
    Train each model on each dataset and evaluate on all. Track progress.
    """
    results = {}
    for name, func in tqdm(model_funcs.items(), desc="Evaluating models", unit="model"):
        per_model = []
        for train_set in tqdm(split_data, desc=f"Training {name}", unit="dataset", leave=False):
            mdl = func(train_set['X_train'], train_set['X_test'], train_set['y_train'], train_set['y_test'], train=True)
            row = []
            for test_set in tqdm(split_data, desc=f"Testing {name}", unit="dataset", leave=False):
                row.append(func(test_set['X_train'], test_set['X_test'], test_set['y_train'], test_set['y_test'], train=False, model=mdl))
            per_model.append(row)
        results[name] = per_model
    return results


def plot_generalization_matrix(name: str, model_results: list, metric: str, parameter_names: list, cmap: str = 'viridis', out_dir: str = 'plots'):
    os.makedirs(out_dir, exist_ok=True)
    n = len(model_results)
    data = np.zeros((n, n))
    idx = {'accuracy':0, 'precision':1, 'recall':2, 'f1':3}[metric]
    for i, row in enumerate(model_results):
        for j, res in enumerate(row):
            data[i, j] = res[idx]
    plt.figure(figsize=(max(6, n*0.5), max(6, n*0.5)))
    sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap,
                annot_kws={"size":6},
                xticklabels=parameter_names, yticklabels=parameter_names,
                vmin=0.6, vmax=1)
    plt.title(f"Generalization matrix: {name} ({metric})")
    plt.xlabel("Test datasets")
    plt.ylabel("Train datasets")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_{metric}_gen_matrix.png"))
    plt.close()


def plot_intra_dataset_performance_table(results: dict, metric: str, parameter_names: list, cmap: str = 'viridis', out_dir: str = 'plots'):
    os.makedirs(out_dir, exist_ok=True)
    model_names = list(results.keys())
    idx = {'accuracy':0, 'precision':1, 'recall':2, 'f1':3}[metric]
    intra = []
    for name in model_names:
        res = results[name]
        diag = [res[i][i][idx] for i in range(len(res))]
        intra.append(diag)
    data = np.array(intra).T  # datasets x models
    plt.figure(figsize=(max(8, len(model_names)*0.8), len(parameter_names)*0.6))
    sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap,
                annot_kws={"size":6},
                xticklabels=model_names, yticklabels=parameter_names,
                vmin=0.6, vmax=1)
    plt.xlabel("Model")
    plt.ylabel("Dataset")
    plt.title(f"Intra-dataset test results ({metric})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"intra_dataset_{metric}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Model creation and evaluation script")
    parser.add_argument('datasets_dir', type=str, help='Directory with feature CSVs')
    parser.add_argument('--plots_dir',  type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()

    csv_paths = get_csv_paths(args.datasets_dir)
    csv_paths = sort_paths_by_difficulty(csv_paths)
    parameter_names = get_parameter_combinations(csv_paths)
    datasets = load_datasets(csv_paths)
    split_data = split_datasets_in_train_test(datasets)

    model_funcs = {
        'Logistic Regression': logistic_regression,
        'Random Forest':       random_forest,
        'SVM Linear':          svm_factory('linear'),
        'SVM Poly':            svm_factory('poly'),
        'SVM RBF':             svm_factory('rbf'),
        'XGBoost':             xgboost_model,
        'LightGBM':            lightgbm_model,
        'CatBoost':            catboost_model,
        'Bernoulli NB':        bernoulli_nb,
        'Neural Network':      neural_network
    }

    results = evaluate_models(split_data, model_funcs)

    # plot generalization matrices
    for name, res in results.items():
        plot_generalization_matrix(name, res, metric='f1', parameter_names=parameter_names,
                                   cmap='RdYlGn', out_dir=args.plots_dir)

    # plot intra-dataset performance for F1
    plot_intra_dataset_performance_table(results, metric='f1', parameter_names=parameter_names,
                                         cmap='RdYlGn', out_dir=args.plots_dir)

if __name__ == '__main__':
    main()
