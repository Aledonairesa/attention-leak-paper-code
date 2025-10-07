import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from catboost import CatBoostClassifier

from utils.preprocessing_features import normalize_columns

# --- Utility functions ---

def calculate_classification_metrics(y_true, y_pred):
    """Return accuracy, precision, recall, and F1 score."""
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )


def sort_paths_by_difficulty(csv_paths):
    """Reorder paths by a fixed difficulty index sequence."""
    order = [18,17,19,16,10,9,11,8,2,1,3,0,14,13,15,12,6,5,7,4]
    if len(csv_paths) >= len(order):
        return [csv_paths[i] for i in order]
    return sorted(csv_paths)


def get_parameter_combinations(csv_paths):
    """Extract parameter-combo tokens from filenames."""
    combos = []
    for path in csv_paths:
        parts = path.stem.split('_')
        combos.append('_'.join(parts[-6:-3]))
    return combos


def get_csv_paths_by_prefix(data_dir: Path, prefix: str):
    """List CSVs matching prefix, sorted by difficulty."""
    paths = sorted(data_dir.glob(f"{prefix}*.csv"))
    return sort_paths_by_difficulty(paths)


def read_datasets(paths):
    """Load list of DataFrames from CSV paths."""
    return [pd.read_csv(p) for p in paths]


def combine_and_split_all(dfs, test_size=0.2, random_state=42):
    """Concat all dfs and split into train/test."""
    full = pd.concat(dfs, ignore_index=True)
    X = full.drop(columns=["target"])
    y = full["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_each_dataset(dfs, test_size=0.2, random_state=42):
    """Split each df individually into train/test."""
    splits = []
    for df in dfs:
        X = df.drop(columns=["target"])
        y = df["target"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
        splits.append({'X_train':X_tr,'X_test':X_te,'y_train':y_tr,'y_test':y_te})
    return splits


def train_catboost(X_train, y_train, extra_drops=None):
    """Normalize, drop cols, and train CatBoostClassifier."""
    cols = ['diff_lenframe_to_last','diff_lenframe_to_last_mean']
    X = normalize_columns(X_train.copy(), cols)
    X = X.drop(columns=['time_to_last_frame','length_previous'], errors='ignore')
    if extra_drops:
        X = X.drop(columns=extra_drops, errors='ignore')
    model = CatBoostClassifier(iterations=50, random_state=42, verbose=0)
    model.fit(X, y_train)
    return model


def evaluate_catboost(model, X_test, y_test, extra_drops=None):
    """Preprocess test set, predict and return metrics, cm, and importances."""
    cols = ['diff_lenframe_to_last','diff_lenframe_to_last_mean']
    X = normalize_columns(X_test.copy(), cols)
    X = X.drop(columns=['time_to_last_frame','length_previous'], errors='ignore')
    if extra_drops:
        X = X.drop(columns=extra_drops, errors='ignore')
    y_pred = model.predict(X)
    acc, prec, rec, f1_val = calculate_classification_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    imp = model.get_feature_importance()
    imp_df = pd.DataFrame({'Feature':X.columns,'Importance':imp}).sort_values('Importance',ascending=False)
    return acc, prec, rec, f1_val, cm, imp_df


def plot_and_save_confusion_matrix(cm, path: Path):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Negative','Pred Positive'],
                yticklabels=['Act Negative','Act Positive'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_and_save_specialized_heatmap(results, name, metric, labels, path: Path,
                                      vmin=0.6, vmax=1.0):
    vals = [d[metric] for d in results[list(results.keys())[-1]][name]]
    arr = np.array([vals])
    plt.figure(figsize=(max(6,len(vals)*0.5),1.5))
    sns.heatmap(arr, annot=True, fmt='.2f', annot_kws={'size':6},
                cmap='RdYlGn', cbar_kws={'label':'Value','ticks':np.linspace(vmin,vmax,5)},
                xticklabels=labels, yticklabels=[name], vmin=vmin, vmax=vmax)
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_global_metrics(metrics, path: Path):
    df = pd.DataFrame(metrics).T
    df.columns=['Accuracy','Precision','Recall','F1']
    df.index.name='Model'
    df=df.reset_index()
    mf = df.melt(id_vars='Model',var_name='Metric',value_name='Value')
    plt.figure(figsize=(10,5))
    ax=sns.barplot(data=mf,x='Metric',y='Value',hue='Model')
    plt.title('Global Metrics Comparison')
    for p in ax.patches:
        h=p.get_height()
        if np.isfinite(h) and h>0:
            ax.annotate(f"{h:.2f}",(p.get_x()+p.get_width()/2.,h),ha='center',va='bottom',fontsize=7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path,bbox_inches='tight')
    plt.close()


def plot_global_specialized(summary, labels, path: Path):
    models=list(summary.keys())
    arr=np.array([summary[m] for m in models])
    plt.figure(figsize=(max(6,len(labels)*0.5),len(models)*0.5+1))
    sns.heatmap(arr,annot=True,fmt='.2f',annot_kws={'size':6},
                cmap='RdYlGn',cbar_kws={'label':'F1 Value','ticks':np.linspace(0.6,1.0,5)},
                xticklabels=labels,yticklabels=models,vmin=0.6,vmax=1.0)
    plt.xlabel('Param Combination')
    plt.title('Specialized F1 Comparison')
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig(path,bbox_inches='tight')
    plt.close()


def compute_specialized_f1_scores(model, splits):
    return [evaluate_catboost(model,s['X_test'],s['y_test'])[3] for s in splits]


def main():
    # 1) Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train & evaluate CatBoost variants')
    parser.add_argument('data_dir', type=Path, help='CSV data directory')
    parser.add_argument('--output_dir', type=Path, default=Path('./results'), help='Results dir')
    parser.add_argument('--models', nargs='+', default=['base','equalized','asn_equalized','nogoogle_asn_equalized'],
                        help='Variants to run')
    args = parser.parse_args()

    # 2) Setup: prefixes and result containers
    prefixes = {
        'base': 'feat_dataset_', 'equalized': 'feat_equalized_', 
        'asn_equalized': 'feat_asn_equalized_', 'nogoogle_asn_equalized': 'feat_nogoogle_asn_equalized_' 
    }
    metrics_sum, spec_sum = {}, {}
    param_labels = None

    # 3) Train & evaluate each standard variant
    for m in tqdm(args.models, desc='Models'):
        print(f"Processing variant: {m}")
        csvs = get_csv_paths_by_prefix(args.data_dir, prefixes[m])
        datasets = read_datasets(csvs)
        if param_labels is None:
            param_labels = get_parameter_combinations(csvs)

        # 3a) Full-data split & training
        X_tr, X_te, y_tr, y_te = combine_and_split_all(datasets)
        model = train_catboost(X_tr, y_tr)

        # 3b) Save model & full-test evaluation
        out = args.output_dir / m
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / 'model.pkl')
        acc, prec, rec, f1_val, cm, imp_df = evaluate_catboost(model, X_te, y_te)
        metrics_sum[m] = [acc, prec, rec, f1_val]
        pd.DataFrame({'accuracy': [acc], 'precision': [prec], 'recall': [rec], 'f1': [f1_val]})\
            .to_csv(out / 'metrics.csv', index=False)
        imp_df.to_csv(out / 'feature_importance.csv', index=False)
        plot_and_save_confusion_matrix(cm, out / 'confusion_matrix.png')

        # 3c) Specialized evaluation
        splits = split_each_dataset(datasets)
        f1s = compute_specialized_f1_scores(model, splits)
        spec_sum[m] = f1s
        plot_and_save_specialized_heatmap({0: {m: [{'f1': v} for v in f1s]}},
                                          m, 'f1', param_labels, out / 'specialized.png')

    # 4) Hyperparameter-optimized variant
    print("\nStarting hyperparameter optimization on 'nogoogle_asn_equalized' data...")
    csvs = get_csv_paths_by_prefix(args.data_dir, prefixes['nogoogle_asn_equalized'])
    datasets = read_datasets(csvs)
    X_tr, X_te, y_tr, y_te = combine_and_split_all(datasets)

    # 4a) Setup search
    param_dist = {
        'learning_rate': uniform(0.05, 0.4),
        'depth': randint(5, 10),
        'l2_leaf_reg': randint(1, 9),
        'border_count': randint(32, 129)
    }
    base = CatBoostClassifier(iterations=50, random_state=42, verbose=0)
    rs = RandomizedSearchCV(base, param_dist, n_iter=100, cv=3,
                            scoring='f1', random_state=42,
                            verbose=1, n_jobs=-1)

    # Preprocess training data for hyperparameter tuning
    cols_to_norm = ['diff_lenframe_to_last','diff_lenframe_to_last_mean']
    X_tr_proc = normalize_columns(X_tr.copy(), cols_to_norm)
    X_tr_proc = X_tr_proc.drop(columns=['time_to_last_frame','length_previous'], errors='ignore')
    print("Running randomized search CV on preprocessed training data...")
    rs.fit(X_tr_proc, y_tr)

    best = rs.best_estimator_
    print(f"Best hyperparameters: {rs.best_params_}")

    # 4b) Save & evaluate tuned model
    h_out = args.output_dir / 'hyperopt'
    h_out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, h_out / 'model.pkl')
    acc, prec, rec, f1_val, cm, imp_df = evaluate_catboost(best, X_te, y_te)
    metrics_sum['hyperopt'] = [acc, prec, rec, f1_val]
    pd.DataFrame({'accuracy': [acc], 'precision': [prec], 'recall': [rec], 'f1': [f1_val]})\
        .to_csv(h_out / 'metrics.csv', index=False)
    imp_df.to_csv(h_out / 'feature_importance.csv', index=False)
    plot_and_save_confusion_matrix(cm, h_out / 'confusion_matrix.png')
    splits = split_each_dataset(datasets)
    f1s = compute_specialized_f1_scores(best, splits)
    spec_sum['hyperopt'] = f1s
    plot_and_save_specialized_heatmap({0: {'hyperopt': [{'f1': v} for v in f1s]}},
                                      'hyperopt', 'f1', param_labels, h_out / 'specialized.png')

    # 5) Generate global comparison plots
    print("\nGenerating global comparison plots...")
    plot_global_metrics(metrics_sum, args.output_dir / 'metrics_comparison.png')
    plot_global_specialized(spec_sum, param_labels, args.output_dir / 'specialized_comparison.png')


if __name__ == '__main__':
    main()
