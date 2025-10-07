import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def calculate_classification_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1 score."""
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )


def get_csv_paths(datasets_dir):
    """Return sorted list of CSV file paths by difficulty ordering."""
    paths = [
        os.path.join(datasets_dir, f)
        for f in os.listdir(datasets_dir)
        if f.endswith('.csv') and f.startswith('feat_dataset_')
    ]
    order = [18, 17, 19, 16, 10, 9, 11, 8, 2, 1, 3, 0, 14, 13, 15, 12, 6, 5, 7, 4]
    return [paths[i] for i in order if i < len(paths)]


def read_and_split(csv_paths):
    """Read CSVs, split into train/test, and collect labels."""
    datasets = []
    labels = []
    for path in csv_paths:
        df = pd.read_csv(path)
        labels.append('_'.join(os.path.basename(path).split('_')[-6:-3]))
        X = df.drop(columns=['target'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        datasets.append({
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
        })
    return datasets, labels


def plot_heatmap(values, labels, title, samples, out_file, vmin=0.6, vmax=1.0, cmap='RdYlGn'):
    """Plot and save a single-row heatmap with fixed color scale."""
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow([values], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    for i, v in enumerate(values):
        ax.text(i, 0, f'{v:.2f}', ha='center', va='center')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=80)
    ax.set_yticks([])
    ax.set_title(f'{title} (samples={samples})')
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    im.set_clim(vmin, vmax)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate specialized and combined heatmaps.')
    parser.add_argument('datasets_dir', help='Path to datasets directory')
    parser.add_argument('--plots_dir', default='./plots', help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    print(f"Loading datasets from {args.datasets_dir}...")

    csv_paths = get_csv_paths(args.datasets_dir)
    print(f"Found {len(csv_paths)} CSV files.")
    data_splits, labels = read_and_split(csv_paths)

    X_train = pd.concat([d['X_train'] for d in data_splits], ignore_index=True)
    y_train = pd.concat([d['y_train'] for d in data_splits], ignore_index=True)
    samples = X_train.shape[0]
    print(f"Combined training samples: {samples}")

    classifiers = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    }
    metric = 'f1'
    idx = 3
    results = {}

    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        print(f"Evaluating {name}...")
        vals = []
        for ds in tqdm(data_splits, desc=f"{name}", unit="dataset"):
            y_pred = clf.predict(ds['X_test'])
            vals.append(calculate_classification_metrics(ds['y_test'], y_pred)[idx])
        out_file = os.path.join(args.plots_dir, f'{name.lower()}_{metric}.png')
        plot_heatmap(vals, labels, name, samples, out_file)
        results[name] = vals
        print(f"Saved {name} plot to {out_file}")

    print("Generating combined heatmap...")
    n_models = len(classifiers)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 2.8 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]
    for ax, (name, vals) in zip(axes, results.items()):
        im = ax.imshow([vals], aspect='auto', cmap='RdYlGn', vmin=0.6, vmax=1.0)
        for i, v in enumerate(vals):
            ax.text(i, 0, f'{v:.2f}', ha='center', va='center')
        ax.set_yticks([])
        ax.set_title(name)
    axes[-1].set_xticks(range(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=80)

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.15, label='Value')
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.25, hspace=0.3)

    combo_path = os.path.join(args.plots_dir, f'combined_{metric}.png')
    fig.savefig(combo_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {combo_path}")


if __name__ == '__main__':
    main()
