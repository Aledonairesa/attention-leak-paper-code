import os
import glob
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Configuration
SEED = 42
BASE_DIR = os.path.join("Users", "Alice", "Splits", "5.0s")
OUTPUT_DIR = "select_features_results"
EXCLUDE_FILENAME = "split_50.csv"
N_SELECT = 6
CORR_THRESHOLD = 0.95
VAR_THRESHOLD = 0.01  # 1% variance threshold on [0,1] scaled data
PCA_COMPONENTS = 40

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. List and select random split files
all_files = sorted(glob.glob(os.path.join(BASE_DIR, "split_*.csv")))
# Exclude the 50th file
candidates = [f for f in all_files if not f.endswith(EXCLUDE_FILENAME)]

selected = random.sample(candidates, N_SELECT)
print("Selected files:")

with open("holdout_splits.txt", "w") as f_out:
    for f in selected:
        print(f)
        f_out.write(f + "\n")

# 2. Load and merge selected splits
df_list = [pd.read_csv(f) for f in selected]
merged_df = pd.concat(df_list, ignore_index=True)

# 3. Save all feature names (excluding time and target columns)
exclude_cols = ["time", "num_tasks", "num_browser_tasks", "num_non_browser_tasks"]
features_all = [c for c in merged_df.columns if c not in exclude_cols]
with open(os.path.join(OUTPUT_DIR, "features_all.pkl"), "wb") as f:
    pickle.dump(features_all, f)
print(f"Number of features saved in features_all.pkl: {len(features_all)}")

# 4. Prepare feature dataframe
features_df = merged_df[features_all]

# 5. Correlation-based filtering (z-score normalization)
normalized_for_corr = (features_df - features_df.mean()) / features_df.std()
corr_matrix = normalized_for_corr.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
features_corr_filtered = [f for f in features_all if f not in to_drop_corr]

# 6. Variance-based filtering (min-max scaling then VarianceThreshold)
scaled = features_df[features_corr_filtered].copy()
min_vals = scaled.min()
max_vals = scaled.max()
# Clip extreme values to reduce outlier impact
q_low = 0.01
q_high = 0.99
scaled = scaled.clip(lower=scaled.quantile(q_low), upper=scaled.quantile(q_high), axis=1)
# Min-max scale to [0,1]
scaled = (scaled - min_vals) / (max_vals - min_vals)

var_selector = VarianceThreshold(threshold=VAR_THRESHOLD)
var_selector.fit(scaled)
selected_mask = var_selector.get_support()
features_final = list(np.array(features_corr_filtered)[selected_mask])

# 7. Save final feature list and print summary
with open(os.path.join(OUTPUT_DIR, "features_corr_var.pkl"), "wb") as f:
    pickle.dump(features_final, f)
print(f"Features removed due to correlation > {CORR_THRESHOLD}: {len(to_drop_corr)}")
print(f"Features removed due to low variance < {VAR_THRESHOLD*100}%: {len(features_corr_filtered) - len(features_final)}")
print(f"Total features remaining: {len(features_final)}")

# 8. Identify features removed specifically due to low-variance filtering
to_drop_var = [f for f in features_corr_filtered if f not in features_final]

# Randomly sample features for plotting
random.seed(SEED)
num_plots = 8
plot_kept = random.sample(features_final, min(num_plots, len(features_final)))
plot_removed = random.sample(to_drop_var, min(num_plots, len(to_drop_var)))

# 2x8 grid of distributions
fig, axes = plt.subplots(2, num_plots, figsize=(num_plots * 3, 8))

# Top row: distributions of non-filtered (kept) features
for i, feat in enumerate(plot_kept):
    ax = axes[0, i]
    ax.hist(scaled[feat].dropna(), bins=30)
    ax.set_title(f"Kept: {feat}")
for j in range(len(plot_kept), num_plots):
    axes[0, j].axis('off')

# Bottom row: distributions of low-variance-removed features
for i, feat in enumerate(plot_removed):
    ax = axes[1, i]
    ax.hist(scaled[feat].dropna(), bins=30)
    ax.set_title(f"Removed: {feat}")
for j in range(len(plot_removed), num_plots):
    axes[1, j].axis('off')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "variance_filter_distributions.png"))
print("Saved distribution plot to variance_filter_distributions.png")

# 9. PCA on selected features (with z-score normalization)
data_for_pca = features_df[features_final]
data_for_pca = (data_for_pca - data_for_pca.mean()) / data_for_pca.std()
n_components = min(PCA_COMPONENTS, data_for_pca.shape[1])
pca = PCA(n_components=n_components, random_state=SEED)
pca.fit(data_for_pca)
with open(os.path.join(OUTPUT_DIR, "pca_40_components.pkl"), "wb") as f:
    pickle.dump(pca, f)
print(f"Saved PCA ({n_components} components) to pca_40_components.pkl")

# 10. Plot cumulative explained variance and mark selected cutoff
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o')
cutoff = n_components if n_components < PCA_COMPONENTS else PCA_COMPONENTS
plt.axvline(cutoff, color='grey', linestyle='--')
var_at_n = cumulative[cutoff - 1] * 100
plt.text(cutoff + 1, cumulative[cutoff - 1], f"{var_at_n:.2f}% at {cutoff}", va='center')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_explained_variance.png'))
print("Saved PCA explained variance plot to pca_explained_variance.png")

# 11. Recursive Feature Elimination with Random Forest
X_rfe = merged_df[features_final]
y_rfe = merged_df['num_tasks']
rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rfe = RFE(estimator=rf, n_features_to_select=40, step=1)
rfe.fit(X_rfe, y_rfe)
selected_rfe = list(X_rfe.columns[rfe.support_])
with open(os.path.join(OUTPUT_DIR, "features_RFE.pkl"), "wb") as f:
    pickle.dump(selected_rfe, f)
print(f"Saved RFE-selected features ({len(selected_rfe)}) to features_RFE.pkl")
