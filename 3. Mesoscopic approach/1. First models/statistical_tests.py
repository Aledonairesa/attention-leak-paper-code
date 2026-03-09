import pickle
import numpy as np
from scipy import stats
from pathlib import Path

# --- 1. File Paths ---
PKL_DIR = Path("1. First models/prediction_pkls/5.0s")
TCN_FILE = PKL_DIR / "TCN_features_all.pkl"
XGB_FILE = PKL_DIR / "XGB_features_all.pkl"

def load_and_aggregate(pkl_path):
    """Loads a prediction dict and sums the intervals to get split-level totals."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    totals_dict = {}
    for split_key, values in data.items():
        # Sum the raw interval values to get the total task switches per split
        gt_total = sum(values["true"])
        pred_total = sum(values["pred"])
        totals_dict[split_key] = {"gt": gt_total, "pred": pred_total}
        
    return totals_dict

# --- 2. Load the Data ---
tcn_data = load_and_aggregate(TCN_FILE)
xgb_data = load_and_aggregate(XGB_FILE)

# Ensure both files have the exact same splits
assert set(tcn_data.keys()) == set(xgb_data.keys()), "Splits do not match between TCN and XGB files!"

# --- 3. Build the Aligned Arrays ---
gt_list = []
tcn_pred_list = []
xgb_pred_list = []

# Sort keys to ensure consistent ordering, though dict matching is enough
for split_key in sorted(tcn_data.keys()):
    # Ground truth is the same for both, we can just grab it from TCN
    gt_list.append(tcn_data[split_key]["gt"]) 
    tcn_pred_list.append(tcn_data[split_key]["pred"])
    xgb_pred_list.append(xgb_data[split_key]["pred"])

gt = np.array(gt_list)
pred_tcn = np.array(tcn_pred_list)
pred_xgb = np.array(xgb_pred_list)

N = len(gt)
print(f"Loaded data for {N} evaluation splits.\n")

# --- 4. Wilcoxon Signed-Rank Test for MAE ---
# Calculate absolute errors per split
abs_err_tcn = np.abs(pred_tcn - gt)
abs_err_xgb = np.abs(pred_xgb - gt)

# Perform the Wilcoxon test
w_stat, p_val_wilcoxon = stats.wilcoxon(abs_err_tcn, abs_err_xgb)
print(f"--- MAE Comparison ---")
print(f"Mean TCN MAE: {np.mean(abs_err_tcn):.2f}")
print(f"Mean XGB MAE: {np.mean(abs_err_xgb):.2f}")
print(f"Wilcoxon Test: W = {w_stat:.2f}, p-value = {p_val_wilcoxon:.4f}\n")

# --- 5. Steiger's Z-test for Dependent Correlations ---
def steigers_z(r12, r13, r23, n):
    """
    Computes Steiger's Z-test for dependent correlations.
    r12: correlation of GT and Model 1 (TCN)
    r13: correlation of GT and Model 2 (XGBoost)
    r23: correlation between Model 1 and Model 2
    """
    # Fisher's r-to-z transformation
    z12 = 0.5 * np.log((1 + r12) / (1 - r12))
    z13 = 0.5 * np.log((1 + r13) / (1 - r13))
    
    # Covariance of the two correlations
    c = (r23 * (1 - r12**2 - r13**2) - 0.5 * r12 * r13 * (1 - r12**2 - r13**2 - r23**2)) / ((1 - r12**2) * (1 - r13**2))
    
    # Steiger's Z statistic
    z_stat = (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - c)))
    
    # Two-tailed p-value
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_val

# Calculate the three required correlations
r_gt_tcn, _ = stats.pearsonr(gt, pred_tcn)
r_gt_xgb, _ = stats.pearsonr(gt, pred_xgb)
r_tcn_xgb, _ = stats.pearsonr(pred_tcn, pred_xgb)

z_stat, p_val_steiger = steigers_z(r_gt_tcn, r_gt_xgb, r_tcn_xgb, N)

print(f"--- Pearson's r Comparison ---")
print(f"TCN r: {r_gt_tcn:.4f}")
print(f"XGB r: {r_gt_xgb:.4f}")
print(f"Steiger's Z-test: Z = {z_stat:.3f}, p-value = {p_val_steiger:.4f}")