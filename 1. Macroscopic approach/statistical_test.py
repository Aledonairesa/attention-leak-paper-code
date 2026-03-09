import pandas as pd
import numpy as np
import scipy.stats as stats

def steiger_z(r_xy, r_xz, r_yz, n):
    """
    Steiger's Z-test to compare two dependent correlations (r_xy and r_xz)
    that share a common baseline variable (x).
    
    Parameters:
    r_xy : Correlation between the truth (X) and model 1 (Y)
    r_xz : Correlation between the truth (X) and model 2 (Z)
    r_yz : Correlation between model 1 (Y) and model 2 (Z)
    n    : Sample size
    
    Returns:
    z    : Steiger's Z statistic
    p    : Two-tailed p-value
    """
    # Fisher's z transformations
    z_xy = 0.5 * np.log((1 + abs(r_xy)) / (1 - abs(r_xy)))
    z_xz = 0.5 * np.log((1 + abs(r_xz)) / (1 - abs(r_xz)))
    
    # Intermediate calculations for asymptotic variance
    rm2 = (r_xy**2 + r_xz**2) / 2.0
    f = (1 - r_yz) / (2 * (1 - rm2))
    h = (1 - f * rm2) / (1 - rm2)
    c = (r_yz * (1 - rm2) - 0.5 * r_xy * r_xz * (1 - r_xy**2 - r_xz**2 - r_yz**2)) / ((1 - r_xy**2) * (1 - r_xz**2))
    
    # Calculate Z and two-tailed p-value
    z = (z_xy - z_xz) * np.sqrt((n - 3) / (2 * (1 - c) * h))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p

def main():
    # Load data and adjust format
    df = pd.read_csv('results/mixing_correlation_results/all_intermediate_results.csv')
    df_pivoted = df.pivot(
        index=['task_index', 'mixing_pct'], 
        columns='func_name', 
        values='value'
    ).reset_index()

    # Extract the variables
    n = len(df_pivoted) # Total datasets
    truth = df_pivoted['task_index']
    main_ip = df_pivoted['num_main_ips']
    start_10w = df_pivoted['num_start_matches']
    start_4c = df_pivoted['num_start_matches_cons']

    # Calculate the necessary Pearson correlations
    r_truth_main, _ = stats.pearsonr(truth, main_ip)
    r_truth_10w, _  = stats.pearsonr(truth, start_10w)
    r_truth_4c, _   = stats.pearsonr(truth, start_4c)
    r_main_10w, _   = stats.pearsonr(main_ip, start_10w)
    r_main_4c, _    = stats.pearsonr(main_ip, start_4c)

    # Perform Steiger's Z-test for both comparisons
    z_10w, p_10w = steiger_z(r_truth_main, r_truth_10w, r_main_10w, n)
    z_4c, p_4c   = steiger_z(r_truth_main, r_truth_4c, r_main_4c, n)

    # Print the results
    print(f"--- Steiger's Z-test Results (Pooled Data, N={n}) ---")
    print(f"Main-IP vs Start (10w) : Z = {z_10w:.3f}, p-value = {p_10w:.3f}")
    print(f"Main-IP vs Start (4c)  : Z = {z_4c:.3f}, p-value = {p_4c:.3f}")

if __name__ == "__main__":
    main()