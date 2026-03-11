# Task Untangling

The main script `untangle_tasks.py` runs experiments across different models and `alpha/beta` parameter pairs.  

---

## Usage

Run the script by providing a configuration file:

```bash
python untangle_tasks.py config.json
```

The `config.json` file defines:

- The models to test  
- The `alpha/beta` parameter pairs  
- How the web traffic simulation dataset is generated  
- Which websites are included  
- The time shifts between websites  

---

## Example Configuration

Here is an example `config.json`:

```json
{
  "data_dir": "C:/.../data/filtered_raw_dataset_1/",
  "output_dir": "./results",
  "websites": ["bbc.com", "elmundo.es", "orange.fr", "tiktok.com", "booking.com"],
  "time_shifts": [1.2, 2.5, 3.8, 3.8],
  "plot_generated_dataset": true,
  "model_path": [
    ["Base", "./../3. Model optimization/results/base/model.pkl"],
    ["Equalized", "./../3. Model optimization/results/equalized/model.pkl"],
    ["Hyperopt", "./../3. Model optimization/results/hyperopt/model.pkl"]
  ],
  "alpha_beta_pairs": [[0.50, 0.50], [0.10, 0.50], [0.10, 0.90], [0.01, 0.99]]
}
```

---

## Plotting Results

After running `untangle_tasks.py`, you can visualize the results using `plot_global_results.py`.  

Generate global heatmaps:

```bash
python plot_global_results.py "./results"
```

Plot the evolution of a specific metric for a given `alpha/beta` pair by specifying `--alpha`, `--beta`, and `--evolution_metric`:

```bash
python plot_global_results.py "./results" --alpha 0.02 --beta 0.98 --evolution_metric pair_f1
```

---

# Mitigation experiments

The files `inject_mixed_df.py`, `config_injection.json`, and `untangle_tasks_injection.py` correspond to the mitigation experiments for this approach. To run them, first adjust the parameters as necessary in `config_injection.json`, then run `untangle_tasks_injection.py` as `untangle_tasks.py`. The results will be saved in the same manner.
