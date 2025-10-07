# Feature Extraction

The script `full_feature_extraction.py` generates the features used by the microscopic approach models.  
It depends on:

- `extract_features.py`  
- `generate_dataset.py`  
- utility scripts inside the `utils` folder  

---

## Usage

To run `full_feature_extraction.py`, you must provide:

1. The path to the raw data folders produced by the web calls.  
2. An output directory where the final features and intermediate `.pkl` files will be stored.  
3. A `config.json` file defining the hyperparameters.  

Example:

```bash
python full_feature_extraction.py \
  --datasets_path ./../data/filtered_raw_dataset_1 \
  --output_folder ./dataset_1 \
  --config_file config.json
```

---

## Default Configuration

The default `config.json` used to generate the datasets looks like this:

```json
{
  "num_samples": 16000,
  "num_webs_list": [80, 40, 10, 5, 1],
  "num_datasets_per_web_list": [1, 2, 8, 16, 80],
  "levels_list": [[50], [20], [5], [1]],
  "level_weights": [1],
  "time_between_tasks": 1,
  "seed": 42,
  "equalize_num_size": 3000,
  "to_filter_range_list": [
    ["142.250.0.0", "142.251.255.255"]
  ]
}
```
