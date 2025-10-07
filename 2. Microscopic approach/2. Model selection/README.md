# Model Selection

The script `model_selection.py` performs a deeper exploration of the models' capabilities for **packet-by-packet classification**.  
It works by:

1. Creating a **mixed dataset** for training.  
2. Testing on **specialized datasets** to evaluate generalization.  
3. Running experiments with the three most promising models: **XGBoost**, **CatBoost**, and **LightGBM**.  

---

## Usage

Run the script with:

1. The path to the processed datasets (CSV files).  
2. *(Optional)* A directory where plots will be saved.  

Example:

```bash
python model_selection.py "./../../Data generation - Docker/3. Preprocessing and feature extraction/dataset_1/" --plots_dir "./plots"
```

