# Model Optimization

The script `model_optimization.py` is used to train the **CatBoost** model with different preprocessing methods, and to run a final hyperparameter optimization using the `nogoogle_asn_equalized` dataset.  

By default, it evaluates the following variants:  

- `base`  
- `equalized`  
- `asn_equalized`  
- `nogoogle_asn_equalized`  

---

## Usage

Run the script with:

1. The path to the processed datasets (CSV files).  
2. *(Optional)* An output directory where results will be saved.  

Example:

```bash
python model_optimization.py "./../../Data generation - Docker/3. Preprocessing and feature extraction/dataset_1/" --output_dir "./results"
```
