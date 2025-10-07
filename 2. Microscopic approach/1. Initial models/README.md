# Initial Models

The script `initial_models.py` is used to evaluate how well different out-of-the-box models perform.  
It generates:

- **Generalization matrices**  
- **Intra-dataset plots**  

These assess how accurately the models can predict whether a given frame belongs to the same web connection as a group of previous frames.

---

## Usage

Run the script with:

1. The path to the processed datasets (CSV files).  
2. *(Optional)* A directory where plots will be saved.  

Example:

```bash
python initial_models.py "./../../Data generation - Docker/3. Preprocessing and feature extraction/dataset_1/" --plots_dir "./plots"
```
