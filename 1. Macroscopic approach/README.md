## 1. Create Mixed Datasets

The script `create_mixed_datasets.py` produces the `mixed_datasets.pkl` file, which contains all the dataset mixtures of web connections used for testing the functions.

Example:

```bash
python create_mixed_datasets.py --raw-data-dir "./../data/filtered_raw_dataset_1/" --save-path "." --config config.json
```

The `config.json` defines parameters for dataset generation. Example:

```json
{
  "min_tasks": 4,
  "max_tasks": 15,
  "time_extension": 15,
  "min_mixing": 0,
  "max_mixing": 60,
  "by_mixing": 10,
  "max_iter": 30,
  "seed": 1
}
```

---

## 2. Train Main IP Model

The script `main_IPs.py` creates the model `main_ip_model.pkl`, used in one of the tested functions.  
This model is trained on manually annotated data (`main_IPs.pkl`) and a set of filtered raw datasets.

Example:

```bash
python main_IPs.py main_IPs.pkl "./../data/filtered_raw_dataset_1"
```

---

## 3. Test Functions

The script `test_functions.py` evaluates all the functions defined in `functions.py`.  
It produces:

- A set of plots for each function  
- A final summary with metrics for all functions  

Example:

```bash
python test_functions.py
```

You can also specify which functions to test using the `--functions` argument. By default, all functions are evaluated.



