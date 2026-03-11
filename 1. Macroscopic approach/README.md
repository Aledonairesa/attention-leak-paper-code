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
- A CSV file with intermediate results.

Example:

```bash
python test_functions.py
```

You can also specify which functions to test using the `--functions` argument. By default, all functions are evaluated.

## 4. Statistical significance tests
The script `statistical_test.py` performs Steiger's Z-test on the most promising candidate functions (Main-IP, Start (10w), Start (4c)) using the pooled datasets (N=84). It uses the CSV file with the intermediate results, and can be executed as follows:

```bash
python statistical_test.py
```

## 5. Mitigation experiments
To run the mitigation experiments on this approach, first run `inject_frames.py` to make the injections in the dataset (creating a new one), as follows:

```bash
python --input .\mixed_datasets.pkl --output .\mixed_datasets_injected_250.pkl --injections 250 --len-spread 3.0
```

Then, with this new injected dataset, run `test_functions.py` as usual, specifiying with the argument `--data-path` the path to the injected dataset. The results will be saved in the same manner.
