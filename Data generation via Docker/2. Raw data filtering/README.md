# Filtering the Raw Dataset

Use the `filter_raw_dataset.py` script to clean the raw Docker-generated data.  
The filtering is done in two steps:

1. Keep only **connection datasets** with more than `150` frames (these are considered *good* connections).  
2. Keep only **websites** that have more than `80` good connection datasets.  

---

## Usage

Run the script with the following arguments:

```bash
python filter_raw_dataset.py "path/to/raw_data_1" 150 80 "output/path/to/filtered_raw_data_1"
```

- `path/to/raw_data_1`: path to the original dataset.  
- `150`: minimum number of frames for a connection dataset to be considered *good*.  
- `80`: minimum number of good connections required for a website to be kept.  
- `output/path/to/filtered_raw_data_1`: directory where the filtered dataset will be saved.  

Change the paths as needed for your setup.
