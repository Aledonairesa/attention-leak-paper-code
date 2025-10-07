# Prepare data for training

This folder provides a minimal pipeline to convert raw app CSVs into interval activity features and fixed-duration splits.

---

## 1) Preprocess raw CSVs

**Script:** `preprocess_app_data.py`  
**Purpose:** read `Users/<User>/Raw/traces_<i>.csv` and `timestamps_<i>.csv`; clean, align times to a common zero, and save to `Users/<User>/Preprocessed/`.

**Run:**
```bash
python preprocess_app_data.py --users User1 User2 --files 1 2 3
```

**Input:** `Users/<User>/Raw/traces_<i>.csv`, `timestamps_<i>.csv`  
**Output:** `Users/<User>/Preprocessed/traces_<i>.csv`, `timestamps_<i>.csv`

---

## 2) Extract interval activity (+180 features) per file

**Script:** `extract_activity.py`  
**Purpose:** build interval-binned activity tables from Preprocessed traces + timestamps; writes `activity_<i>.csv` under the interval folder.

**Run (example for 5 s bins):**
```bash
python extract_activity.py --users User1 User2 --files 1 2 3 --interval 5
```

**Input:** `Users/<User>/Preprocessed/traces_<i>.csv`, `timestamps_<i>.csv`  
**Output:** `Users/<User>/Activity/<interval>s/activity_<i>.csv`

---

## 3) Generate fixed-duration splits from activity

**Script:** `generate_splits.py`  
**Purpose:** merge all `activity_*.csv` for a user and split into **equal-duration segments**; writes `split_k.csv` under `Splits/<interval>s/`. Also serves to create **train–test partitions per user** for downstream experiments.

**Run (e.g., 60-minute splits over 5 s activity):**
```bash
python generate_splits.py --users User1 User2 --interval 5 --duration 60
```

**Input:** `Users/<User>/Activity/<interval>s/activity_*.csv`  
**Output:** `Users/<User>/Splits/<interval>s/split_<k>.csv`
