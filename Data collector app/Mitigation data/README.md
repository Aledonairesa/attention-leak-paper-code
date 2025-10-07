# Mitigation strategy data generation

These scripts generate and manage **synthetic TCP 3-way handshakes injections** as a mitigation strategy.

---

## 1) Inject into a single trace

**Script:** `inject_frames.py`  
**Purpose:** add groups of 3 synthetic frames (SYN, SYN-ACK, ACK) into a trace CSV, sampled from previous interval statistics.   

**Example:**
```bash
python inject_frames.py traces_1.csv -o traces_1_aug.csv -t 5 -n 2
```

- `-t`: interval length in minutes (default 5)  
- `-n`: number of handshake groups per interval (default 1)  
- Output: a new augmented CSV file  

---

## 2) Batch inject all user traces

**Script:** `batch_inject.py`  
**Purpose:** apply injection to all `traces_*.csv` in `Users/<User>/Preprocessed/` and write results into a new injected user folder.   

**Example:**
```bash
python batch_inject.py Alice --users-dir ./Users --injections 5 --interval 3
```

- Input: `Users/Alice/Preprocessed/traces_*.csv`  
- Output: `Users/Alice_Injected_5/Preprocessed/traces_*.csv`  

---

## 3) Generate mixed mitigation dataset

**Script:** `generate_multi_mitigation.py`  
**Purpose:** build a *mixed injected* dataset by sampling each split from different injected folders (e.g., `Injected_1`, `Injected_600`, `Injected_1000`); logs decisions.   

**Example:**
```bash
python generate_multi_mitigation.py 42
```

- Creates: `Users/Alice_Injected_Mix/Splits/5.0s/split_*.csv`  
- Logs mapping (which injected N was used for each split) into `gener
