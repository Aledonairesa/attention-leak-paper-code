# The Attention Leak: Behavioral Inference from Temporal Metadata in Network Traffic

This repository contains the scripts accompanying the paper:

> **The Attention Leak: Behavioral Inference from Temporal Metadata in Network Traffic**  
> By Alejandro Donaire*, Carlos Borrego*, and Marc Juarez**
> 
> **Department of Information and Communications Engineering, Autonomous University of Barcelona (Barcelona, Spain)* \
> ***School of Informatics, University of Edinburgh (Edinburgh, Scotland)*

---

## Overview

This work demonstrates how **task-switching dynamics** can be inferred from temporal metadata alone, exposing a new privacy risk.

We provide:

- **Data generation tools** (via Docker and local collectors)  
- **Three inference pipelines**:  
  - *Macroscopic*: dataset-level statistics  
  - *Microscopic*: packet-level segmentation  
  - *Mesoscopic*: fixed-interval feature extraction (best-performing)  
- **Model training and evaluation scripts**  
- **Mitigation experiments**: injecting synthetic TCP handshakes to obscure task-switch signals  

---

## Repository Structure

```
├── 1. Macroscopic approach        # Coarse dataset-level inference
├── 2. Microscopic approach        # Packet-level classification
├── 3. Mesoscopic approach         # Interval-based models
├── Data collector app             # Lightweight tool for collecting real traces
├── Data generation via Docker     # Scripts for controlled synthetic traces
├── requirements.txt               # Python dependencies
└── README.md  
```

Each subfolder contains its own **README.md** with detailed usage instructions and examples.

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Aledonairesa/attention-leak-paper-code.git
cd attention-leak-paper-code
```

Set up a Python environment (Python 3.10.16 recommended) and install requirements:

```bash
pip install -r requirements.txt
```
