# Edge AI Real-Time Anomaly Detection

Lightweight neural network for **binary IoT botnet anomaly detection** on resource-constrained edge devices. The project trains compact models on the [N-BaIoT](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) dataset, converts the best compact model to **INT8 TensorFlow Lite**, and serves a Streamlit dashboard for live inference and model comparison.

**Deployed detector:** 4,385-parameter MLP quantized to **8.69 KB** INT8 TFLite, with **~99.99%** test accuracy on the held-out N-BaIoT split used in this repo.

---

## Contents

- [Problem](#problem)
- [What this project does](#what-this-project-does)
- [Dataset](#dataset)
- [Models](#models)
- [Pipeline](#pipeline)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Quick start (dashboard)](#quick-start-dashboard)
- [Full experiment pipeline](#full-experiment-pipeline)
- [Dashboard](#dashboard)
- [Experimental results](#experimental-results)
- [Scripts reference](#scripts-reference)
- [Notes and limitations](#notes-and-limitations)

---

## Problem

IoT devices (cameras, doorbells, routers) are frequent botnet targets (Mirai, BASHLITE, and related families). Cloud-only detection adds latency, bandwidth cost, and privacy risk. This project asks whether a **small, quantized neural network** can classify **normal vs anomalous** traffic locally, with a footprint small enough for edge hardware.

---

## What this project does

1. Loads N-BaIoT CSVs (Danmini Doorbell subset), labels benign traffic as **NORMAL (0)** and attack traffic as **ANOMALY (1)**.
2. Builds a **leakage-safe** train/test split, then fits a median imputer and standard scaler **on training data only**.
3. Trains three Keras models on the same 115-feature inputs:
   - Baseline MLP (larger)
   - Lightweight MLP (deployed architecture)
   - KronNet-inspired compact network (research comparison)
4. Converts the lightweight MLP to **FP32 TFLite** and **full-integer INT8 TFLite**.
5. Evaluates accuracy, precision, recall, F1, ROC-AUC, PR-AUC, size, parameters, and latency on the **same test set**.
6. Runs real-time / streaming demos and a Streamlit UI that can score one sample on **all four deployed variants**.

---

## Dataset

| Item | Value |
|------|--------|
| Source | N-BaIoT (UCI) |
| Device used here | `data/raw/nbaiot/Danmini_Doorbell` |
| Labeling | Files named `benign_traffic` → 0 (NORMAL); all other CSVs → 1 (ANOMALY) |
| Rows per file | Cap of **20,000** (`MAX_ROWS_PER_FILE` in `src/prepare_data.py`) |
| Features | **115** numeric statistical traffic features after cleaning |
| Split | 80% train / 20% test, stratified, `random_state=42` |
| Preprocessing | Inf → NaN; drop all-NaN columns; **median impute** then **StandardScaler**, both fit on train only |
| Saved arrays | `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` |
| Test set size (current artifacts) | **44,000** samples (4,000 normal / 40,000 anomaly) |

Place raw CSVs under:

```text
data/raw/nbaiot/Danmini_Doorbell/**/*.csv
```

Then run `python src/inspect_dataset.py` to verify files are found, followed by `python src/prepare_data.py`.

Saved preprocessing objects:

- `models/imputer.pkl`
- `models/scaler.pkl`

---

## Models

### 1. Lightweight MLP (primary / edge model)

Architecture used in training (`src/train.py`) and shown in the dashboard:

```text
115 features
  → Dense(32, ReLU) + L2(1e-4)
  → Dense(16, ReLU) + L2(1e-4)
  → Dense(8, ReLU)  + L2(1e-4)
  → Dense(1, sigmoid)     # P(anomaly) ≥ 0.5 → ANOMALY
```

| Property | Value |
|----------|--------|
| Parameters | **4,385** |
| Optimizer | Adam, lr = 0.001 |
| Loss | Binary cross-entropy |
| Batch / epochs | 256 / up to 50 |
| Regularization | L2, class weights, EarlyStopping, ReduceLROnPlateau |
| Keras file | `models/lightweight_model.keras` (~81.5 KB on disk in this checkout) |
| FP32 TFLite | `models/lightweight_model_fp32.tflite` |
| INT8 TFLite | `models/lightweight_model_int8.tflite` (**8.69 KB**) |

INT8 conversion (`src/convert_to_tflite.py`) uses a 1,000-sample representative set from training data, `TFLITE_BUILTINS_INT8`, and integer input/output types.

### 2. Baseline MLP

Larger capacity comparison (`src/train_baseline.py`):

```text
115 → 128 → 64 → 32 → 16 → 1 (sigmoid)
```

**25,729 parameters**, `models/baseline_model.keras`.

### 3. KronNet-inspired (comparison, not a paper replica)

`src/train_kronnet.py` builds a compact encoder plus a **2×2 learned transform** inspired by Kronecker-style blocks. It is **adapted** for this 115-feature binary task; it is not an exact reproduction of the original KronNet paper (multiclass architecture / training).

**9,737 parameters**, `models/kronnet_model.keras`.

### 4. Lightweight MLP (INT8 TFLite)

Same 4,385-parameter network as (1), quantized for edge inference. The dashboard and `src/realtime_detector.py` / `src/evaluate_tflite.py` dequantize INT8 outputs back to a probability in `[0, 1]`.

---

## Pipeline

```text
N-BaIoT CSVs (Danmini Doorbell)
        ↓
Inspect + load (cap 20k rows/file)
        ↓
Clean inf/NaN, keep 115 numeric features
        ↓
Stratified 80/20 split
        ↓
Imputer + scaler (fit on train only)
        ↓
Train Baseline / Lightweight / KronNet-inspired
        ↓
Evaluate Keras models on the same test set
        ↓
Convert Lightweight MLP → FP32 + INT8 TFLite
        ↓
Evaluate INT8 + compare_models.py
        ↓
Figures, CSV, Streamlit dashboard
        ↓
NORMAL / ANOMALY
```

---

## Project structure

```text
edge_anomaly_detection/
├── app.py                      # Streamlit dashboard (main UI)
├── app_old.py                  # Previous dashboard revision
├── README.md
├── requirements.txt
│
├── src/
│   ├── inspect_dataset.py      # List/preview raw N-BaIoT CSVs
│   ├── prepare_data.py         # Leakage-safe preprocess + save arrays
│   ├── train.py                # Lightweight MLP
│   ├── evaluate.py             # Lightweight Keras metrics + confusion matrix
│   ├── train_baseline.py       # Baseline MLP
│   ├── evaluate_baseline.py
│   ├── train_kronnet.py        # KronNet-inspired model
│   ├── evaluate_kronnet.py
│   ├── convert_to_tflite.py    # FP32 + INT8 conversion (use this)
│   ├── evaluate_tflite.py      # INT8 test-set evaluation
│   ├── compare_models.py       # Four-way comparison CSV + plots
│   ├── generate_results.py     # Static comparison figures
│   ├── generate_training_curves.py
│   ├── final_results.py        # Writes results/final_results.txt
│   ├── realtime_detector.py    # CLI INT8 detector (imputer + scaler + TFLite)
│   ├── realtime_simulator.py   # Stream INT8 over the test set
│   ├── convert_tflite.py       # Alternate/legacy conversion helper
│   ├── benchmark.py            # Placeholder
│   └── config.py               # Placeholder
│
├── data/
│   ├── raw/nbaiot/             # Place N-BaIoT CSVs here (not always committed)
│   └── processed/              # X_*.npy, y_*.npy
│
├── models/
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── lightweight_model.keras
│   ├── lightweight_model_fp32.tflite
│   ├── lightweight_model_int8.tflite
│   ├── baseline_model.keras
│   └── kronnet_model.keras
│
├── results/
│   ├── lightweight_metrics.txt
│   ├── baseline_metrics.txt
│   ├── int8_metrics.txt
│   ├── kronnet_metrics.txt
│   ├── model_comparison.csv    # Source of truth for dashboard tables
│   ├── final_results.txt
│   ├── training_history.npz
│   ├── kronnet_training_history.npz
│   ├── *_confusion_matrix.png
│   ├── model_comparison_*.png
│   └── figures/                # Training curves and bar charts
│
└── venv/                       # Local Python environment (optional)
```

---

## Requirements

Python **3.10–3.12** recommended. Packages used by training, evaluation, and the app:

```text
numpy
pandas
scikit-learn
tensorflow
joblib
matplotlib
streamlit
psutil          # optional; CPU/RAM bars in the sidebar
```

Create and use a virtual environment (Windows PowerShell):

```powershell
cd "c:\Users\karthikeya mallela\OneDrive\Desktop\edge_anomaly_detection"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn tensorflow joblib matplotlib streamlit psutil
```

If script activation is blocked:

```powershell
.\venv\Scripts\python.exe -m pip install numpy pandas scikit-learn tensorflow joblib matplotlib streamlit psutil
```

---

## Quick start (dashboard)

Trained models and processed test arrays are already in this checkout. From the **project root**:

```powershell
cd "c:\Users\karthikeya mallela\OneDrive\Desktop\edge_anomaly_detection"
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Without activating the venv:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

The app loads:

- `models/lightweight_model_int8.tflite`
- `models/scaler.pkl`, `models/imputer.pkl`
- `data/processed/X_test.npy`, `y_test.npy`
- Keras models if present (`lightweight_model.keras`, `baseline_model.keras`, `kronnet_model.keras`)
- `results/model_comparison.csv` and figure PNGs

---

## Full experiment pipeline

Run **from the project root** so relative paths (`data/`, `models/`, `results/`) resolve.

```powershell
# 1. Confirm raw CSVs
python src/inspect_dataset.py

# 2. Build processed arrays (overwrites data/processed and imputer/scaler)
python src/prepare_data.py

# 3. Train
python src/train.py
python src/train_baseline.py
python src/train_kronnet.py

# 4. Evaluate Keras models
python src/evaluate.py
python src/evaluate_baseline.py
python src/evaluate_kronnet.py

# 5. Quantize lightweight MLP
python src/convert_to_tflite.py

# 6. Evaluate INT8
python src/evaluate_tflite.py

# 7. Head-to-head table + plots (dashboard reads this CSV)
python src/compare_models.py

# 8. Extra figures and summary report
python src/generate_training_curves.py
python src/generate_results.py
python src/final_results.py

# 9. CLI live demo (optional)
python src/realtime_simulator.py
python src/realtime_detector.py
```

---

## Dashboard

`app.py` — **Edge AI Real-Time Anomaly Detector**.

| Area | Behavior |
|------|----------|
| Sidebar | Pick **All Models**, INT8 TFLite, Lightweight MLP, Baseline MLP, or KronNet-inspired (only listed if the `.keras` file exists). Shows INT8 size, 4,385 parameters, feature count, optional CPU/RAM. |
| Sample | **Generate Test Sample** draws a random held-out N-BaIoT test row and shows the true label. |
| Predict | Single-model prediction (label, confidence, latency) or a table of **all four models on the same input**. |
| Live monitoring | Streams 10–100 random test samples with running accuracy. |
| History | Last 100 detections; CSV download; anomaly-probability line chart. |
| Research | Loads `results/model_comparison.csv`, training curves, and confusion matrix. |

Decision rule: sigmoid probability **≥ 0.5** → ANOMALY, else NORMAL.

---

## Experimental results

Figures and text files under `results/` come from this repository’s training/eval scripts. Metrics below mix the **saved metric files** and **`results/model_comparison.csv`** (re-run `compare_models.py` after retraining; latency depends on the host CPU).

### Saved evaluation files

| Model | Accuracy | Precision | Recall | F1 | Parameters | Size |
|-------|----------|-----------|--------|-----|------------|------|
| Lightweight MLP (`lightweight_metrics.txt`) | 0.999909 | 0.999950 | 0.999950 | 0.999950 | 4,385 | 81.53 KB (Keras file) |
| Baseline MLP (`baseline_metrics.txt`) | 0.999886 | 0.999925 | 0.999950 | 0.999938 | 25,729 | 336.91 KB |
| INT8 TFLite (`int8_metrics.txt`) | 0.999886 | 0.999950 | 0.999925 | 0.999937 | 4,385 | **8.69 KB** |
| KronNet-inspired (`kronnet_metrics.txt`) | 0.999636 | 0.999900 | 0.999700 | 0.999800 | 9,737 | 165.95 KB |

KronNet test confusion matrix (44,000 samples): TN 3996, FP 4, FN 12, TP 39988.

### Comparison CSV (same test set)

`results/model_comparison.csv` (rounded):

| Model | Accuracy | F1 | Parameters | File size (KB) |
|-------|----------|-----|------------|----------------|
| Lightweight MLP | 0.999909 | 0.999950 | 4,385 | 81.53 |
| Lightweight MLP (INT8 TFLite) | 0.999909 | 0.999950 | 4,385 | **8.69** |
| Baseline MLP | 0.999886 | 0.999938 | 25,729 | 336.91 |
| KronNet-inspired | 0.999636 | 0.999800 | 9,737 | 165.95 |

### Edge compression (from `results/final_results.txt`)

| Optimization | Value |
|--------------|--------|
| Parameter reduction vs baseline | **~82.96%** (25,729 → 4,385) |
| Baseline Keras → INT8 size | **~97.4%** smaller (336.91 KB → 8.69 KB) |
| Lightweight FP32 TFLite → INT8 | **~55.8%** smaller (19.65 KB → 8.69 KB) |

High N-BaIoT scores are expected for this device/subset and labeling scheme; they do not by themselves prove generalization to other IoT devices or unseen attack families.

---

## Scripts reference

| Script | Role |
|--------|------|
| `src/inspect_dataset.py` | Recursively lists CSVs under `data/raw/nbaiot/` and prints sample columns. |
| `src/prepare_data.py` | Leakage-safe dataset build. |
| `src/train.py` / `src/evaluate.py` | Lightweight MLP train / test. |
| `src/train_baseline.py` / `src/evaluate_baseline.py` | Baseline MLP. |
| `src/train_kronnet.py` / `src/evaluate_kronnet.py` | KronNet-inspired model. |
| `src/convert_to_tflite.py` | Writes FP32 and INT8 `.tflite` files. |
| `src/evaluate_tflite.py` | Full test-set INT8 loop with quantization. |
| `src/compare_models.py` | Unified metrics CSV + comparison PNGs. |
| `src/generate_training_curves.py` | Accuracy/loss plots from `training_history.npz`. |
| `src/generate_results.py` | Parameter / size / accuracy / F1 bar charts. |
| `src/final_results.py` | Human-readable summary `results/final_results.txt`. |
| `src/realtime_simulator.py` | Sequential INT8 inference over test samples. |
| `src/realtime_detector.py` | Applies imputer + scaler then INT8 predict (raw-feature path). |
| `app.py` | Interactive Streamlit UI. |

---

## Notes and limitations

- **Device subset:** Training uses Danmini Doorbell CSVs only, not the full multi-device N-BaIoT corpus.
- **Class imbalance:** Anomaly rows dominate after the per-file cap; class weights are used during training.
- **Test samples in the UI are already scaled.** Dashboard samples come from `X_test.npy`. `realtime_detector.py` is the path that expects raw features and applies imputer + scaler.
- **INT8 latency in CSV vs metric files** can differ: per-sample Python interpreter loops vs batched Keras `predict`. Use relative rankings, not a hardware datasheet number.
- **KronNet** here is a compact inspired architecture for a fair size/accuracy comparison, not a claim of reproducing the original paper.
- `requirements.txt` in the repo may be empty; install the packages listed under [Requirements](#requirements).
- `src/config.py` and `src/benchmark.py` are currently unused placeholders.

---

## Citation (dataset)

Meidan, Y., et al. *N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders.* IEEE Pervasive Computing, 2018. UCI Machine Learning Repository: Detection of IoT Botnet Attacks N-BaIoT.
