# Bank Fraud Detection System - Complete Analysis

> A hybrid machine learning approach combining Rule-Based Detection, Random Forest, and Autoencoder for identifying fraudulent bank transactions.

---

## Table of Contents

1. [What Does This Project Do?](#what-does-this-project-do)
2. [Project Files Overview](#project-files-overview)
3. [How The System Works](#how-the-system-works)
4. [The 7 Fraud Detection Rules](#the-7-fraud-detection-rules)
5. [Machine Learning Models](#machine-learning-models)
6. [Data Pipeline](#data-pipeline)
7. [The Ensemble Approach](#the-ensemble-approach)
8. [Output Files](#output-files)
9. [Key Observations](#key-observations)

---

## What Does This Project Do?

This project detects fraudulent bank transactions using a **three-layer approach**:

```
Layer 1: Rule-Based Detection    → Catches obvious fraud patterns
Layer 2: Random Forest           → Learns from labeled examples
Layer 3: Autoencoder             → Detects anomalies/unusual behavior
```

**Why three layers?**

| Approach | Strength | Weakness |
|----------|----------|----------|
| Rules | Fast, interpretable, catches known patterns | Can't adapt to new fraud types |
| Random Forest | High accuracy on known fraud types | Needs labeled data |
| Autoencoder | Catches unknown/novel fraud patterns | Higher false positive rate |

By combining all three, the system catches both **known fraud patterns** and **new/unusual fraud attempts**.

---

## Project Files Overview

### Notebooks

| File | Purpose | Data Source |
|------|---------|-------------|
| `Final_Code_05Jan2026.ipynb` | **Main implementation** - Full pipeline with synthetic data generation | Generates data using CTGAN |
| `Running_Final_Code_05Jan2026.ipynb` | Run pipeline on real data | Loads `dataset.csv` |
| `Entire Dataset from Net.ipynb` | Same as above | Loads `dataset.csv` |
| `Generating entire dataset from Sample.ipynb` | Expand small sample to large dataset | Loads `sample_with_3_percent_fraud.csv`, generates 100K via CTGAN |

### Data Files

| File | Description |
|------|-------------|
| `dataset.csv` | Real transaction data (CSV format) |
| `sample_with_3_percent_fraud.csv` | Small sample with 3% fraud rate |

### Model Files (Generated)

| File | Contents |
|------|----------|
| `rf_model.pkl` | Trained Random Forest model |
| `ae_model.keras` | Trained Autoencoder (Keras format) |
| `scaler.pkl` | Feature scaler for preprocessing |
| `thresholds.pkl` | Optimal α and threshold values |

---

## How The System Works

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   DATA INPUT                                                        │
│   (Real CSV or CTGAN-generated synthetic data)                      │
│                          │                                          │
│                          ▼                                          │
│   ┌─────────────────────────────────────────┐                       │
│   │         RULE-BASED ENGINE               │                       │
│   │         (7 Fraud Detection Rules)       │                       │
│   │         Output: isFlaggedFraud          │                       │
│   └─────────────────────────────────────────┘                       │
│                          │                                          │
│                          ▼                                          │
│   ┌─────────────────────────────────────────┐                       │
│   │         DATA PREPROCESSING              │                       │
│   │         • Label Encoding                │                       │
│   │         • MinMax Scaling (0-1)          │                       │
│   │         • SMOTE (Balance Classes)       │                       │
│   └─────────────────────────────────────────┘                       │
│                          │                                          │
│              ┌───────────┴───────────┐                              │
│              ▼                       ▼                              │
│   ┌──────────────────┐    ┌──────────────────┐                      │
│   │  RANDOM FOREST   │    │   AUTOENCODER    │                      │
│   │  (Supervised)    │    │  (Unsupervised)  │                      │
│   │                  │    │                  │                      │
│   │  Output: p_rf    │    │  Output: p_ae    │                      │
│   │  (fraud prob)    │    │  (anomaly score) │                      │
│   └────────┬─────────┘    └────────┬─────────┘                      │
│            │                       │                                │
│            └───────────┬───────────┘                                │
│                        ▼                                            │
│   ┌─────────────────────────────────────────┐                       │
│   │         ENSEMBLE SCORING                │                       │
│   │                                         │                       │
│   │   score = α × p_rf + (1-α) × p_ae       │                       │
│   │                                         │                       │
│   │   if score ≥ threshold → FRAUD          │                       │
│   │   else → NOT FRAUD                      │                       │
│   └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The 7 Fraud Detection Rules

The rule engine checks each transaction against 7 conditions. If **any rule triggers**, the transaction is flagged.

### Rule Details

| # | Rule Name | Condition | Why It Indicates Fraud |
|---|-----------|-----------|------------------------|
| 1 | **Amount Threshold** | `amount > $50,000` | Large transactions are higher risk |
| 2 | **Velocity** | `> 5 transactions in last hour` | Rapid transactions suggest automated fraud |
| 3 | **Time of Day** | `transaction between 12AM - 5AM` | Unusual hours for legitimate banking |
| 4 | **Type Mismatch** | `CASH_OUT < $10` OR `TRANSFER > $80,000` | Unusual amount for transaction type |
| 5 | **New Beneficiary** | `first transaction to this recipient` | New recipients are higher risk |
| 6 | **Geolocation** | `distance > 500km between accounts` | Geographic impossibility/anomaly |
| 7 | **Behavior Anomaly** | `amount > 3 standard deviations from user's average` | Unusual behavior for this account |

### How Rules Are Applied

```python
def apply_rules(df):
    # For each transaction:
    # 1. Check all 7 rules
    # 2. Set rule1=1, rule2=1, etc. if triggered
    # 3. Set isFlaggedFraud = 1 if ANY rule triggered

    df['isFlaggedFraud'] = df[['rule1','rule2',...,'rule7']].max(axis=1)
```

---

## Machine Learning Models

### Model 1: Random Forest (Supervised Learning)

**What it is**: An ensemble of 100 decision trees that vote on whether a transaction is fraud.

**How it works**:
```
Training Data (with fraud labels)
         │
         ▼
    ┌─────────┐
    │ Tree 1  │──► Vote: Fraud
    │ Tree 2  │──► Vote: Not Fraud
    │ Tree 3  │──► Vote: Fraud
    │  ...    │
    │ Tree 100│──► Vote: Fraud
    └─────────┘
         │
         ▼
    Majority Vote → Final Prediction
```

**Output**: `p_rf` = Probability of fraud (0.0 to 1.0)

**Strengths**:
- High accuracy on patterns seen in training
- Handles mixed feature types well
- Provides feature importance

---

### Model 2: Autoencoder (Unsupervised Anomaly Detection)

**What it is**: A neural network trained to compress and reconstruct **normal** transactions.

**Architecture**:
```
Input Layer (9 features)
      │
      ▼
Dense Layer (4 neurons, ReLU)     ─┐
      │                            │ ENCODER
      ▼                            │ (Compress)
Dense Layer (2 neurons, ReLU)     ─┘  ← Bottleneck
      │
      ▼                            ─┐
Dense Layer (4 neurons, ReLU)       │ DECODER
      │                             │ (Reconstruct)
      ▼                            ─┘
Output Layer (9 features, Sigmoid)
```

**How it detects fraud**:

```
Normal Transaction ──► Autoencoder ──► Good Reconstruction ──► Low Error ✓
Fraud Transaction  ──► Autoencoder ──► Poor Reconstruction ──► High Error ⚠
```

The autoencoder learns what "normal" looks like. When it sees fraud (which it wasn't trained on), it can't reconstruct it well → **high reconstruction error = anomaly**.

**Output**: `p_ae` = Normalized reconstruction error (0.0 to 1.0)

---

## Data Pipeline

### Step 1: Data Generation/Loading

**Option A - Synthetic Data (Final_Code notebook)**:
```python
# Create small seed dataset
seed_df = simulate_seed(n=2000, fraud_rate=0.03)

# Train CTGAN (Generative Adversarial Network)
ctgan = CTGAN(epochs=50, batch_size=100)
ctgan.fit(seed_df, discrete_columns=['type','nameOrig','nameDest'])

# Generate 100,000 synthetic transactions
syn = ctgan.sample(100000)
```

**Option B - Real Data (Running_Final_Code notebook)**:
```python
seed_df = pd.read_csv('dataset.csv')
syn = seed_df.sample(n=100000, random_state=42)
```

### Step 2: Feature Engineering

**Transaction Features**:

| Feature | Description | Example |
|---------|-------------|---------|
| `step` | Time step (hours in 30-day period) | 156 |
| `type` | Transaction type | PAYMENT, TRANSFER, CASH_OUT, DEBIT |
| `amount` | Transaction amount | 5432.50 |
| `nameOrig` | Origin account ID | C1234567890 |
| `oldbalanceOrg` | Origin balance before | 10000.00 |
| `newbalanceOrig` | Origin balance after | 4567.50 |
| `nameDest` | Destination account ID | M9876543210 |
| `oldbalanceDest` | Destination balance before | 0.00 |
| `newbalanceDest` | Destination balance after | 5432.50 |
| `isFraud` | Ground truth label | 0 or 1 |

### Step 3: Preprocessing Pipeline

```
Raw Data
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. LABEL ENCODING                                           │
│    • type: PAYMENT→0, TRANSFER→1, CASH_OUT→2, DEBIT→3      │
│    • nameOrig/nameDest: Convert account IDs to integers     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAIN/TEST SPLIT                                         │
│    • 80% Training, 20% Testing                              │
│    • Stratified (maintains fraud ratio in both sets)        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. MIN-MAX SCALING                                          │
│    • Scales all features to [0, 1] range                    │
│    • Required for neural network (Autoencoder)              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. SMOTE (Synthetic Minority Oversampling)                  │
│    • Original: ~3% fraud, ~97% non-fraud                    │
│    • After SMOTE: 50% fraud, 50% non-fraud                  │
│    • Creates synthetic fraud samples for balanced training  │
└─────────────────────────────────────────────────────────────┘
```

---

## The Ensemble Approach

### Why Combine Models?

| Model | Good At | Bad At |
|-------|---------|--------|
| Random Forest | Known fraud patterns | Novel/unseen fraud types |
| Autoencoder | Novel anomalies | May flag legitimate unusual transactions |

**Solution**: Combine both with a weighted score!

### The Scoring Formula

```
final_score = α × p_rf + (1 - α) × p_ae
```

Where:
- `p_rf` = Random Forest fraud probability (0 to 1)
- `p_ae` = Autoencoder anomaly score (0 to 1)
- `α` = Weight parameter (0 to 1)

**Interpretation**:
- `α = 1.0` → Trust Random Forest 100%
- `α = 0.0` → Trust Autoencoder 100%
- `α = 0.7` → 70% RF + 30% AE (example optimal value)

### Finding Optimal α and Threshold

```python
# Grid search over α values
best = {'f1': 0}
for alpha in [0.0, 0.1, 0.2, ..., 1.0]:
    score = alpha * p_rf + (1-alpha) * p_ae

    # Find threshold that maximizes F1-score
    precision, recall, thresholds = precision_recall_curve(y_test, score)
    f1_scores = 2 * precision * recall / (precision + recall)
    best_idx = argmax(f1_scores)

    if f1_scores[best_idx] > best['f1']:
        best = {'alpha': alpha, 'threshold': thresholds[best_idx], 'f1': f1_scores[best_idx]}
```

### Final Prediction

```python
final_score = best_alpha * p_rf + (1 - best_alpha) * p_ae
prediction = 1 if final_score >= best_threshold else 0
```

---

## Output Files

After training, the following files are saved:

### `rf_model.pkl`
```python
# Random Forest classifier
# Load with:
import joblib
rf = joblib.load('rf_model.pkl')
predictions = rf.predict(X_new)
```

### `ae_model.keras`
```python
# Keras Autoencoder
# Load with:
from tensorflow.keras.models import load_model
ae = load_model('ae_model.keras')
reconstructions = ae.predict(X_new)
```

### `scaler.pkl`
```python
# MinMaxScaler for preprocessing
# Load with:
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X_new)
```

### `thresholds.pkl`
```python
# Optimal ensemble parameters
# Load with:
thresholds = joblib.load('thresholds.pkl')
# Returns: {'best_alpha': 0.7, 'best_thresh': 0.45}
```

---

## Evaluation Metrics

The system is evaluated using:

| Metric | What It Measures | Formula |
|--------|------------------|---------|
| **Accuracy** | Overall correctness | (TP + TN) / Total |
| **Precision** | Of predicted frauds, how many are real? | TP / (TP + FP) |
| **Recall** | Of actual frauds, how many did we catch? | TP / (TP + FN) |
| **F1-Score** | Balance of Precision & Recall | 2 × (P × R) / (P + R) |
| **AUC-ROC** | Model's ability to distinguish classes | Area under ROC curve |

### Visualizations Generated

1. **Confusion Matrix** - Shows TP, TN, FP, FN counts
2. **ROC Curve** - True Positive Rate vs False Positive Rate
3. **Precision-Recall Curve** - Precision vs Recall at different thresholds
4. **SHAP Summary Plot** - Feature importance for Random Forest

---

## Key Observations

### Strengths

1. **Hybrid approach** - Combines rule-based, supervised, and unsupervised methods
2. **Handles imbalanced data** - Uses SMOTE to balance training
3. **Explainable** - SHAP values show which features matter
4. **Flexible** - α parameter allows tuning RF vs AE importance

### Limitations

1. **Rule engine output not used in ML** - `isFlaggedFraud` is computed but the models train on `isFraud` only
2. **Geolocation is simulated** - Random lat/lon, not real geographic data
3. **Notebooks are duplicated** - Same code appears in multiple files
4. **No real-time capability** - Batch processing only

### Potential Improvements

- Use `isFlaggedFraud` as an additional feature for ML models
- Implement real geolocation data
- Create a REST API for real-time predictions
- Add model versioning and experiment tracking
- Consolidate duplicate code into reusable modules

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn tensorflow keras ctgan imbalanced-learn shap geopy matplotlib seaborn joblib
```

### Running the Pipeline

1. **With synthetic data**: Run `Final_Code_05Jan2026.ipynb`
2. **With real data**: Place `dataset.csv` in folder, run `Running_Final_Code_05Jan2026.ipynb`

### Using Saved Models

```python
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load models
rf = joblib.load('rf_model.pkl')
ae = load_model('ae_model.h5')
scaler = joblib.load('scaler.pkl')
params = joblib.load('thresholds.pkl')

# Preprocess new data
X_new_scaled = scaler.transform(X_new)

# Get predictions
p_rf = rf.predict_proba(X_new_scaled)[:, 1]
recon = ae.predict(X_new_scaled)
err = np.mean((recon - X_new_scaled)**2, axis=1)
p_ae = (err - err.min()) / (err.max() - err.min())

# Ensemble score
score = params['best_alpha'] * p_rf + (1 - params['best_alpha']) * p_ae
predictions = (score >= params['best_thresh']).astype(int)
```

---

## Summary

This Bank Fraud Detection system uses a **three-pronged approach**:

1. **Rule Engine** - 7 interpretable rules for known fraud patterns
2. **Random Forest** - Supervised learning for high accuracy
3. **Autoencoder** - Unsupervised anomaly detection for novel fraud

The final prediction combines Random Forest probability and Autoencoder anomaly score using an optimized weighted ensemble, achieving robust fraud detection across both known and unknown fraud types.
