# Bank Fraud Detection System

A hybrid machine learning system for detecting fraudulent bank transactions using Rule-Based Detection, Random Forest, and Autoencoder.

## Overview

This project implements a three-layer fraud detection approach:

1. **Rule-Based Engine** - 7 interpretable rules for known fraud patterns
2. **Random Forest** - Supervised classifier for high accuracy
3. **Autoencoder** - Unsupervised anomaly detection for novel fraud

The final prediction uses a weighted ensemble: `score = α × RF_prob + (1-α) × AE_anomaly_score`

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
Bank_fraud_detection/
├── Final_Code_05Jan2026.ipynb          # Main implementation (synthetic data)
├── Running_Final_Code_05Jan2026.ipynb  # Inference on real CSV data
├── Generating entire dataset from Sample.ipynb  # CTGAN data generation
├── Entire Dataset from Net.ipynb       # Pipeline with external data
├── PROJECT_ANALYSIS.md                 # Detailed technical documentation
├── CLAUDE.md                           # Development guide
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Usage

### Training (with synthetic data)

Run `Final_Code_05Jan2026.ipynb` - generates synthetic data via CTGAN, trains models, and saves:
- `rf_model.pkl` - Random Forest model
- `ae_model.keras` - Autoencoder model
- `scaler.pkl` - Feature scaler
- `thresholds.pkl` - Optimal α and threshold

### Training (with real data)

1. Place your transaction data as `dataset.csv` (CSV format)
2. Run `Running_Final_Code_05Jan2026.ipynb`

### Inference

```python
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load models
rf = joblib.load('rf_model.pkl')
ae = load_model('ae_model.keras')
scaler = joblib.load('scaler.pkl')
params = joblib.load('thresholds.pkl')

# Preprocess and predict
X_scaled = scaler.transform(X_new)
p_rf = rf.predict_proba(X_scaled)[:, 1]
recon = ae.predict(X_scaled)
p_ae = (np.mean((recon - X_scaled)**2, axis=1))
p_ae = (p_ae - p_ae.min()) / (p_ae.max() - p_ae.min())

# Ensemble prediction
score = params['best_alpha'] * p_rf + (1 - params['best_alpha']) * p_ae
predictions = (score >= params['best_thresh']).astype(int)
```

## Data Schema

| Feature | Description |
|---------|-------------|
| `step` | Time step (hours over 30 days) |
| `type` | PAYMENT, TRANSFER, CASH_OUT, DEBIT |
| `amount` | Transaction amount |
| `nameOrig` | Origin account ID |
| `oldbalanceOrg` | Origin balance before |
| `newbalanceOrig` | Origin balance after |
| `nameDest` | Destination account ID |
| `oldbalanceDest` | Destination balance before |
| `newbalanceDest` | Destination balance after |
| `isFraud` | Ground truth (0 or 1) |

## The 7 Fraud Detection Rules

| Rule | Condition |
|------|-----------|
| Amount Threshold | amount > $50,000 |
| Velocity | > 5 transactions in last hour |
| Time of Day | Transaction between 12AM-5AM |
| Type Mismatch | CASH_OUT < $10 or TRANSFER > $80,000 |
| New Beneficiary | First transaction to recipient |
| Geolocation | Distance > 500km |
| Behavior Anomaly | Amount > 3σ from account average |

## Architecture

```
Transaction Data
       │
       ▼
┌──────────────────────┐
│  Rule-Based Engine   │
│  (7 fraud rules)     │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Preprocessing       │
│  • Label Encoding    │
│  • MinMax Scaling    │
│  • SMOTE Balancing   │
└──────────────────────┘
       │
       ├────────────────────┐
       ▼                    ▼
┌─────────────┐      ┌─────────────┐
│Random Forest│      │ Autoencoder │
│  (p_rf)     │      │   (p_ae)    │
└─────────────┘      └─────────────┘
       │                    │
       └─────────┬──────────┘
                 ▼
┌──────────────────────────────┐
│  Ensemble: α·p_rf + (1-α)·p_ae │
│  Threshold → Fraud/Not Fraud   │
└──────────────────────────────┘
```

## Results

Evaluation metrics (on test set):
- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion Matrix
- ROC and Precision-Recall Curves
- SHAP feature importance

## Documentation

See `PROJECT_ANALYSIS.md` for detailed technical documentation including:
- Complete pipeline explanation
- Model architecture details
- Ensemble scoring methodology
- Code walkthrough
