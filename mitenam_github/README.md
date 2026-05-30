# MITeNAM: Multi-Teacher Input-Injection Neural Additive Model

Anonymous code repository for CIKM 2026 short paper submission.

This repository contains the official implementation of MITeNAM, a hybrid framework that mitigates the performance-interpretability trade-off in tabular medical data classification by combining a weighted multi-teacher gradient-boosting ensemble with a Neural Additive Model (NAM) ensemble student via input-injection and a joint BCE + knowledge distillation loss.

## Overview

MITeNAM consists of three components:

1. **Multi-Teacher Ensemble**: XGBoost, CatBoost, and LightGBM combined with fixed weights (0.25, 0.50, 0.25)
2. **Input-Injection**: Teacher ensemble probability is appended to the student's input features
3. **NAM Student Ensemble**: 3 NAMs with different random seeds (42, 1234, 5678), averaged at inference

The student is trained with a joint loss:

```
L = α · BCE(z_s, y) + (1 - α) · T² · KD(σ(z_s/T), σ(z_t/T)) + λ · R(f)
```

where α = 0.5, T = 2.0, λ = 0.01 (standard Hinton KD setup with L2 regularization on NAM per-feature outputs).

## Dataset

**UCI Diabetes 130-US Hospitals** ([UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)):

- Raw size: 101,766 inpatient encounters
- Cohort filter: Exclude expired/hospice patients (`discharge_disposition_id ∈ {11, 13, 14, 19, 20, 21}`)
- Final cohort: **n = 99,343** (11.39% positive rate, 30-day readmission)

## Results

Averaged over 3 outer seeds × 5-fold stratified cross-validation (15 runs total):

| Model | AUROC | AUPRC | F1 | DeLong p (vs MITeNAM) |
|---|---|---|---|---|
| **MITeNAM (Ours)** | **0.6709** | **0.2236** | **0.2838** | — |
| XGBoost | 0.6676 | 0.2215 | 0.2813 | 6.0 × 10⁻⁶ ↑ |
| CatBoost | 0.6687 | 0.2214 | 0.2828 | 1.1 × 10⁻³ ↑ |
| LightGBM | 0.6703 | 0.2256 | 0.2833 | 0.300 (eq.) |
| EBM | 0.6666 | 0.2224 | 0.2794 | 2.6 × 10⁻⁵ ↑ |

↑ indicates MITeNAM statistically outperforms the baseline at p < 0.05.

MITeNAM outperforms 3 of 4 baselines with statistical significance and is statistically equivalent to the strongest GB baseline (LightGBM) while additionally providing intrinsic interpretability through additive NAM shape functions.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── cohort.py             # UCI cohort generation
│   ├── nam.py                # NAM model definition
│   ├── train_teachers.py     # Train XGBoost, CatBoost, LightGBM
│   ├── train_mitenam.py      # Train MITeNAM (main model)
│   ├── evaluate.py           # Metrics + DeLong test
│   └── utils.py              # Common utilities
└── scripts/
    └── run_full_pipeline.py  # End-to-end reproducibility script
```

## Installation

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

## Usage

### Quick start (reproduce all results)

```bash
# 1. Download UCI Diabetes 130-US Hospitals dataset
#    Place diabetic_data.csv in ./data/

# 2. Run full pipeline
python scripts/run_full_pipeline.py --data_dir ./data --output_dir ./results
```

### Step-by-step

```bash
# 1. Generate cohort (101,766 → 99,343)
python src/cohort.py --input ./data/diabetic_data.csv --output ./data/uci_cohort.csv

# 2. Train teacher models (XGBoost, CatBoost, LightGBM)
python src/train_teachers.py --cohort ./data/uci_cohort.csv --output ./results/teachers/

# 3. Train MITeNAM
python src/train_mitenam.py --cohort ./data/uci_cohort.csv \
    --teachers ./results/teachers/ \
    --output ./results/mitenam/ \
    --weights cat_heavy   # 0.25 / 0.50 / 0.25

# 4. Evaluate (AUROC, AUPRC, F1, DeLong test)
python src/evaluate.py --results ./results/ --output ./results/tables/
```

### Teacher weight ablation

```bash
# Reproduce ablation: equal / cat_heavy / top2_heavy
for cfg in equal cat_heavy top2_heavy; do
    python src/train_mitenam.py --weights $cfg --output ./results/mitenam_${cfg}/
done
```

## Hyperparameters

All hyperparameters are fixed as listed in the paper:

**Teacher models** (XGBoost, CatBoost, LightGBM):
- `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`
- `scale_pos_weight` set to handle class imbalance

**NAM student** (per-feature subnetwork):
- Architecture: 1 → 64 → 64 → 32 → 1 (3-layer MLP)
- Activation: ReLU; Dropout: 0.1
- Random feature dropout: 0.05 (training only)
- Ensemble: 3 NAMs with seeds {42, 1234, 5678}, sigmoid outputs averaged

**Optimizer & loss**:
- Adam, learning rate = 1e-3, weight_decay = 1e-5
- α = 0.5 (BCE weight), T = 2.0 (KD temperature), λ = 0.01 (L2 reg)
- Validation-AUROC-based early stopping (patience = 12)

**Cross-validation**:
- 3 outer seeds: {42, 0, 2026}
- 5-fold stratified within each seed
- DeLong test for AUROC comparison

## Citation

This is an anonymized repository for double-blind review at CIKM 2026.
Citation information will be added upon paper acceptance.

## License

MIT License (see `LICENSE` file).
