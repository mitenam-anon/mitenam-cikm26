# MITeNAM: Multi-Teacher Input-Injection Neural Additive Model

Reproducible implementation of the experimental pipeline reported in our CIKM 2026 short paper, for the UCI Diabetes 130-US Hospitals 30-day readmission prediction task.

## Repository Structure

```
mitenam_cikm26/
├── README.md
├── data/
│   └── diabetic_data.csv           ← UCI dataset (place here before running)
├── mitenam_uci_pipeline.ipynb      ← main reproduction notebook
└── results/                        ← auto-generated after running
    ├── uci_xgb.pkl
    ├── uci_cat.pkl
    ├── uci_lgb.pkl
    ├── uci_ebm.pkl
    ├── uci_mitenam.pkl
    ├── figure2.pdf
    └── figure2.png
```

## Dataset Setup

The UCI Diabetes 130-US Hospitals dataset must be downloaded separately and placed in the `data/` folder:

1. Download `diabetic_data.csv` from the UCI ML Repository:
   https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Place the file at `mitenam_cikm26/data/diabetic_data.csv`.

## Quick Start

### Option 1: Google Colab (recommended)

1. Upload the entire `mitenam_cikm26/` folder to your Google Drive root.
2. Make sure `diabetic_data.csv` is inside `mitenam_cikm26/data/`.
3. Open `mitenam_uci_pipeline.ipynb` in Colab.
4. Set runtime to GPU: `Runtime → Change runtime type → T4 GPU` (or A100 for faster).
5. Run all cells (`Runtime → Run all`).

Expected runtime:
- T4 GPU (free Colab): ~1.5 hours
- A100 GPU (Colab Pro): ~40–50 minutes

### Option 2: Local environment

```bash
pip install numpy pandas scikit-learn scipy torch xgboost catboost lightgbm interpret matplotlib
```

In Cell 1 of the notebook, change the `BASE` path to your local project root:
```python
BASE = '/path/to/mitenam_cikm26/'
```
Then run with Jupyter.

## Reproducibility

All random seeds are fixed:
- **Outer CV seeds**: `[42, 0, 2026]` (3 outer seeds × 5-fold stratified = 15 fold-level evaluations)
- **NAM ensemble seeds**: `[42, 1234, 5678]` (3 NAM students averaged at inference)
- **Inner train/val split seed**: `42` (stratified 15% validation split for early stopping)

## Method Summary

**MITeNAM** combines three GBDT teachers (XGBoost, CatBoost, LightGBM) via a fixed cat-heavy weighted average — $(w_{xgb}, w_{cat}, w_{lgb}) = (0.25, 0.50, 0.25)$ — and injects the resulting teacher probability as an additional input feature into a NAM student. The student is trained with a joint binary cross-entropy and temperature-scaled distillation loss.

### Hyperparameters

| Component | Setting |
| --- | --- |
| GBDT teachers (XGBoost, CatBoost, LightGBM) | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=neg/pos`, `random_state=42` |
| EBM baseline | Default settings, `random_state=42` |
| NAM subnetwork | 3-layer MLP (1 → 64 → 64 → 32 → 1), ReLU activations, dropout 0.1 |
| Categorical embedding | 1-D learnable embedding before NAM subnetwork |
| Feature dropout | 0.05 (random per-feature output zeroing during training) |
| NAM optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Batch size | 2048 |
| Epochs / early stopping | 50 epochs max, patience=10 on validation AUROC |
| KD loss | $\alpha=0.5$, $T=2.0$, L2 ridge regularization coefficient 0.01 |
| Ensemble weights | cat_heavy: $(w_{xgb}, w_{cat}, w_{lgb}) = (0.25, 0.50, 0.25)$ |

### Evaluation Protocol

- **AUROC** and **AUPRC**: standard, threshold-independent.
- **F1**: computed at the threshold maximizing F1 on the out-of-fold predictions, matching the paper's protocol.
- **Statistical test**: real DeLong's test (Sun & Xu, 2014 fast implementation of DeLong 1988) for comparing two correlated AUROC values.

### Data Preprocessing

1. Target: `readmitted == '<30'` (binary 30-day readmission)
2. Replace `'?'` with `NaN`
3. Drop columns: `encounter_id`, `patient_nbr`, `weight`, `payer_code`, `medical_specialty`, `examide`, `citoglipton`, `readmitted`
4. Exclude encounters with `discharge_disposition_id ∈ {11, 13, 14, 19, 20, 21}` (expired or hospice)
5. Age band `[a-b)` → midpoint `a + 5`
6. ICD-9 diagnosis codes (`diag_1`, `diag_2`, `diag_3`) → 9 broad categories
   (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other)
7. Numerical features: median imputation, then `StandardScaler`
8. Categorical features: fill missing with `'Missing'`, then label encoding for NAM embedding (one-hot for GBDT baselines and EBM)

Final cohort: **99,343 encounters**, 30-day readmission positive rate **11.39%**.

## Expected Results (Table 3)

| Model | AUROC | AUPRC | F1 |
| --- | ---: | ---: | ---: |
| XGBoost | 0.6676 | 0.2215 | 0.2813 |
| CatBoost | 0.6687 | 0.2214 | 0.2828 |
| LightGBM | 0.6703 | 0.2256 | 0.2833 |
| EBM | 0.6666 | 0.2224 | 0.2794 |
| **MITeNAM (Ours)** | **0.6709** | **0.2236** | **0.2838** |

Under the DeLong test, MITeNAM statistically outperforms XGBoost, CatBoost, and EBM ($p < 0.01$), and is statistically indistinguishable from LightGBM ($p > 0.05$).

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0 (GPU recommended)
- scikit-learn, scipy, numpy, pandas
- xgboost, catboost, lightgbm
- interpret (for EBM)
- matplotlib (for Figure 2)

## License

Released under the MIT License for the CIKM 2026 anonymous review process.
