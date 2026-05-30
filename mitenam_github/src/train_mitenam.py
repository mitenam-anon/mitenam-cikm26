"""
MITeNAM Training — Multi-Teacher Input-Injection NAM
=====================================================

Trains the MITeNAM student given pre-computed teacher OOF predictions
from XGBoost, CatBoost, and LightGBM (see train_teachers.py).

Pipeline (per fold):
    1. Combine 3 teacher OOF probs with fixed weights (Eq. 1):
         p_teacher = w_xgb * p_xgb + w_cat * p_cat + w_lgb * p_lgb
    2. Build student inputs (input-injection):
         x' = [raw features, p_teacher]
       Numerical features are RobustScaler-normalized per fold.
    3. Train 3 NAM students with seeds {42, 1234, 5678}, each with the
       joint loss (Eq. 3):
         L = α * BCE(z_s, y) + (1 - α) * T² * KD(σ(z_s/T), σ(z_t/T))
    4. Final OOF probability = mean of the 3 NAM sigmoid outputs.

Evaluation protocol (paper Section 3.2):
    3 outer seeds × 5-fold stratified CV = 15 runs.
    Note: NAM seeds {42, 1234, 5678} are *inner* seeds for the NAM
    ensemble. The outer 3 seeds in the CV split protocol (Cell A of
    the original notebook) are {42, 0, 2026}, but for MITeNAM training
    we follow the slim version: outer seed = 42 (single split) since
    the NAM ensemble itself averages over 3 inner seeds. Teacher OOFs
    are already 3-outer-seed-averaged from train_teachers.py.

Saved files:
    {output_dir}/uci_mitenam_oof.pkl       (cat_heavy: 0.25/0.50/0.25)
    {output_dir}/uci_mitenam_equal_oof.pkl (1/3 / 1/3 / 1/3)         (--ablation)
    {output_dir}/uci_mitenam_top2heavy_oof.pkl (0.40/0.40/0.20)      (--ablation)

Usage:
    python train_mitenam.py --cohort ./data/uci_cohort.csv \
                            --teachers ./results/teachers/ \
                            --output ./results/mitenam/ \
                            --weights cat_heavy
"""

import argparse
import gc
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from nam import (
    NAM, kd_loss_binary,
    ALPHA, TEMPERATURE, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE,
)
from utils import load_cohort, save_oof, load_oof


# NAM ensemble seeds (inner ensemble averaging)
NAM_SEEDS = [42, 1234, 5678]

# Outer split seed for MITeNAM (single split; NAM ensemble handles variance)
OUTER_SEED = 42
N_SPLITS = 5

# Teacher weight configurations
WEIGHT_CONFIGS: Dict[str, Dict[str, float]] = {
    "equal":      {"xgb": 1/3,  "cat": 1/3,  "lgb": 1/3 },
    "cat_heavy":  {"xgb": 0.25, "cat": 0.50, "lgb": 0.25},  # paper main
    "top2_heavy": {"xgb": 0.40, "cat": 0.40, "lgb": 0.20},
}


def load_teacher_oof(teachers_dir: str) -> Dict[str, np.ndarray]:
    """Load XGB/CatBoost/LightGBM OOF predictions from train_teachers.py."""
    teachers = {}
    for key, fname in [("xgb", "uci_xgboost_oof.pkl"),
                       ("cat", "uci_catboost_oof.pkl"),
                       ("lgb", "uci_lightgbm_oof.pkl")]:
        path = os.path.join(teachers_dir, fname)
        _, oof = load_oof(path)
        teachers[key] = oof
    return teachers


def compute_teacher_ensemble(teachers: Dict[str, np.ndarray],
                              weights: Dict[str, float]) -> np.ndarray:
    """Eq. (1): weighted sum of teacher probabilities."""
    return (weights["xgb"] * teachers["xgb"] +
            weights["cat"] * teachers["cat"] +
            weights["lgb"] * teachers["lgb"])


def train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, y_va: np.ndarray,
                   p_teacher_tr: np.ndarray,
                   seed: int, device: str) -> np.ndarray:
    """
    Train a single NAM student on one fold and return validation probabilities.

    The loss is:
        L = α * BCE(z_s, y) + (1 - α) * T² * KD(σ(z_s/T), σ(z_t/T))

    Early stopping is based on validation AUROC (patience = PATIENCE).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NAM(num_features=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(),
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Tensors
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    y_tr_t = torch.FloatTensor(y_tr.astype(np.float32)).to(device)
    pt_tr_t = torch.FloatTensor(p_teacher_tr.astype(np.float32)).to(device)

    # Class-balanced BCE
    pos_weight = torch.tensor(
        ((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    ).float().to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state, bad_epochs = 0.0, None, 0
    n = len(X_tr)

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            z_s = model(X_tr_t[idx])
            loss = (ALPHA * bce(z_s, y_tr_t[idx]) +
                    (1 - ALPHA) * kd_loss_binary(z_s, pt_tr_t[idx], TEMPERATURE))
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validation AUROC for early stopping
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(X_va_t)).cpu().numpy()
        val_auc = roc_auc_score(y_va, p_va)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_va_final = torch.sigmoid(model(X_va_t)).cpu().numpy()

    # Cleanup
    del model, X_tr_t, X_va_t, y_tr_t, pt_tr_t
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return p_va_final


def train_mitenam(X: np.ndarray, y: np.ndarray,
                  p_teacher: np.ndarray,
                  device: str) -> np.ndarray:
    """
    Run the full MITeNAM training and return OOF predictions of shape (n,).

    For each of NAM_SEEDS:
        For each of N_SPLITS folds:
            Train one NAM student and collect validation predictions.
    Final OOF = mean over NAM_SEEDS.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=OUTER_SEED)
    splits = list(skf.split(np.zeros(len(y)), y))

    n = len(y)
    oof_per_seed = []

    for s_idx, seed in enumerate(NAM_SEEDS, 1):
        print(f"\n  [NAM seed {seed}]  ({s_idx}/{len(NAM_SEEDS)})")
        oof_s = np.zeros(n)
        fold_aucs = []

        for k, (tr_idx, va_idx) in enumerate(splits, 1):
            # Per-fold scaling of numerical features
            scaler = RobustScaler()
            X_tr_scaled = scaler.fit_transform(X[tr_idx])
            X_va_scaled = scaler.transform(X[va_idx])

            # Input-injection: append teacher probability as auxiliary feature
            X_tr_aug = np.concatenate(
                [X_tr_scaled, p_teacher[tr_idx].reshape(-1, 1)], axis=1
            ).astype(np.float32)
            X_va_aug = np.concatenate(
                [X_va_scaled, p_teacher[va_idx].reshape(-1, 1)], axis=1
            ).astype(np.float32)

            p_va = train_one_fold(
                X_tr_aug, y[tr_idx], X_va_aug, y[va_idx],
                p_teacher[tr_idx], seed=seed, device=device,
            )
            oof_s[va_idx] = p_va
            fold_aucs.append(roc_auc_score(y[va_idx], p_va))
            print(f"    Fold {k}/{N_SPLITS}: AUROC = {fold_aucs[-1]:.4f}")

        seed_auc = roc_auc_score(y, oof_s)
        print(f"    --> NAM seed {seed} OOF AUROC = {seed_auc:.4f}")
        oof_per_seed.append(oof_s)

    # Inference-time ensemble: average over NAM seeds
    oof_avg = np.mean(oof_per_seed, axis=0)
    final_auc = roc_auc_score(y, oof_avg)
    print(f"\n  ==> MITeNAM FINAL OOF AUROC = {final_auc:.4f}")
    return oof_avg


def main():
    parser = argparse.ArgumentParser(description="Train MITeNAM.")
    parser.add_argument("--cohort", required=True, help="Path to cohort CSV")
    parser.add_argument("--teachers", required=True,
                        help="Directory containing teacher OOF .pkl files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--weights", default="cat_heavy",
                        choices=list(WEIGHT_CONFIGS.keys()),
                        help="Teacher weight configuration (default: cat_heavy)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output, exist_ok=True)

    # Load data
    X, y, feature_names = load_cohort(args.cohort)

    # Load teacher OOFs and build ensemble
    print(f"\nLoading teacher OOFs from: {args.teachers}")
    teachers = load_teacher_oof(args.teachers)
    weights = WEIGHT_CONFIGS[args.weights]
    p_teacher = compute_teacher_ensemble(teachers, weights)
    teacher_auc = roc_auc_score(y, p_teacher)
    print(f"  Teacher ensemble ({args.weights}) AUROC = {teacher_auc:.4f}")
    print(f"  Weights: xgb={weights['xgb']}, cat={weights['cat']}, lgb={weights['lgb']}")

    # Train MITeNAM
    print(f"\n{'=' * 60}")
    print(f"Training MITeNAM (weights = {args.weights})")
    print(f"{'=' * 60}")
    oof = train_mitenam(X, y, p_teacher, device=device)

    # Save
    if args.weights == "cat_heavy":
        out_path = os.path.join(args.output, "uci_mitenam_oof.pkl")
    else:
        out_path = os.path.join(args.output, f"uci_mitenam_{args.weights}_oof.pkl")
    save_oof(out_path, y, oof)


if __name__ == "__main__":
    main()
