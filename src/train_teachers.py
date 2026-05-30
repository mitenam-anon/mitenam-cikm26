"""
Train Baseline Models — XGBoost, CatBoost, LightGBM, EBM
=========================================================

All baselines are evaluated with 3 outer seeds × 5-fold stratified
cross-validation (15 runs total). Out-of-fold (OOF) predictions are
saved for downstream use by MITeNAM (as teacher signals) and the
evaluation script (for AUROC/AUPRC/F1/DeLong test).

Saved files (one per baseline):
    {output_dir}/uci_xgboost_oof.pkl
    {output_dir}/uci_catboost_oof.pkl
    {output_dir}/uci_lightgbm_oof.pkl
    {output_dir}/uci_ebm_oof.pkl

Each file is a dict: {'y_true': (n,), 'oof_probs': (n,)}.

Hyperparameters (paper Section 3.2):
    XGBoost / CatBoost / LightGBM:
        n_estimators = 300, max_depth = 6, learning_rate = 0.05
        scale_pos_weight = (n_neg / n_pos)
    EBM:
        max_bins = 256, interactions = 10

Usage:
    python train_teachers.py --cohort ./data/uci_cohort.csv \
                             --output ./results/teachers/
"""

import argparse
import os
from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils import load_cohort, save_oof


# 3 outer seeds × 5-fold = 15 runs (paper Section 3.2)
OUTER_SEEDS = [42, 0, 2026]
N_SPLITS = 5

# Per-model output filenames
MODEL_FILES = {
    "xgboost":  "uci_xgboost_oof.pkl",
    "catboost": "uci_catboost_oof.pkl",
    "lightgbm": "uci_lightgbm_oof.pkl",
    "ebm":      "uci_ebm_oof.pkl",
}


def fit_predict_xgboost(X_tr, y_tr, X_te):
    import xgboost as xgb
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw, tree_method="hist",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def fit_predict_catboost(X_tr, y_tr, X_te):
    from catboost import CatBoostClassifier
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="AUC",
        random_state=42, verbose=False, thread_count=-1,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def fit_predict_lightgbm(X_tr, y_tr, X_te):
    import lightgbm as lgb
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=spw, objective="binary",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def fit_predict_ebm(X_tr, y_tr, X_te):
    from interpret.glassbox import ExplainableBoostingClassifier
    model = ExplainableBoostingClassifier(
        max_bins=256, interactions=10, n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


MODEL_FNS = {
    "xgboost":  fit_predict_xgboost,
    "catboost": fit_predict_catboost,
    "lightgbm": fit_predict_lightgbm,
    "ebm":      fit_predict_ebm,
}


def run_3seed_5fold(name: str, fit_predict_fn: Callable,
                    X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Run 3 outer seeds × 5-fold stratified CV.

    Returns
    -------
    np.ndarray
        OOF predictions of shape (n,) — average over 3 outer seeds.
    """
    from sklearn.metrics import roc_auc_score

    n = len(y)
    oof_seeds = np.zeros((len(OUTER_SEEDS), n))

    for s_idx, seed in enumerate(OUTER_SEEDS):
        print(f"\n  [Seed {seed}]")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof = np.zeros(n)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            p = fit_predict_fn(X[tr_idx], y[tr_idx], X[te_idx])
            seed_oof[te_idx] = p
            auc = roc_auc_score(y[te_idx], p)
            print(f"    Fold {fold}/{N_SPLITS}: AUROC = {auc:.4f}")
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"    --> Seed {seed} OOF AUROC = {seed_auc:.4f}")
        oof_seeds[s_idx] = seed_oof

    oof_avg = oof_seeds.mean(axis=0)
    final_auc = roc_auc_score(y, oof_avg)
    print(f"\n  ==> {name} FINAL AUROC (3-seed avg) = {final_auc:.4f}")
    return oof_avg


def main():
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--cohort", required=True, help="Path to cohort CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--models", nargs="+",
                        default=list(MODEL_FILES.keys()),
                        help="Which models to train (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    X, y, _ = load_cohort(args.cohort)

    for name in args.models:
        if name not in MODEL_FNS:
            print(f"[WARN] Unknown model '{name}', skipping.")
            continue
        out_path = os.path.join(args.output, MODEL_FILES[name])
        if os.path.exists(out_path):
            print(f"\n[SKIP] {name} — already exists at {out_path}")
            continue
        print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}")
        oof = run_3seed_5fold(name, MODEL_FNS[name], X, y)
        save_oof(out_path, y, oof)


if __name__ == "__main__":
    main()
