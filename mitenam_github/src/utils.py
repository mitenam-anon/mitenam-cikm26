"""
Common utilities — feature engineering and I/O helpers.
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd


# Identifier columns to drop before training
ID_COLS = ["encounter_id", "patient_nbr"]

# Raw label column (replaced by binary `readmit_30d`)
RAW_LABEL_COL = "readmitted"

# Target column (binary, created by cohort.py)
TARGET_COL = "readmit_30d"


def load_cohort(cohort_csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load cohort CSV and prepare feature matrix and target.

    Returns
    -------
    X : np.ndarray
        Feature matrix (n, d) — all columns encoded as float32.
        Categorical strings are label-encoded (deterministic by sorted unique values).
    y : np.ndarray
        Binary target (n,) — int64.
    feature_names : List[str]
        Column names of X in order.
    """
    df = pd.read_csv(cohort_csv_path)
    y = df[TARGET_COL].astype(int).values

    drop_cols = ID_COLS + [TARGET_COL, RAW_LABEL_COL]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode any non-numeric columns deterministically
    for col in X_df.columns:
        if X_df[col].dtype == object:
            unique_vals = sorted(X_df[col].astype(str).unique())
            mapping = {v: i for i, v in enumerate(unique_vals)}
            X_df[col] = X_df[col].astype(str).map(mapping)

    X = X_df.values.astype(np.float32)
    feature_names = X_df.columns.tolist()

    print(f"Loaded cohort: X={X.shape}, y positive={int(y.sum())}/{len(y)} "
          f"({100 * y.mean():.2f}%)")
    return X, y, feature_names


def save_oof(path: str, y_true: np.ndarray, oof_probs: np.ndarray) -> None:
    """
    Save out-of-fold predictions to a pickle file.

    Format: {'y_true': np.array (n,), 'oof_probs': np.array (n,)}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "y_true": np.asarray(y_true).astype(int).ravel(),
        "oof_probs": np.asarray(oof_probs).astype(float).ravel(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Saved: {path}")


def load_oof(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load out-of-fold predictions from a pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return np.asarray(obj["y_true"]).ravel(), np.asarray(obj["oof_probs"]).ravel()


def make_stratified_folds(y: np.ndarray, n_splits: int, seed: int):
    """Return list of (train_idx, val_idx) tuples for stratified k-fold."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
