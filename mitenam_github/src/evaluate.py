"""
Evaluation — Compute Table 3 (Main Results) and Table for Ablation
===================================================================

Given saved OOF probability files from train_teachers.py and
train_mitenam.py, this script computes:
    - AUROC, AUPRC, F1 (at F1-maximizing threshold), Brier score
    - DeLong test p-value for AUROC comparison vs MITeNAM
    - Markdown table, LaTeX table, and CSV outputs

DeLong test implementation:
    Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm for
    Comparing the Areas Under Correlated Receiver Operating
    Characteristic Curves".

Usage:
    python evaluate.py --results ./results/ --output ./results/tables/
"""

import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              f1_score, roc_auc_score)

from utils import load_oof


REFERENCE_MODEL = "MITeNAM (Ours)"

MAIN_RESULTS_FILES = {
    "MITeNAM (Ours)": "mitenam/uci_mitenam_oof.pkl",
    "XGBoost":        "teachers/uci_xgboost_oof.pkl",
    "CatBoost":       "teachers/uci_catboost_oof.pkl",
    "LightGBM":       "teachers/uci_lightgbm_oof.pkl",
    "EBM":            "teachers/uci_ebm_oof.pkl",
}

ABLATION_FILES = {
    "Equal (1/3, 1/3, 1/3)":               "mitenam/uci_mitenam_equal_oof.pkl",
    "cat_heavy (0.25, 0.50, 0.25) [Ours]": "mitenam/uci_mitenam_oof.pkl",
    "top2_heavy (0.40, 0.40, 0.20)":       "mitenam/uci_mitenam_top2heavy_oof.pkl",
}


# ----------------------------------------------------------------------
# DeLong test — Sun & Xu (2014) fast implementation
# ----------------------------------------------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive_examples = preds_sorted_transposed[:, :m]
    negative_examples = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_p_value(y_true: np.ndarray,
                   prob_a: np.ndarray,
                   prob_b: np.ndarray) -> float:
    """Return p-value for AUROC(A) vs AUROC(B) (two-sided)."""
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    preds_sorted = np.vstack([prob_a, prob_b])[:, order]
    aucs, delongcov = _fast_delong(preds_sorted, label_1_count)
    l = np.array([[1, -1]])
    z = float(l @ aucs) / np.sqrt(float(l @ delongcov @ l.T))
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(pval)


# ----------------------------------------------------------------------
# F1 at F1-maximizing threshold
# ----------------------------------------------------------------------
def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Grid search threshold that maximizes F1."""
    thresholds = np.unique(y_prob)
    if len(thresholds) > 200:
        thresholds = np.quantile(y_prob, np.linspace(0.01, 0.99, 199))
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, oof: np.ndarray) -> Dict[str, float]:
    auroc = roc_auc_score(y_true, oof)
    auprc = average_precision_score(y_true, oof)
    brier = brier_score_loss(y_true, oof)
    thr = find_best_f1_threshold(y_true, oof)
    f1 = f1_score(y_true, (oof >= thr).astype(int), zero_division=0)
    return {"AUROC": auroc, "AUPRC": auprc, "F1": f1, "Brier": brier, "thr": thr}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def evaluate(results_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: load all models and compute basic metrics
    print("=" * 70)
    print("  Phase 1: Loading OOF predictions and computing metrics")
    print("=" * 70)

    y_true_global: Optional[np.ndarray] = None
    main_results: Dict[str, dict] = {}

    for model_name, rel_path in MAIN_RESULTS_FILES.items():
        path = os.path.join(results_dir, rel_path)
        if not os.path.exists(path):
            print(f"\n[{model_name}] SKIP — file not found at {path}")
            main_results[model_name] = None
            continue
        y_true, oof = load_oof(path)
        if y_true_global is None:
            y_true_global = y_true
            print(f"\n[INFO] y_true: n = {len(y_true_global):,}, "
                  f"positive rate = {100 * y_true_global.mean():.2f}%")
        m = compute_metrics(y_true_global, oof)
        m["oof"] = oof
        main_results[model_name] = m
        print(f"\n[{model_name}]")
        print(f"  AUROC = {m['AUROC']:.4f}  AUPRC = {m['AUPRC']:.4f}  "
              f"F1 = {m['F1']:.4f} (thr = {m['thr']:.3f})  Brier = {m['Brier']:.4f}")

    # Phase 2: DeLong test vs MITeNAM
    print("\n" + "=" * 70)
    print(f"  Phase 2: DeLong test (vs {REFERENCE_MODEL})")
    print("=" * 70)
    ref = main_results.get(REFERENCE_MODEL)
    if ref is not None:
        for model, r in main_results.items():
            if r is None or model == REFERENCE_MODEL:
                continue
            p = delong_p_value(y_true_global.astype(float),
                               ref["oof"], r["oof"])
            r["DeLong_p"] = p
            print(f"  {model:<15} p = {p:.6f}")

    # Phase 3: Main result table — Markdown + LaTeX + CSV
    print("\n" + "=" * 70)
    print("  Phase 3: Main results table")
    print("=" * 70)

    rows = []
    for model, r in main_results.items():
        if r is None:
            continue
        rows.append({
            "Model": model,
            "AUROC": round(r["AUROC"], 4),
            "AUPRC": round(r["AUPRC"], 4),
            "F1":    round(r["F1"], 4),
            "Brier": round(r["Brier"], 4),
            "DeLong_p_vs_MITeNAM":
                None if model == REFERENCE_MODEL else round(r.get("DeLong_p", np.nan), 6),
        })
    df_main = pd.DataFrame(rows)
    print("\n" + df_main.to_string(index=False))

    csv_path = os.path.join(output_dir, "table1_uci_main.csv")
    df_main.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Phase 4: Ablation table
    print("\n" + "=" * 70)
    print("  Phase 4: Teacher Weight Ablation")
    print("=" * 70)
    ab_rows = []
    weight_map = {
        "Equal (1/3, 1/3, 1/3)":               ("0.333", "0.333", "0.333"),
        "cat_heavy (0.25, 0.50, 0.25) [Ours]": ("0.25",  "0.50",  "0.25"),
        "top2_heavy (0.40, 0.40, 0.20)":       ("0.40",  "0.40",  "0.20"),
    }
    for cfg, rel_path in ABLATION_FILES.items():
        path = os.path.join(results_dir, rel_path)
        if not os.path.exists(path):
            print(f"  [{cfg}] SKIP — file not found")
            continue
        y_true, oof = load_oof(path)
        auroc = roc_auc_score(y_true, oof)
        w = weight_map[cfg]
        ab_rows.append({
            "Configuration": cfg,
            "w_xgb": w[0], "w_cat": w[1], "w_lgb": w[2],
            "AUROC": round(auroc, 4),
        })
        print(f"  {cfg:<45} AUROC = {auroc:.4f}")

    if ab_rows:
        df_ab = pd.DataFrame(ab_rows)
        ab_csv_path = os.path.join(output_dir, "table2_uci_ablation.csv")
        df_ab.to_csv(ab_csv_path, index=False)
        print(f"\nSaved: {ab_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MITeNAM and baselines.")
    parser.add_argument("--results", required=True,
                        help="Root directory containing teachers/ and mitenam/ subdirs")
    parser.add_argument("--output", required=True, help="Output directory for tables")
    args = parser.parse_args()

    evaluate(args.results, args.output)


if __name__ == "__main__":
    main()
