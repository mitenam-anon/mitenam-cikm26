"""
End-to-End Reproducibility Script
==================================

Runs the full MITeNAM pipeline:
    1. Generate cohort (101,766 -> 99,343)
    2. Train teacher models (XGBoost, CatBoost, LightGBM, EBM)
    3. Train MITeNAM (cat_heavy: 0.25 / 0.50 / 0.25)
    4. (Optional) Train ablation variants (equal, top2_heavy)
    5. Evaluate everything and generate result tables

Total runtime on Google Colab (T4 GPU): ~3 hours.

Usage:
    # Download diabetic_data.csv from UCI Machine Learning Repository
    # https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
    # and place it in data/

    python scripts/run_full_pipeline.py --data_dir ./data --output_dir ./results
"""

import argparse
import os
import subprocess
import sys


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def run(cmd_args):
    """Run a python -m command from the src directory."""
    print(f"\n{'=' * 70}")
    print(f"$ python {' '.join(cmd_args)}")
    print('=' * 70)
    result = subprocess.run(
        [sys.executable] + cmd_args,
        cwd=SRC_DIR, check=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full MITeNAM pipeline.")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing diabetic_data.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for cohort, models, tables")
    parser.add_argument("--ablation", action="store_true",
                        help="Also train equal and top2_heavy variants")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    raw_csv = os.path.join(data_dir, "diabetic_data.csv")
    cohort_csv = os.path.join(data_dir, "uci_cohort.csv")
    teachers_dir = os.path.join(output_dir, "teachers")
    mitenam_dir = os.path.join(output_dir, "mitenam")
    tables_dir = os.path.join(output_dir, "tables")

    # 1. Cohort
    if not os.path.exists(cohort_csv):
        run(["cohort.py", "--input", raw_csv, "--output", cohort_csv])
    else:
        print(f"\n[SKIP] cohort already exists at {cohort_csv}")

    # 2. Teachers
    run(["train_teachers.py", "--cohort", cohort_csv, "--output", teachers_dir])

    # 3. MITeNAM (cat_heavy, main model)
    run(["train_mitenam.py",
         "--cohort", cohort_csv,
         "--teachers", teachers_dir,
         "--output", mitenam_dir,
         "--weights", "cat_heavy"])

    # 4. Ablation (optional)
    if args.ablation:
        for w in ["equal", "top2_heavy"]:
            run(["train_mitenam.py",
                 "--cohort", cohort_csv,
                 "--teachers", teachers_dir,
                 "--output", mitenam_dir,
                 "--weights", w])

    # 5. Evaluate
    run(["evaluate.py", "--results", output_dir, "--output", tables_dir])

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)
    print(f"  Cohort   : {cohort_csv}")
    print(f"  Teachers : {teachers_dir}")
    print(f"  MITeNAM  : {mitenam_dir}")
    print(f"  Tables   : {tables_dir}")


if __name__ == "__main__":
    main()
