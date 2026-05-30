"""
UCI Diabetes 130-US Hospitals — Cohort Generation
==================================================

Filters raw `diabetic_data.csv` (n = 101,766) to the analysis cohort
(n = 99,343) used in the paper.

Cohort filter (single step):
    Exclude patients who expired or were transferred to hospice
    (discharge_disposition_id in {11, 13, 14, 19, 20, 21}).

Target:
    readmit_30d = 1 if readmitted == '<30' else 0
    Positive rate: 11.39% (11,314 / 99,343)

Usage:
    python cohort.py --input ./data/diabetic_data.csv \
                     --output ./data/uci_cohort.csv
"""

import argparse
import pandas as pd


# Patients who expired or transferred to hospice (UCI codebook):
#   11: expired
#   13: hospice / home
#   14: hospice / medical facility
#   19: expired at home, Medicaid only, hospice
#   20: expired in a medical facility (e.g. hospital or SNF), Medicaid only, hospice
#   21: expired, place unknown, Medicaid only, hospice
EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


def build_cohort(raw_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Build the analysis cohort and save to CSV.

    Parameters
    ----------
    raw_csv_path : str
        Path to raw UCI Diabetes CSV (diabetic_data.csv).
    output_csv_path : str
        Path to save the filtered cohort CSV.

    Returns
    -------
    pd.DataFrame
        Cohort dataframe with the target column `readmit_30d` added.
    """
    print(f"Loading raw data from: {raw_csv_path}")
    raw = pd.read_csv(raw_csv_path)
    print(f"  Raw size: {len(raw):,}")

    # Filter: exclude expired / hospice patients
    cohort = raw[~raw["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)].copy()
    cohort = cohort.reset_index(drop=True)
    print(f"  After excluding expired/hospice: {len(cohort):,}")

    # Create binary target
    cohort["readmit_30d"] = (cohort["readmitted"] == "<30").astype(int)

    n_pos = int(cohort["readmit_30d"].sum())
    n_total = len(cohort)
    print(f"  Positive (30-day readmit): {n_pos:,} ({100 * n_pos / n_total:.2f}%)")

    cohort.to_csv(output_csv_path, index=False)
    print(f"Saved cohort to: {output_csv_path}")

    return cohort


def main():
    parser = argparse.ArgumentParser(description="Build UCI Diabetes cohort.")
    parser.add_argument("--input", required=True,
                        help="Path to raw diabetic_data.csv")
    parser.add_argument("--output", required=True,
                        help="Path to save the cohort CSV")
    args = parser.parse_args()

    build_cohort(args.input, args.output)


if __name__ == "__main__":
    main()
