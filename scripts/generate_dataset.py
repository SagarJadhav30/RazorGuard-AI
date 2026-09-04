"""
RazorGuard AI - Dataset Generation & Validation CLI Script
"""

import sys
import os
import argparse
from datetime import datetime

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.utils.data_generator import generate_synthetic_fraud_dataset, save_dataset_files
from ml.preprocessing.validator import validate_payment_dataset


def main():
    parser = argparse.ArgumentParser(description="RazorGuard AI Synthetic Dataset Generator")
    parser.add_argument("--samples", type=int, default=100000, help="Number of records to generate (default 100000)")
    parser.add_argument("--fraud-rate", type=float, default=0.04, help="Fraud rate percentage (default 0.04)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default 42)")
    args = parser.parse_args()

    print("==================================================================")
    print("  RazorGuard AI - Generating Synthetic Payment Fraud Dataset")
    print(f"  Samples: {args.samples:,} | Fraud Rate: {args.fraud_rate*100:.1f}% | Seed: {args.seed}")
    print("==================================================================")

    start_time = datetime.now()
    df = generate_synthetic_fraud_dataset(n_samples=args.samples, fraud_rate=args.fraud_rate, random_seed=args.seed)

    raw_path, proc_path = save_dataset_files(df)
    print(f"[+] Dataset saved to:")
    print(f"    - Raw: {raw_path}")
    print(f"    - Processed: {proc_path}")

    # Validate dataset
    print("\nExecuting Quality Audit & Validation...")
    report = validate_payment_dataset(df)

    duration = (datetime.now() - start_time).total_seconds()

    print("\n------------------------------------------------------------------")
    print(f"  VALIDATION STATUS: {report['validation_status']}")
    print("------------------------------------------------------------------")
    print(f"  Dataset Shape:         {report['dataset_shape']['num_records']:,} rows × {report['dataset_shape']['num_features']} columns")
    print(f"  Legitimate Records:    {report['class_distribution']['legitimate_count']:,}")
    print(f"  Fraudulent Records:    {report['class_distribution']['fraud_count']:,} ({report['class_distribution']['fraud_rate_percentage']}%)")
    print(f"  Missing Values:        {report['data_integrity']['total_missing_values']}")
    print(f"  Duplicate Tx IDs:      {report['data_integrity']['duplicate_transaction_ids']}")
    print(f"  Invalid Values:        {report['data_integrity']['total_invalid_values']}")
    print(f"  Max Feature Corr (y):  {report['target_leakage_audit']['max_feature_correlation']}")
    print(f"  Report Exported:       docs/dataset_report.json")
    print(f"  Completed in:          {duration:.2f} seconds")
    print("==================================================================")


if __name__ == "__main__":
    main()
