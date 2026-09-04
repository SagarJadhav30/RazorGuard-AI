"""
RazorGuard AI - Preprocessing Pipeline Builder & Serializer Script
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.preprocessing.pipeline import RazorGuardPreprocessor, separate_target


def main():
    data_path = os.path.join("data", "processed", "processed_fraud_transactions.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run scripts/generate_dataset.py first.")

    print("==================================================================")
    print("  RazorGuard AI - Building & Fitting Preprocessing Pipeline")
    print("==================================================================")

    start_time = datetime.now()
    df = pd.read_csv(data_path)

    # 1. Separate Target
    X, y = separate_target(df, target_col="is_fraud")

    # 2. Train / Test Split (80% Train, 20% Held-Out Test Set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Dataset Loaded: {len(df):,} total records")
    print(f"  - Training Set:   {len(X_train):,} samples")
    print(f"  - Held-Out Test:  {len(X_test):,} samples")

    # 3. Fit Preprocessor ONLY on Training Set
    print("\nFitting RazorGuardPreprocessor strictly on Training Set...")
    preprocessor = RazorGuardPreprocessor()
    preprocessor.fit(X_train)

    # 4. Save Preprocessor Artifact
    artifact_path = os.path.join("models", "preprocessor.joblib")
    preprocessor.save(artifact_path)
    print(f"[+] Serialized fitted preprocessor to: {artifact_path}")

    # 5. Transform Train & Test Sets
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    duration = (datetime.now() - start_time).total_seconds()

    print("\n------------------------------------------------------------------")
    print("  PREPROCESSING PIPELINE BUILD COMPLETE")
    print("------------------------------------------------------------------")
    print(f"  Raw Features Count:        {X.shape[1]}")
    print(f"  Processed Features Count:  {X_train_proc.shape[1]}")
    print(f"  Engineered Features Added: 10")
    print(f"  Preprocessed Array Shape:  {X_train_proc.shape}")
    print(f"  Artifact File Size:        {os.path.getsize(artifact_path) / 1024:.2f} KB")
    print(f"  Completed in:              {duration:.2f} seconds")
    print("==================================================================")


if __name__ == "__main__":
    main()
