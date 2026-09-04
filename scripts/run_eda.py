"""
RazorGuard AI - Run EDA & Generate Jupyter Notebook CLI Script
"""

import sys
import os
import json
from datetime import datetime

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.evaluation.eda_analyzer import run_comprehensive_eda


def create_jupyter_notebook_artifact(output_path: str = "notebooks/01_exploratory_data_analysis.ipynb"):
    """
    Generates a clean, executable Jupyter Notebook for Exploratory Data Analysis.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RazorGuard AI - Exploratory Data Analysis (EDA)\n",
                    "## Track 02 — Explainable AI Risk Manager for Payment Fraud\n",
                    "\n",
                    "This notebook performs comprehensive exploratory analysis on the **100,000 payment transaction dataset**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "plt.style.use('dark_background')\n",
                    "df = pd.read_csv('../data/processed/processed_fraud_transactions.csv')\n",
                    "print(f'Dataset Shape: {df.shape}')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Data Integrity & Class Distribution"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Missing Values:', df.isnull().sum().sum())\n",
                    "print('Duplicate Tx IDs:', df['transaction_id'].duplicated().sum())\n",
                    "print('\\nClass Distribution:')\n",
                    "print(df['is_fraud'].value_counts(normalize=True).map('{:.2%}'.format))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Transaction Amount Dynamics (Legit vs Fraud)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.groupby('is_fraud')['amount'].describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Top Feature Correlations with Fraud Target"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "numeric_df = df.select_dtypes(include=[np.number])\n",
                    "corr = numeric_df.corr()['is_fraud'].sort_values(ascending=False)\n",
                    "print(corr)"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


def main():
    print("==================================================================")
    print("  RazorGuard AI - Executing Comprehensive Exploratory Data Analysis")
    print("==================================================================")

    start_time = datetime.now()
    summary = run_comprehensive_eda()
    create_jupyter_notebook_artifact()

    duration = (datetime.now() - start_time).total_seconds()

    print("\n------------------------------------------------------------------")
    print("  EDA ANALYSIS COMPLETE")
    print("------------------------------------------------------------------")
    print(f"  Dataset Dimensions:   {summary['dataset_dimensions']['records']:,} rows × {summary['dataset_dimensions']['features']} columns")
    print(f"  Class Balance:         {summary['class_imbalance']['legitimate']:,} Legit | {summary['class_imbalance']['fraud']:,} Fraud ({summary['class_imbalance']['fraud_rate_pct']}%)")
    print(f"  Missing / Duplicates:  {summary['quality']['missing_values']} missing, {summary['quality']['duplicate_records']} duplicates")
    print(f"  Report Exported:       docs/eda_report.md")
    print(f"  Charts Directory:      docs/eda_charts/")
    print(f"  Jupyter Notebook:      notebooks/01_exploratory_data_analysis.ipynb")
    print(f"  Completed in:          {duration:.2f} seconds")
    print("==================================================================")


if __name__ == "__main__":
    main()
