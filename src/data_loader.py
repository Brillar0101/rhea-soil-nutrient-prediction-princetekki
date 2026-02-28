"""
Data loading and initial preprocessing for Rhea competition.
"""
import pandas as pd
import numpy as np
from config import *


def load_all_data():
    """Load all competition data files."""
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    sample_dates = pd.read_csv(SAMPLE_DATES_FILE)
    target_keep = pd.read_csv(TARGET_KEEP_FILE)
    sample_sub = pd.read_csv(SAMPLE_SUB_FILE)

    try:
        data_dict = pd.read_csv(DATA_DICT_FILE)
        print("Data Dictionary:")
        print(data_dict.to_string())
        print()
    except Exception:
        data_dict = None

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample dates shape: {sample_dates.shape}")
    print(f"Target keep shape: {target_keep.shape}")
    print(f"Sample submission shape: {sample_sub.shape}")
    print()
    print(f"Train columns: {list(train.columns)}")
    print(f"Test columns: {list(test.columns)}")
    print(f"Sample dates columns: {list(sample_dates.columns)}")
    print(f"Target keep columns: {list(target_keep.columns)}")

    return train, test, sample_dates, target_keep, sample_sub, data_dict


def explore_data(train, test, sample_dates, target_keep):
    """Detailed EDA of the datasets."""
    print("=" * 80)
    print("TRAIN DATA EXPLORATION")
    print("=" * 80)
    print(train.head())
    print()
    print(train.describe())
    print()
    print("Missing values in train:")
    print(train.isnull().sum())
    print()
    print("Data types:")
    print(train.dtypes)
    print()

    print("=" * 80)
    print("TEST DATA EXPLORATION")
    print("=" * 80)
    print(test.head())
    print()
    print(test.describe())
    print()
    print("Missing values in test:")
    print(test.isnull().sum())
    print()

    print("=" * 80)
    print("SAMPLE DATES EXPLORATION")
    print("=" * 80)
    print(sample_dates.head())
    print()
    print(sample_dates.describe())
    print()

    print("=" * 80)
    print("TARGET KEEP EXPLORATION")
    print("=" * 80)
    print(target_keep.head())
    print()
    # How many targets exist per nutrient
    for col in TARGETS:
        if col in target_keep.columns:
            print(f"  {col}: {target_keep[col].sum()} entries to predict")

    print()

    # Check target distributions
    print("=" * 80)
    print("TARGET DISTRIBUTIONS IN TRAIN")
    print("=" * 80)
    for col in TARGETS:
        if col in train.columns:
            vals = train[col].dropna()
            print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"min={vals.min():.4f}, max={vals.max():.4f}, "
                  f"skew={vals.skew():.4f}, missing={train[col].isnull().sum()}")


if __name__ == "__main__":
    train, test, sample_dates, target_keep, sample_sub, data_dict = load_all_data()
    explore_data(train, test, sample_dates, target_keep)
