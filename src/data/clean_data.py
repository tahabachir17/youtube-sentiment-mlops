"""
Data cleaning and preprocessing pipeline.
Cleans raw Reddit dataset, standardizes columns to 'text' and 'label',
removes missing/empty entries, performs stratified train/test split,
and saves processed datasets into data/processed/.
"""

import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Add root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.text_cleaning import clean_text

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "reddit.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "data_summary.json")


def clean_and_split_data(
    raw_path: str = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw Reddit dataset, normalizes text, splits into train/test,
    and persists datasets in data/processed/.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    print(f"[*] Reading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_count = len(df)

    # Standardize column names
    col_mapping = {}
    if "clean_comment" in df.columns:
        col_mapping["clean_comment"] = "text"
    elif "comment" in df.columns:
        col_mapping["comment"] = "text"

    if "category" in df.columns:
        col_mapping["category"] = "label"
    elif "sentiment" in df.columns:
        col_mapping["sentiment"] = "label"

    df = df.rename(columns=col_mapping)

    # Ensure required columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected 'text' and 'label' columns, got {list(df.columns)}")

    # Drop nulls
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)

    # Filter to valid labels (-1, 0, 1)
    df = df[df["label"].isin([-1, 0, 1])].copy()

    print("[*] Applying text cleaning & emoji normalization...")
    df["text"] = df["text"].apply(clean_text)

    # Filter out empty or whitespace-only texts
    df = df[df["text"].str.len() > 2].copy()
    df = df.drop_duplicates(subset=["text"]).copy()
    final_count = len(df)

    print(f"[+] Cleaned rows: {final_count} (dropped {initial_count - final_count} invalid/duplicate/empty rows)")

    # Stratified train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )

    # Save to processed directory
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"[+] Saved train dataset to: {TRAIN_PATH} ({len(train_df)} rows)")
    print(f"[+] Saved test dataset to:  {TEST_PATH} ({len(test_df)} rows)")

    # Compute summary metrics
    summary = {
        "raw_total": initial_count,
        "clean_total": final_count,
        "train_count": len(train_df),
        "test_count": len(test_df),
        "train_class_distribution": train_df["label"].value_counts().to_dict(),
        "test_class_distribution": test_df["label"].value_counts().to_dict(),
        "train_class_percentages": (train_df["label"].value_counts(normalize=True) * 100).round(2).to_dict(),
        "text_length_mean": float(df["text"].str.len().mean()),
        "text_word_count_mean": float(df["text"].str.split().str.len().mean())
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved data summary report to: {REPORT_PATH}")

    print("\nTrain class distribution:")
    print(train_df["label"].value_counts().sort_index())
    print("\nTest class distribution:")
    print(test_df["label"].value_counts().sort_index())

    return train_df, test_df


if __name__ == "__main__":
    clean_and_split_data()
