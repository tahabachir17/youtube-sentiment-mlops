"""
Download script for the Reddit Sentiment Analysis dataset.
Downloads raw dataset from GitHub and displays descriptive statistics.
"""

import os
import sys
import urllib.request
import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis"
    "/refs/heads/main/data/reddit.csv"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit.csv")


def download_dataset(url: str = DATASET_URL, output_path: str = OUTPUT_FILE) -> str:
    """
    Downloads dataset from GitHub and saves it to output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[*] Downloading dataset from: {url}")
    print(f"[*] Target destination: {output_path}")

    # Set User-Agent to avoid any 403 blocks from GitHub raw endpoints
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Download complete! File size: {file_size_kb:.2f} KB")
    return output_path


def inspect_raw_dataset(file_path: str = OUTPUT_FILE) -> pd.DataFrame:
    """
    Reads the downloaded CSV and prints key statistics.
    """
    print("\n" + "=" * 50)
    print("      DATASET EXPLORATION & STATISTICS")
    print("=" * 50)

    df = pd.read_csv(file_path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Check for target column
    target_col = None
    for col in ["category", "label", "sentiment"]:
        if col in df.columns:
            target_col = col
            break

    if target_col:
        print(f"\nLabel Distribution ('{target_col}'):")
        counts = df[target_col].value_counts(dropna=False).sort_index()
        percentages = df[target_col].value_counts(normalize=True, dropna=False).sort_index() * 100
        stats_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages.round(2)})
        print(stats_df)

    print("\nFirst 3 rows:")
    print(df.head(3))
    print("=" * 50)
    return df


if __name__ == "__main__":
    download_dataset()
    inspect_raw_dataset()
