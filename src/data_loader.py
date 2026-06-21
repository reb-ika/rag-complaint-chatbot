"""
data_loader.py
Handles loading and filtering the CFPB complaint dataset.
"""

import pandas as pd
import os

# Products we care about and their clean category names
TARGET_PRODUCTS = [
    'Credit card',
    'Credit card or prepaid card',
    'Checking or savings account',
    'Bank account or service',
    'Money transfer, virtual currency, or money service',
    'Money transfers',
    'Consumer Loan',
    'Payday loan, title loan, or personal loan',
    'Payday loan, title loan, personal loan, or advance loan'
]

PRODUCT_MAPPING = {
    'Credit card': 'Credit Card',
    'Credit card or prepaid card': 'Credit Card',
    'Checking or savings account': 'Savings Account',
    'Bank account or service': 'Savings Account',
    'Money transfer, virtual currency, or money service': 'Money Transfer',
    'Money transfers': 'Money Transfer',
    'Consumer Loan': 'Personal Loan',
    'Payday loan, title loan, or personal loan': 'Personal Loan',
    'Payday loan, title loan, personal loan, or advance loan': 'Personal Loan'
}


def clean_text(text):
    """
    Clean a single complaint narrative.
    Lowercases, removes boilerplate and extra whitespace.
    """
    import re
    try:
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Warning: could not clean text — {e}")
        return ""


def load_and_filter(raw_path, output_path, chunk_size=50000):
    """
    Load the raw CFPB CSV in chunks, filter for target products,
    clean narratives, and save to output_path.

    Args:
        raw_path (str): Path to the raw complaints CSV
        output_path (str): Path to save the filtered CSV
        chunk_size (int): Number of rows to process at a time

    Returns:
        int: Total number of rows saved
    """
    # Validate input file exists
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_saved = 0
    writer_done = False

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):

                # Filter for target products
                chunk = chunk[chunk['Product'].isin(TARGET_PRODUCTS)]

                # Remove empty narratives
                chunk = chunk[chunk['Consumer complaint narrative'].notna()]

                if len(chunk) == 0:
                    continue

                # Map to clean category names
                chunk['product_category'] = chunk['Product'].map(PRODUCT_MAPPING)

                # Clean narrative text
                chunk['cleaned_narrative'] = chunk['Consumer complaint narrative'].apply(clean_text)

                # Remove very short narratives
                chunk = chunk[chunk['cleaned_narrative'].str.len() >= 50]

                # Keep only useful columns
                chunk_final = chunk[[
                    'Complaint ID', 'Date received', 'Product',
                    'product_category', 'Issue', 'Sub-issue',
                    'Company', 'State', 'cleaned_narrative'
                ]].copy()

                chunk_final.columns = [
                    'complaint_id', 'date_received', 'product',
                    'product_category', 'issue', 'sub_issue',
                    'company', 'state', 'cleaned_narrative'
                ]

                if not writer_done:
                    chunk_final.to_csv(f, index=False)
                    writer_done = True
                else:
                    chunk_final.to_csv(f, index=False, header=False)

                total_saved += len(chunk_final)
                print(f"Saved {total_saved} rows...")

    except Exception as e:
        raise RuntimeError(f"Failed to process data: {e}")

    print(f"\nDone! Total rows saved: {total_saved}")
    return total_saved


def load_filtered(path, nrows=None):
    """
    Load the already-filtered complaints CSV.

    Args:
        path (str): Path to filtered_complaints.csv
        nrows (int): Optional limit on rows to load

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Filtered data not found: {path}")

    try:
        df = pd.read_csv(path, nrows=nrows)
        print(f"Loaded {len(df)} rows from {path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load filtered data: {e}")


def create_stratified_sample(df, sample_size=10000, random_state=42):
    """
    Create a stratified sample ensuring equal representation
    across all product categories.

    Args:
        df (pd.DataFrame): Full filtered dataframe
        sample_size (int): Total number of samples desired
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Stratified sample
    """
    categories = df['product_category'].unique()
    per_category = sample_size // len(categories)

    print(f"Creating stratified sample:")
    print(f"  Total categories: {len(categories)}")
    print(f"  Samples per category: {per_category}")

    samples = []
    for category in categories:
        cat_df = df[df['product_category'] == category]
        n = min(per_category, len(cat_df))
        sample = cat_df.sample(n, random_state=random_state)
        samples.append(sample)
        print(f"  {category}: {n} samples")

    df_sample = pd.concat(samples, ignore_index=True)
    print(f"\nTotal stratified sample size: {len(df_sample)}")
    return df_sample