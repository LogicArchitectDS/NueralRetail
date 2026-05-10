"""
Bronze data ingestion for the Online Retail II dataset.
Replaces the old Olist ingestion path.
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

def _resolve_data_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    
    env_value = os.environ.get("NEURALRETAIL_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()

    candidates = [
        Path("/home/seshu/NueralRetail_Solo/data"),
        project_root / "data",
        Path.cwd() / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    
    raise FileNotFoundError("Could not find a data directory.")

def ingest_online_retail():
    """Ingest Online Retail II Excel data into the Bronze Parquet layer."""
    data_root = _resolve_data_root()
    landing_path = data_root / "landing" / "online_retail_ii" / "online_retail_ii.xlsx"
    bronze_path = data_root / "bronze" / "transactions.parquet"

    print(f"[INGEST] Reading from: {landing_path}")
    if not landing_path.exists():
        raise FileNotFoundError(f"Source file not found: {landing_path}")

    # Load Excel data
    df = pd.read_excel(landing_path)

    # Standardize column names to lowercase snake_case
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Rename columns to canonical names
    rename_map = {
        'invoice': 'invoice_no',
        'stockcode': 'stock_code',
        'invoicedate': 'invoice_date',
        'price': 'unit_price',
        'customer_id': 'customer_id' 
    }
    df = df.rename(columns=rename_map)

    # Derive canonical fields
    df['invoice_no'] = df['invoice_no'].astype(str)
    df['stock_code'] = df['stock_code'].astype(str)
    df['is_cancelled'] = df['invoice_no'].str.startswith('C')
    df['revenue'] = df['quantity'] * df['unit_price']

    print(f"[INGEST] Initial row count: {len(df)}")

    # Data cleaning rules
    # 1. Remove rows with missing invoice_date
    df = df.dropna(subset=['invoice_date'])
    
    # 2. Remove rows with missing stock_code
    df = df.dropna(subset=['stock_code'])

    # 3. Handle cancellations and quantity for demand modeling
    # Remove cancelled transactions and quantity <= 0
    df = df[~df['is_cancelled']]
    df = df[df['quantity'] > 0]

    # 4. Remove non-positive unit_price for revenue modeling
    df = df[df['unit_price'] > 0]

    # customer_id is kept nullable as per instructions

    print(f"[INGEST] Cleaned row count: {len(df)}")

    # Ensure output directory exists
    bronze_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Bronze Parquet - using microseconds to ensure Spark compatibility
    df.to_parquet(bronze_path, index=False, coerce_timestamps='us', allow_truncated_timestamps=True)
    print(f"[INGEST] Success: Online Retail II promoted to Bronze at {bronze_path}\n")

if __name__ == "__main__":
    ingest_online_retail()
