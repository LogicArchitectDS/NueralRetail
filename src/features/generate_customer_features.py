import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
INPUT_FILE = 'data/bronze/transactions.parquet'
CHURN_OUTPUT = 'data/features/churn_features.parquet'
RFM_OUTPUT = 'data/features/rfm_features.parquet'
ARTIFACT_RFM = 'artifacts/rfm_features.parquet'
KPI_SUMMARY = 'artifacts/kpi_summary.json'

def main():
    print(f"Loading canonical dataset from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run ingestion first.")
        return

    df = pd.read_parquet(INPUT_FILE)

    # 1. Filter data for customer-level modeling
    df = df.dropna(subset=['customer_id'])
    
    if 'is_cancelled' in df.columns:
        df = df[~df['is_cancelled']]

    # 2. Calculate Customer-level features
    print("Calculating Customer-level features...")
    
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    today_date = df['invoice_date'].max()
    print(f"Reference Date (Today): {today_date}")

    # Aggregate core RFM + Basket Value + Country
    # We take the mode for country (most common)
    agg_dict = {
        'invoice_date': 'max',
        'invoice_no': 'nunique',
        'revenue': 'sum',
        'quantity': 'sum'
    }
    
    if 'country' in df.columns:
        agg_dict['country'] = lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"

    customer_df = df.groupby('customer_id').agg(agg_dict).reset_index()

    customer_df.rename(columns={
        'invoice_date': 'last_purchase',
        'invoice_no': 'Frequency',
        'revenue': 'Monetary',
        'quantity': 'TotalQuantity'
    }, inplace=True)

    # Derived Features
    customer_df['Recency'] = (today_date - customer_df['last_purchase']).dt.days
    customer_df['avg_basket_value'] = (customer_df['Monetary'] / customer_df['Frequency']).round(2)
    
    # 3. is_churned heuristic
    CHURN_THRESHOLD = 180
    customer_df['is_churned'] = (customer_df['Recency'] > CHURN_THRESHOLD).astype(int)

    # 4. Compatibility: Ensure both Case variants
    customer_df['frequency'] = customer_df['Frequency']
    customer_df['monetary'] = customer_df['Monetary']
    customer_df['recency'] = customer_df['Recency']
    customer_df['churn_probability'] = customer_df['is_churned'].astype(float)

    # 5. Save outputs
    os.makedirs(os.path.dirname(CHURN_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(ARTIFACT_RFM), exist_ok=True)

    print(f"Saving to {CHURN_OUTPUT}...")
    customer_df.to_parquet(CHURN_OUTPUT, index=False)
    print(f"Saving to {ARTIFACT_RFM}...")
    customer_df.to_parquet(ARTIFACT_RFM, index=False)

    print("\nFeature Engineering Complete.")
    print(f"Total Customers: {len(customer_df)}")
    print(f"Columns: {customer_df.columns.tolist()}")
    print("-" * 30)

if __name__ == "__main__":
    main()
