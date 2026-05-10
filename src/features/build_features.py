import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
INPUT_FILE = "data/bronze/transactions.parquet"
SILVER_OUTPUT = "data/silver/sales_features.parquet"
DEMAND_OUTPUT = "data/features/demand_features.parquet"

def main():
    print(f"Loading canonical transactional data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Ensure ingestion is complete.")
        return

    # 1. Load data
    df_raw = pd.read_parquet(INPUT_FILE)
    df_raw['invoice_date'] = pd.to_datetime(df_raw['invoice_date'])
    
    # 2. Aggregation by Day
    print("Aggregating daily demand and revenue metrics...")
    # Group by Date
    daily_df = df_raw.groupby(df_raw['invoice_date'].dt.date).agg(
        demand_units=('quantity', 'sum'),
        daily_revenue=('revenue', 'sum'),
        order_count=('invoice_no', 'nunique'),
        avg_unit_price=('unit_price', 'mean')
    ).reset_index()
    
    daily_df.rename(columns={'invoice_date': 'date'}, inplace=True)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    # 3. Temporal Features
    print("Engineering temporal features...")
    daily_df['Date'] = daily_df['date'] # Alias for LSTM compatibility
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    daily_df['is_weekend'] = daily_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    daily_df['is_month_end'] = daily_df['date'].dt.is_month_end.astype(int)

    # 4. Holiday & Promotion Regressors (2009-2011 period for Online Retail II)
    print("Engineering holiday and promotion regressors...")
    holidays = [
        '2009-12-25', '2010-12-25', '2011-12-25', # Christmas
        '2010-01-01', '2011-01-01',              # New Year
        '2010-11-26', '2011-11-25',              # Black Friday
        '2010-04-05', '2011-04-25',              # Easter Monday (UK/General)
    ]
    daily_df['is_holiday'] = daily_df['date'].dt.date.astype(str).isin(holidays).astype(int)

    def check_promo(date):
        # 3-day windows at start of quarter
        return 1 if (date.month in [1, 4, 7, 10] and date.day in [1, 2, 3]) else 0
    daily_df['promo_active'] = daily_df['date'].apply(check_promo)

    # 5. Demand Alias for LSTM Compatibility (using demand_units as target)
    daily_df['Demand'] = daily_df['demand_units']

    # 6. Lag Features for Prophet Compatibility
    print("Calculating demand lags...")
    daily_df['rev_lag_1d'] = daily_df['daily_revenue'].shift(1).fillna(0)
    daily_df['rev_lag_7d'] = daily_df['daily_revenue'].shift(7).fillna(0)

    # 7. Save Outputs
    os.makedirs(os.path.dirname(SILVER_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(DEMAND_OUTPUT), exist_ok=True)
    
    daily_df.to_parquet(SILVER_OUTPUT, index=False)
    daily_df.to_parquet(DEMAND_OUTPUT, index=False)

    # 8. Telemetry
    print("-" * 30)
    print(f"Demand Series Generation Complete.")
    print(f"Date Range : {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Row Count  : {len(daily_df)}")
    print(f"Columns    : {daily_df.columns.tolist()}")
    print("-" * 30)
    print("Preview (First 5 Rows):")
    print(daily_df.head())
    print("-" * 30)
    print(f"Success: Exported to {SILVER_OUTPUT} and {DEMAND_OUTPUT}")

if __name__ == "__main__":
    main()
