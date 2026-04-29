import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = "data/landing/olist/olist_orders_dataset.csv"
OUTPUT_FILE = "data/silver/sales_features.parquet"

def main():
    print(f"Loading raw Olist data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Source file not found: {INPUT_FILE}")

    # 1. Data Ingestion & Base Aggregation
    df_raw = pd.read_csv(INPUT_FILE)
    df_raw['order_purchase_timestamp'] = pd.to_datetime(df_raw['order_purchase_timestamp'])
    
    # Group by Date and calculate daily total transaction count (Demand)
    df = df_raw.groupby(df_raw['order_purchase_timestamp'].dt.date).size().reset_index()
    df.columns = ['Date', 'Demand']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print("Engineering temporal features...")
    # 2. Temporal Features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)

    print("Engineering holiday and promotion regressors...")
    # 3. Holiday Regressor (is_holiday)
    # Mapping specific historical dates and some simulated major spikes
    holidays = [
        '2016-11-25', '2017-11-24', # Black Friday
        '2016-12-25', '2017-12-25', # Christmas
        '2017-01-01', '2018-01-01', # New Year
        '2017-05-01', '2018-05-01', # Labor Day
        '2017-09-07', '2018-09-07'  # Independence Day (Brazil)
    ]
    df['is_holiday'] = df['Date'].dt.date.astype(str).isin(holidays).astype(int)

    # 4. Promotion Regressor (promo_active)
    # Simulate promotional campaigns: 3-day windows at the start of every quarter
    def check_promo(date):
        # Jan 1-3, Apr 1-3, Jul 1-3, Oct 1-3
        return 1 if (date.month in [1, 4, 7, 10] and date.day in [1, 2, 3]) else 0

    df['promo_active'] = df['Date'].apply(check_promo)

    # 5. Save Enriched Data
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    # 6. Telemetry
    print("\nFeature Engineering Complete. Enriched Dataframe Head:")
    print(df.head())
    print("-" * 30)
    print(f"Success: Exported Parquet file to {OUTPUT_FILE}")
    print(f"Final Shape: {df.shape}")

if __name__ == "__main__":
    main()
