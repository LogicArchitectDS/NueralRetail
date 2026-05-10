import os
import pandas as pd

# Updated to use the canonical Bronze source
INPUT_FILE = 'data/bronze/transactions.parquet'
OUTPUT_FILE = 'data/features/churn_features.parquet'

def main():
    print(f"Loading canonical dataset from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run ingestion first.")
        return

    df = pd.read_parquet(INPUT_FILE)

    print("Filtering data for churn analysis...")
    # For churn, we need customer identification
    df = df.dropna(subset=['customer_id'])
    
    # Convert invoice_date to datetime if not already
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])

    # Determine "today's date" as the absolute maximum invoice_date in the entire dataset
    today_date = df['invoice_date'].max()
    print(f"Calculated 'Today's Date' as: {today_date}")

    print("Calculating RFM features...")
    # Group by customer_id (canonical identifier in Online Retail II)
    rfm = df.groupby('customer_id').agg(
        last_order_date=('invoice_date', 'max'),
        Frequency=('invoice_no', 'nunique'),
        Monetary=('revenue', 'sum')
    ).reset_index()

    # Calculate Recency: Number of days since their most recent order
    rfm['Recency'] = (today_date - rfm['last_order_date']).dt.days

    # Define the Target Variable (is_churned)
    # Using 180 days as the threshold, consistent with previous implementation
    rfm['is_churned'] = (rfm['Recency'] > 180).astype(int)

    # Drop the temporary column used for Recency calculation
    rfm.drop(columns=['last_order_date'], inplace=True)

    print("-" * 40)
    print(f"Final Dataframe Shape: {rfm.shape}")
    print("\nClass Balance of 'is_churned':")
    print(rfm['is_churned'].value_counts())
    print("-" * 40)

    # Save the final dataframe as a parquet file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Saving Parquet file to '{OUTPUT_FILE}'...")
    rfm.to_parquet(OUTPUT_FILE, index=False)
    print("Saved successfully!")

if __name__ == "__main__":
    main()
