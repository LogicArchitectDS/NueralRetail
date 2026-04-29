import os
import pandas as pd

# Define input and output paths
CUSTOMERS_FILE = 'data/landing/olist/olist_customers_dataset.csv'
ORDERS_FILE = 'data/landing/olist/olist_orders_dataset.csv'
ITEMS_FILE = 'data/landing/olist/olist_order_items_dataset.csv'
OUTPUT_FILE = 'data/features/churn_features.parquet'

def main():
    print("Loading datasets...")
    try:
        customers = pd.read_csv(CUSTOMERS_FILE)
        orders = pd.read_csv(ORDERS_FILE)
        items = pd.read_csv(ITEMS_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure the CSV files are located in data/landing/olist/")
        return

    print("Merging datasets...")
    # Merge Customers to Orders using customer_id
    merged_df = customers.merge(orders, on='customer_id')

    # Merge the result to Order Items using order_id
    merged_df = merged_df.merge(items, on='order_id')

    print("Filtering data...")
    # Drop any orders that have a status of "canceled" or "unavailable"
    merged_df = merged_df[~merged_df['order_status'].isin(['canceled', 'unavailable'])].copy()

    # Convert order_purchase_timestamp to datetime for calculations
    merged_df['order_purchase_timestamp'] = pd.to_datetime(merged_df['order_purchase_timestamp'])

    # Determine "today's date" as the absolute maximum order_purchase_timestamp in the entire dataset
    today_date = merged_df['order_purchase_timestamp'].max()
    print(f"Calculated 'Today's Date' as: {today_date}")

    print("Calculating RFM features...")
    # Group the data by customer_unique_id
    rfm = merged_df.groupby('customer_unique_id').agg(
        last_order_date=('order_purchase_timestamp', 'max'),
        Frequency=('order_id', 'nunique'),
        Monetary=('price', 'sum')
    ).reset_index()

    # Calculate Recency: Number of days since their most recent order
    rfm['Recency'] = (today_date - rfm['last_order_date']).dt.days

    # Define the Target Variable (is_churned)
    # If Recency > 180 days -> 1, else -> 0
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
