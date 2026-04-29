from feast import FeatureStore
import pandas as pd
from datetime import datetime
import os

def debug_feast():
    repo_path = "src/features/feature_repo"
    store = FeatureStore(repo_path=repo_path)

    # Read source data
    rfm_source_path = "data/features/rfm"
    source_df = pd.read_parquet(rfm_source_path)
    print("Source Data Head:")
    print(source_df.head(2))
    
    test_id = source_df.iloc[0]["customer_unique_id"]
    test_ts = source_df.iloc[0]["event_timestamp"]
    
    print(f"\nTesting with ID: {test_id} and TS: {test_ts}")

    # Entity data for testing (using EXACT values from source)
    entity_df = source_df.iloc[[0]][["customer_unique_id", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = datetime(2025, 1, 1) # Still use a later date


    # Get historical features
    job = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_rfm_features:recency",
            "customer_rfm_features:frequency",
            "customer_rfm_features:monetary",
        ],
    )
    training_df = job.to_df()

    print("\nRetrieved Features:")
    print(training_df)

if __name__ == "__main__":
    debug_feast()
