from feast import FeatureStore
import pandas as pd
from datetime import datetime
import os

def verify_feast():
    repo_path = os.path.join(os.path.dirname(__file__), "../src/features/feature_repo")
    store = FeatureStore(repo_path=repo_path)

    # 1. Verify RFM
    entity_df = pd.DataFrame.from_dict(
        {
            "customer_unique_id": [
                "e533c359f12f908382cf31d51ef1e684",
                "f8d62e8ff724bf75c20ffd5faf7b9e30",
            ],
            "event_timestamp": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
            ],
        }
    )

    print("Fetching RFM features...")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_rfm_features:recency",
            "customer_rfm_features:frequency",
            "customer_rfm_features:monetary",
        ],
    ).to_df()

    print(f"RFM Results shape: {training_df.shape}")
    if not training_df.empty:
        print(training_df.head())
    else:
        print("RFM results are EMPTY")

    # 2. Verify Demand
    entity_df_demand = pd.DataFrame.from_dict(
        {
            "store_id": ["ST_001"],
            "event_timestamp": [datetime(2025, 1, 1)],
        }
    )

    print("\nFetching Demand features...")
    training_df_demand = store.get_historical_features(
        entity_df=entity_df_demand,
        features=[
            "store_demand_features:daily_revenue",
            "store_demand_features:order_count",
        ],
    ).to_df()

    print(f"Demand Results shape: {training_df_demand.shape}")
    if not training_df_demand.empty:
        print(training_df_demand.head())
    else:
        print("Demand results are EMPTY")

if __name__ == "__main__":
    verify_feast()
