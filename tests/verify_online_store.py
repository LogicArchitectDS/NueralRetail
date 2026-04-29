from feast import FeatureStore
import pandas as pd
from pathlib import Path

def verify():
    # Point directly to the Feast repository folder
    repo_path = Path(__file__).resolve().parent.parent / "src" / "features" / "feature_repo"
    
    # Initialize the Feature Store
    store = FeatureStore(repo_path=str(repo_path))

    print("Querying Redis Online Store for Store 'ST_001'...")
    
    # Ask Feast to fetch the real-time features for our model
    feature_vector = store.get_online_features(
        features=[
            "store_demand_lags:daily_revenue",
            "store_demand_lags:order_count",
            "store_demand_lags:rev_lag_1d",
            "store_demand_lags:rev_lag_7d"
        ],
        entity_rows=[{"store_id": "ST_001"}]
    ).to_df()
    
    print("\n=== Redis Response ===")
    print(feature_vector.to_string(index=False))
    print("======================\n")
    
    if feature_vector["daily_revenue"].isnull().all():
        print("[FAIL] Redis returned empty data. Materialization failed.")
    else:
        print("[SUCCESS] Phase 2 is 100% verified! The ML serving layer is online.")

if __name__ == "__main__":
    verify()