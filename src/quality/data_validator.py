import pandas as pd
import great_expectations as ge
import os

class DataQualityEngine:
    """
    Validation engine for NeuralRetail datasets using Great Expectations.
    """
    
    def validate_churn_features(self, file_path="data/features/churn_features.parquet"):
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "success": False}
        
        df = pd.read_parquet(file_path)
        ge_df = ge.from_pandas(df)
        
        results = []
        
        # 1. Column Presence
        results.append(ge_df.expect_column_to_exist("customer_unique_id"))
        results.append(ge_df.expect_column_to_exist("Frequency"))
        results.append(ge_df.expect_column_to_exist("Monetary"))
        results.append(ge_df.expect_column_to_exist("is_churned"))
        
        # 2. Null Checks
        results.append(ge_df.expect_column_values_to_not_be_null("Frequency"))
        results.append(ge_df.expect_column_values_to_not_be_null("Monetary"))
        
        # 3. Range Checks
        results.append(ge_df.expect_column_values_to_be_between("Frequency", min_value=1))
        results.append(ge_df.expect_column_values_to_be_between("Monetary", min_value=0))
        results.append(ge_df.expect_column_values_to_be_in_set("is_churned", [0, 1]))
        
        # Summarize
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)
        
        return {
            "dataset": "churn_features",
            "timestamp": pd.Timestamp.now().isoformat(),
            "success": all(r.success for r in results),
            "passed_tests": success_count,
            "total_tests": total_count,
            "details": [
                {
                    "expectation": r.expectation_config.expectation_type,
                    "success": r.success,
                    "column": r.expectation_config.kwargs.get("column")
                } for r in results
            ]
        }
