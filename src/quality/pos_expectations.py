import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
import pandas as pd
import os

def create_pos_expectation_suite(context):
    suite_name = "pos_sales_suite"
    context.create_expectation_suite(suite_name=suite_name, overwrite_existing=True)
    
    # Define expectations
    suite = context.get_expectation_suite(suite_name)
    
    # 1. Row count should be > 0
    suite.add_expectation(ge.core.ExpectationConfiguration(
        expectation_type="expect_table_row_count_to_be_between",
        kwargs={"min_value": 1}
    ))
    
    # 2. transaction_id must be unique and not null
    suite.add_expectation(ge.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "transaction_id"}
    ))
    
    # 3. amount (price) should be within reasonable range
    suite.add_expectation(ge.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "amount", "min_value": 0, "max_value": 100000}
    ))
    
    context.save_expectation_suite(suite)
    return suite_name

def validate_pos_data(df: pd.DataFrame):
    """
    Validates a pandas/spark dataframe against the POS expectation suite.
    For simulation, we use the Pandas-based validator within the Spark job or as a separate step.
    """
    context = ge.get_context()
    suite_name = create_pos_expectation_suite(context)
    
    batch_request = RuntimeBatchRequest(
        datasource_name="my_datasource", # This would be configured in GE config
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="pos_sales",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_id"}
    )
    
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )
    
    results = validator.validate()
    return results.success
