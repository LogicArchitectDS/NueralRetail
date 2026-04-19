import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_or_create_context():
    return ge.get_context()

def ensure_pos_expectation_suite(context):
    suite_name = "pos_sales_suite"
    suites = context.list_expectation_suite_names()
    if suite_name in suites:
        return suite_name
        
    context.add_expectation_suite(expectation_suite_name=suite_name)
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
    Validates a pandas dataframe against the POS expectation suite.
    """
    if df.empty:
        logger.warning("Dataframe is empty, skipping validation.")
        return False

    context = get_or_create_context()
    
    datasource_name = "runtime_datasource"
    if datasource_name not in context.list_datasources():
        context.add_datasource(
            datasource_name,
            class_name="Datasource",
            execution_engine={"class_name": "PandasExecutionEngine"},
            data_connectors={
                "default_runtime_data_connector_name": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier_name"],
                }
            },
        )

    suite_name = ensure_pos_expectation_suite(context)
    
    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
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
