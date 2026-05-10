import os
import pandas as pd
import mlflow
import mlflow.prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from pathlib import Path  


def _patch_polars_for_cmdstanpy() -> None:
    """
    Bridge cmdstanpy's newer Polars API usage with the project's pinned Polars version.
    """
    try:
        import polars as pl
    except Exception:
        return

    read_csv = getattr(pl, "read_csv", None)
    if read_csv is None or getattr(read_csv, "_cmdstanpy_compat", False):
        return

    def _read_csv_compat(*args, **kwargs):
        if "schema_overrides" in kwargs and "dtypes" not in kwargs:
            kwargs["dtypes"] = kwargs.pop("schema_overrides")
        if "infer_schema" in kwargs and "infer_schema_length" not in kwargs:
            infer_schema = kwargs.pop("infer_schema")
            kwargs["infer_schema_length"] = None if infer_schema else 0
        return read_csv(*args, **kwargs)

    _read_csv_compat._cmdstanpy_compat = True
    pl.read_csv = _read_csv_compat


def _load_training_frame(candidate_paths: list[str]) -> tuple[pd.DataFrame, str]:
    """Pick the first dataset with enough rows for Prophet training."""
    last_error = None

    for path in candidate_paths:
        print(f"Reading training data from {path}...")
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            print(f"Unable to read {path}: {exc}")
            last_error = exc
            continue

        print(f"Loaded {len(df)} raw rows from {path}.")
        if len(df) >= 2:
            return df, path

        print(f"Skipping {path}: Prophet needs at least 2 rows, found {len(df)}.")

    if last_error is not None:
        raise RuntimeError("Failed to load a usable training dataset.") from last_error

    raise RuntimeError(
        "No usable training dataset found. Generate more demand history before training Prophet."
    )


def _safe_mlflow_log_artifact(path: str) -> None:
    try:
        mlflow.log_artifact(path)
    except Exception as exc:
        print(f"Warning: failed to log artifact '{path}' to MLflow: {exc}")


def _safe_mlflow_log_model(model: Prophet, artifact_path: str) -> None:
    try:
        mlflow.prophet.log_model(model, artifact_path=artifact_path)
    except Exception as exc:
        print(f"Warning: failed to log Prophet model to MLflow: {exc}")


def main():
    print("Starting Demand Forecasting Training Pipeline...")
    
    # Configuration
    candidate_data_paths = [
        "data/features/demand_features.parquet",
        "data/features/demand_single.parquet",
    ]
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
    
    print(f"Setting MLflow tracking URI to {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    _patch_polars_for_cmdstanpy()
    
    # Ensure the temporary artifact directory exists
    os.makedirs("tmp_artifacts", exist_ok=True)
    
    try:
        df, data_path = _load_training_frame(candidate_data_paths)
    except Exception as e:
        print(f"Error reading data: {e}")
        return
    
    print("Preparing data...")
    # 1. Rename columns to Prophet's strict standards
    # We force the rename assuming the Spark pipeline outputted 'date' and 'daily_revenue'
    df = df.rename(columns={'date': 'ds', 'daily_revenue': 'y'})

    # 2. SANITIZE: Force the date column into a strict Pandas Datetime format (CRITICAL)
    df['ds'] = pd.to_datetime(df['ds'])

    # 3. SANITIZE: Fill missing lags with 0.0 instead of deleting the rows
    df['rev_lag_1d'] = df['rev_lag_1d'].fillna(0.0)
    df['rev_lag_7d'] = df['rev_lag_7d'].fillna(0.0)
    df['order_count'] = df['order_count'].fillna(0)

    # 4. SANITIZE: Drop ONLY rows where the actual target variable (y) or date (ds) is missing
    df = df.dropna(subset=['y', 'ds'])

    print("Configuring local MLflow Experiment routing...")
    # 1. Define a safe, local path inside your project folder for artifacts
    local_artifact_dir = Path(os.getcwd()) / "mlruns"
    local_artifact_dir.mkdir(exist_ok=True)

    # 2. Force MLflow to use this new experiment and safe path
    experiment_name = "Retail_Demand_Prophet"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=local_artifact_dir.as_uri()
        )
    mlflow.set_experiment(experiment_name)

    print(f"Starting MLflow run in experiment: '{experiment_name}'...")
    print(f"Starting MLflow run in experiment: '{experiment_name}'...")
    with mlflow.start_run(run_name="Prophet_Baseline_V1"):
        
        seasonality_mode = 'multiplicative'
        print(f"Initializing Prophet model (seasonality_mode={seasonality_mode})...")
        m = Prophet(seasonality_mode=seasonality_mode)
        
        print("Adding regressors to the model...")
        regressors = ['rev_lag_1d', 'rev_lag_7d', 'order_count'] 
        for reg in regressors:
            if reg in df.columns:
                m.add_regressor(reg)
                
        print("Fitting the model on historical data...")
        m.fit(df)
        
        print("Logging model parameters to MLflow...")
        mlflow.log_param("seasonality_mode", seasonality_mode)
        mlflow.log_param("regressors", regressors)
        
        print("Running native Prophet cross-validation...")
        try:
            # Using a 30-day horizon for cross-validation
            df_cv = cross_validation(m, horizon='30 days')
            
            print("Calculating performance metrics (MAE, RMSE)...")
            df_p = performance_metrics(df_cv)
            
            # Aggregate the metrics over the horizon (using the mean)
            mae = df_p['mae'].mean()
            rmse = df_p['rmse'].mean()
            
            print(f"Cross-validation MAE: {mae:.4f}")
            print(f"Cross-validation RMSE: {rmse:.4f}")
            
            print("Logging cross-validation metrics to MLflow...")
            mlflow.log_metric("cv_mae", mae)
            mlflow.log_metric("cv_rmse", rmse)
            
        except Exception as e:
            print(f"Cross-validation failed (possibly due to insufficient data for the horizon): {e}")
            
        print("Generating forecast on historical data for plotting...")
        # Since we have external regressors, predict requires them to be present. 
        # Using the historical dataframe provides the known regressors to plot the fit.
        forecast = m.predict(df)
        
        print("Plotting forecast...")
        fig1 = m.plot(forecast)
        forecast_plot_path = "tmp_artifacts/forecast.png"
        fig1.savefig(forecast_plot_path)
        plt.close(fig1)
        
        print("Plotting forecast components (seasonality)...")
        fig2 = m.plot_components(forecast)
        components_plot_path = "tmp_artifacts/components.png"
        fig2.savefig(components_plot_path)
        plt.close(fig2)
        
        print("Logging artifact plots to MLflow...")
        _safe_mlflow_log_artifact(forecast_plot_path)
        _safe_mlflow_log_artifact(components_plot_path)
        
        print("Logging Prophet model to MLflow...")
        _safe_mlflow_log_model(m, artifact_path="prophet_model")
        
        print("Cleaning up temporary artifact files...")
        if os.path.exists(forecast_plot_path):
            os.remove(forecast_plot_path)
        if os.path.exists(components_plot_path):
            os.remove(components_plot_path)
        if os.path.exists("tmp_artifacts"):
            os.rmdir("tmp_artifacts")
        
        print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
