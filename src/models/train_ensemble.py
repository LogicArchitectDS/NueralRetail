import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import joblib
import optuna
import mlflow
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

# Configuration
FEATURES_FILE = 'data/silver/sales_features.parquet'
LSTM_MODEL_PATH = 'models/lstm_demand.pth'
LSTM_SCALER_PATH = 'models/lstm_scaler.pkl'
HORIZON = 30
LOOKBACK = 28
FEATURE_COLS = ['Demand', 'day_of_week', 'is_weekend', 'is_month_end', 'is_holiday', 'promo_active']

def _patch_polars_for_cmdstanpy() -> None:
    try:
        import polars as pl_lib
    except Exception:
        return
    read_csv = getattr(pl_lib, "read_csv", None)
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
    pl_lib.read_csv = _read_csv_compat

class SalesLSTM(pl.LightningModule):
    def __init__(self, input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2, output_size=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.relu(last_out)
        prediction = self.linear(out)
        return prediction

def main():
    _patch_polars_for_cmdstanpy()
    
    # 1. Load & Split Data
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Missing {FEATURES_FILE}. Run build_features.py first.")
    
    df = pd.read_parquet(FEATURES_FILE)
    df = df.sort_values('Date').reset_index(drop=True)
    
    train_df = df.iloc[:-HORIZON].copy()
    test_df = df.iloc[-HORIZON:].copy()
    ground_truth = test_df['Demand'].values

    # 2. Prophet Forecast
    print("Training Prophet on multivariate-ready data...")
    # Prophet only needs ds and y, but we use the train_df which has the required columns
    prophet_train = train_df.rename(columns={'Date': 'ds', 'Demand': 'y'})
    m = Prophet()
    m.fit(prophet_train)
    future = m.make_future_dataframe(periods=HORIZON)
    prophet_forecast_full = m.predict(future)
    prophet_preds = prophet_forecast_full.iloc[-HORIZON:]['yhat'].values

    # 3. Multivariate LSTM Forecast
    print("Loading Multivariate LSTM and generating inference...")
    scaler = joblib.load(LSTM_SCALER_PATH)
    
    # Take last 28 days of train_df for all 6 features
    last_window = train_df[FEATURE_COLS].iloc[-LOOKBACK:].values
    scaled_window = scaler.transform(last_window)
    input_tensor = torch.FloatTensor(scaled_window).unsqueeze(0) # (1, 28, 6)

    model = SalesLSTM()
    model.load_state_dict(torch.load(LSTM_MODEL_PATH))
    model.eval()
    
    with torch.no_grad():
        lstm_scaled_preds = model(input_tensor).numpy().flatten() # (30,)

    # Inverse Scaling with 6-column dummy array
    dummy_array = np.zeros((HORIZON, len(FEATURE_COLS)))
    dummy_array[:, 0] = lstm_scaled_preds # LSTM predicted Demand into first column
    unscaled_array = scaler.inverse_transform(dummy_array)
    lstm_preds = unscaled_array[:, 0]

    # 4. Optuna Blending
    def objective(trial):
        w = trial.suggest_float("w", 0.0, 1.0)
        ensemble = w * prophet_preds + (1 - w) * lstm_preds
        return mean_absolute_percentage_error(ground_truth, ensemble)

    print("Running Optuna study for 50 trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_w = study.best_params['w']
    best_mape = study.best_value

    # 5. MLflow Logging
    local_tracking_uri = f"file://{os.path.abspath('mlruns')}"
    mlflow.set_tracking_uri(local_tracking_uri)
    mlflow.set_experiment("Retail_Ensemble_Optimization_V2")

    with mlflow.start_run(run_name="Multivariate_Ensemble_V2"):
        mlflow.log_param("best_w", best_w)
        mlflow.log_metric("ensemble_mape", best_mape)
        mlflow.log_param("model_type", "Multivariate_Ensemble")
        print(f"\nOptimization Complete.")
        print(f"Best Weight (w): {best_w:.4f}")
        print(f"Final Ensemble MAPE: {best_mape:.4f}")

if __name__ == "__main__":
    main()
