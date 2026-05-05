import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Configuration
FEATURES_FILE = 'data/silver/sales_features.parquet'
SCALER_PATH = 'models/lstm_scaler.pkl'
MODEL_PATH = 'models/lstm_demand.pth'
LOOKBACK = 28
HORIZON = 30
BATCH_SIZE = 32
MAX_EPOCHS = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FEATURE_COLS = ['Demand', 'day_of_week', 'is_weekend', 'is_month_end', 'is_holiday', 'promo_active']

class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback, horizon):
        self.X, self.y = self.create_sequences(data, lookback, horizon)

    def create_sequences(self, data, lookback, horizon):
        X, y = [], []
        # data shape: (N, 6)
        # Target 'Demand' is at index 0
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback]) # (28, 6)
            y.append(data[i + lookback : i + lookback + horizon, 0]) # (30,)
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SalesLSTM(pl.LightningModule):
    def __init__(self, input_size=len(FEATURE_COLS), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.final_loss = 0.0

    def forward(self, x):
        # x shape: (batch, lookback, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the last hidden state to project to the 30-day horizon
        last_out = lstm_out[:, -1, :]
        out = self.relu(last_out)
        prediction = self.linear(out)
        return prediction

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.final_loss = self.trainer.callback_metrics.get("train_loss", torch.tensor(0.0)).item()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

def main():
    # 1. Data Loading
    print(f"Loading enriched multivariate data from {FEATURES_FILE}...")
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Feature file not found at {FEATURES_FILE}. Run build_features.py first.")
    
    df = pd.read_parquet(FEATURES_FILE)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Scaling
    print(f"Scaling multivariate features: {FEATURE_COLS}")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[FEATURE_COLS].values)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Multivariate scaler saved to {SCALER_PATH}")

    # 3. Sequence Generation & DataLoader
    dataset = TimeSeriesDataset(scaled_data, LOOKBACK, HORIZON)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Generated {len(dataset)} sequences with Lookback={LOOKBACK} and Horizon={HORIZON}.")

    # 4. Model & Training
    model = SalesLSTM()
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        enable_checkpointing=False,
        logger=False
    )

    trainer.fit(model, train_loader)

    # 5. Serialization
    torch.save(model.state_dict(), MODEL_PATH)
    
    print("\nMultivariate LSTM trained and saved successfully")
    print(f"Final Epoch Loss: {model.final_loss:.6f}")

if __name__ == "__main__":
    main()
