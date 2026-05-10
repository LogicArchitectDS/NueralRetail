import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import mlflow

# Define paths
INPUT_FILE = 'data/features/churn_features.parquet'

def main():
    print("Loading features...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find the input file at {INPUT_FILE}")
        return

    print("Preprocessing data for tuned churn model...")
    # Based on user request, we include Recency and other customer-level features
    # Feature set expansion: Frequency, Monetary, Recency, avg_basket_value
    feature_cols = ['Frequency', 'Monetary', 'Recency', 'avg_basket_value']
    
    # Ensure all columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Using features: {feature_cols}")

    X = df[feature_cols]
    y = df['is_churned']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("Training Tuned XGBoost model...")
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'random_state': 42,
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight
    }

    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{'='*50}")
    print(f"Tuned AUC-ROC: {auc_roc:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save artifact
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_churn.pkl')
    print("Saved to models/xgboost_churn.pkl")

    # Log to MLflow if URI is set
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        with mlflow.start_run(run_name="xgboost_tuned"):
            mlflow.log_metrics({
                "auc_roc": auc_roc,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            })

if __name__ == "__main__":
    main()
