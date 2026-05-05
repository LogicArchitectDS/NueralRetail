import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
# MLflow tracking for experiment management

# Define paths
INPUT_FILE = 'data/features/churn_features.parquet'

def main():
    print("Loading features...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find the input file at {INPUT_FILE}")
        return

    print("Preprocessing data...")
    # CRITICAL: Drop Recency to prevent data leakage. Drop customer_unique_id.
    cols_to_drop = ['Recency', 'customer_unique_id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Define X (Features: Frequency, Monetary) and y (Target: is_churned)
    X = df[['Frequency', 'Monetary']]
    y = df['is_churned']

    # Split the data into 80% training and 20% testing sets
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    print(f"Class distribution before SMOTE - Negative: {(y_train==0).sum()}, Positive: {(y_train==1).sum()}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE - Negative: {(y_train_resampled==0).sum()}, Positive: {(y_train_resampled==1).sum()}")

    # Calculate scale_pos_weight for XGBoost (alternative/backup to SMOTE)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    print("Starting training model (MLflow tracking disabled)...")

    # Define model parameters with class imbalance handling
    params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'random_state': 42,
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,  # Built-in XGBoost class weighting
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Initialize and train the XGBClassifier on SMOTE-resampled data
    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)

    from sklearn.metrics import roc_auc_score, classification_report
    import mlflow

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n{'='*50}")
    print(f"POST-SMOTE AUC-ROC: {auc_roc:.4f}  (Target: ≥ 0.90)")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    with mlflow.start_run(run_name="xgboost_smote_final"):
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_metric("smote_applied", 1)
        mlflow.log_param("model_version", "xgboost_smote_v2")
        print("Metrics logged to MLflow run: xgboost_smote_final")

    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    import mlflow

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n{'='*50}")
    print(f"POST-SMOTE AUC-ROC: {auc_roc:.4f}  (Target: ≥ 0.90)")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    with mlflow.start_run(run_name="xgboost_smote_final"):
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_metric("smote_applied", 1)
        mlflow.log_param("model_version", "xgboost_smote_v2")
        print(f"Metrics logged to MLflow run: xgboost_smote_final")

    # Predict on the TEST data (original, not resampled)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    print("\nModel Training Complete!")
    print("=" * 50)
    print("EVALUATION METRICS (on held-out test set)")
    print("=" * 50)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1-Score:           {f1:.4f}")
    print(f"AUC-ROC:            {auc_roc:.4f}")
    print("=" * 50)
    print("IMBALANCE HANDLING APPLIED:")
    print(f"  - SMOTE resampling (k=5)")
    print(f"  - scale_pos_weight: {scale_pos_weight:.2f}")
    print("=" * 50)

    # Post-SMOTE evaluation block for Amdox F-04 compliance
    print(f"\n{'='*50}")
    print(f"POST-SMOTE AUC-ROC: {auc_roc:.4f}  (Target: ≥ 0.90)")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    # Log to MLflow
    with mlflow.start_run(run_name="xgboost_smote_final"):
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("smote_applied", 1)
        mlflow.log_param("model_version", "xgboost_smote_v2")
        mlflow.log_param("n_estimators", params['n_estimators'])
        mlflow.log_param("learning_rate", params['learning_rate'])
        mlflow.log_param("max_depth", params['max_depth'])
        print(f"Metrics logged to MLflow run: xgboost_smote_final")

    print("=" * 50)
    print("Saving model artifact for Phase 5...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_churn.pkl')
    print("Saved to models/xgboost_churn.pkl")

if __name__ == "__main__":
    main()