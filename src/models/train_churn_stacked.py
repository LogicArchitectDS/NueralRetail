import os
import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import shap

# Define paths
INPUT_FILE = 'data/features/churn_features.parquet'
MODEL_PATH = 'models/stacked_churn_model.pkl'

def main():
    print("Loading churn features for stacked ensemble...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # Data Preparation
    cols_to_drop = ['Recency', 'customer_unique_id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    X = df[['Frequency', 'Monetary']]
    y = df['is_churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE for imbalance handling
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Base Learners
    print("Initializing base learners (XGBoost + HistGradientBoosting)...")
    base_learners = [
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42))
    ]

    # Stacking Meta-Learner
    print("Building Stacking Ensemble...")
    stacked_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        cv=5
    )

    # Configure MLflow to use local tracking
    mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        with mlflow.start_run(run_name="churn_stacked_ensemble"):
            print("Training Stacked Model...")
            stacked_model.fit(X_train_res, y_train_res)
            
            y_pred_proba = stacked_model.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"\nStacked Ensemble AUC-ROC: {auc_roc:.4f}")
            print(classification_report(y_test, stacked_model.predict(X_test)))

            # Log Metrics
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_param("ensemble_type", "stacking")
            mlflow.log_param("base_learners", "xgb, hgb")
            
            # Save Model
            os.makedirs('models', exist_ok=True)
            joblib.dump(stacked_model, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

            # SHAP Explainability (F-04 requirement)
            print("Generating SHAP summary for meta-learner...")
            X_summary = shap.sample(X_test, 50)
            explainer = shap.KernelExplainer(stacked_model.predict_proba, X_summary)
            joblib.dump(explainer, 'models/stacked_shap_explainer.pkl')
    except Exception as e:
        print(f"Training or MLflow error: {e}")
        print("Falling back to training without MLflow...")
        stacked_model.fit(X_train_res, y_train_res)
        os.makedirs('models', exist_ok=True)
        joblib.dump(stacked_model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH} (Fallback)")

if __name__ == "__main__":
    main()
