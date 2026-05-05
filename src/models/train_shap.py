import os
import pandas as pd
import joblib
import shap

def main():
    print("Loading the trained XGBoost model...")
    # Load the trained model
    model_path = "models/xgboost_churn.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    
    print("Loading the training features sample...")
    # Load data
    data_path = "data/features/churn_features.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_parquet(data_path)
    
    # Extract features matching the model schema
    features = ['Frequency', 'Monetary']
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Required column '{feature}' not found in the dataset.")
            
    # Sample a small portion to fit/initialize the explainer (optional, but good practice for speed)
    X = df[features].sample(min(100, len(df)), random_state=42) if len(df) > 100 else df[features]
    
    print("Initializing SHAP TreeExplainer...")
    # Initialize TreeExplainer
    # Passing data X is optional for tree explainers unless you need interventional feature perturbation,
    # but providing it is generally safe and sometimes required depending on the exact SHAP version and model.
    explainer = shap.TreeExplainer(model)
    
    print("Saving the SHAP explainer artifact...")
    # Serialization
    os.makedirs("models", exist_ok=True)
    output_path = "models/shap_explainer.pkl"
    joblib.dump(explainer, output_path)
    
    print(f"Success! SHAP TreeExplainer generated and saved to {output_path}.")

if __name__ == "__main__":
    main()
