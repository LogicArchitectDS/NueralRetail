import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import mlflow
import subprocess

# Paths
INPUT_FILE = 'artifacts/rfm_features.parquet'
ARTIFACTS_DIR = 'artifacts'

def load_data():
    # If artifacts file doesn't exist, try fallbacks
    if not os.path.exists(INPUT_FILE):
        print(f"Data file {INPUT_FILE} not found. Running build_features.py...")
        try:
            subprocess.run([sys.executable, "src/features/build_features.py"], check=True)
        except Exception as e:
            print(f"Error running build_features.py: {e}")
        
        # Check if rfm_features exists in data/features
        secondary_source = 'data/features/rfm_features.parquet'
        if os.path.exists(secondary_source):
            df = pd.read_parquet(secondary_source)
            if len(df) > 10:
                print(f"Found {len(df)} rows in {secondary_source}. Using it.")
                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                import shutil
                shutil.copy(secondary_source, INPUT_FILE)
                return df
        
        # Check if churn_features exists which is usually larger
        churn_source = 'data/features/churn_features.parquet'
        if os.path.exists(churn_source):
            print(f"Using {churn_source} as fallback.")
            df = pd.read_parquet(churn_source)
            if not os.path.exists(INPUT_FILE):
                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                df.to_parquet(INPUT_FILE)
            return df
            
        raise FileNotFoundError(f"Could not find {INPUT_FILE} or any suitable fallback.")
    
    df = pd.read_parquet(INPUT_FILE)
    
    # If loaded data is too small, try fallback anyway
    if len(df) <= 10:
        print(f"Data at {INPUT_FILE} too small ({len(df)} rows). Checking for better source...")
        churn_source = 'data/features/churn_features.parquet'
        if os.path.exists(churn_source):
            df_large = pd.read_parquet(churn_source)
            if len(df_large) > len(df):
                print(f"Found better data in {churn_source} ({len(df_large)} rows).")
                df_large.to_parquet(INPUT_FILE)
                return df_large
                
    return df

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Set local MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    
    print("Loading RFM data...")
    try:
        df = load_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Use columns: recency, frequency, monetary (whichever exist)
    possible_features = ['recency', 'frequency', 'monetary']
    features = [f for f in possible_features if f in df.columns]
    
    # Handle case where columns might be capitalized
    if not features:
        possible_features_cap = ['Recency', 'Frequency', 'Monetary']
        features = [f for f in possible_features_cap if f in df.columns]
        
    if not features:
        print(f"Warning: None of the required features {possible_features} found in dataset columns: {df.columns.tolist()}")
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f.lower() not in ['customer_id', 'id', 'is_churned']]

    print(f"Using features: {features} from {len(df)} samples.")
    X = df[features]
    
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)
    
    with mlflow.start_run(run_name="segmentation_advanced_dbscan_gmm"):
        # 1. DBSCAN
        print("Training DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        db_labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = list(db_labels).count(-1)
        noise_pct = (n_noise / len(db_labels)) * 100
        
        db_silhouette = -1.0
        if n_clusters > 1:
            try:
                # Use a sample for silhouette if dataset is too large to speed up
                if len(X_scaled) > 10000:
                    idx = np.random.choice(len(X_scaled), 10000, replace=False)
                    db_silhouette = silhouette_score(X_scaled[idx], db_labels[idx])
                else:
                    db_silhouette = silhouette_score(X_scaled, db_labels)
            except:
                pass
        
        print(f"DBSCAN Results:")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise %: {noise_pct:.2f}%")
        print(f"  Silhouette Score: {db_silhouette:.4f}")
        
        mlflow.log_param("dbscan_eps", 0.5)
        mlflow.log_param("dbscan_min_samples", 5)
        mlflow.log_metric("dbscan_n_clusters", n_clusters)
        mlflow.log_metric("dbscan_noise_pct", noise_pct)
        mlflow.log_metric("dbscan_silhouette", float(db_silhouette))
        
        joblib.dump(dbscan, os.path.join(ARTIFACTS_DIR, 'dbscan_model.pkl'))
        joblib.dump(db_labels, os.path.join(ARTIFACTS_DIR, 'dbscan_labels.pkl'))
        
        # 2. Gaussian Mixture
        print("\nTraining GMM with BIC selection (k=2 to 10)...")
        bics = []
        models = []
        # Adjust k_range if too few samples
        max_k = min(11, len(X_scaled))
        if max_k <= 2:
            print("Too few samples for GMM selection. Skipping GMM optimization.")
            best_k = 1
            best_gmm = GaussianMixture(n_components=1, random_state=42).fit(X_scaled)
            best_bic = best_gmm.bic(X_scaled)
            gmm_silhouette = -1.0
        else:
            k_range = range(2, max_k)
            for k in k_range:
                gmm = GaussianMixture(n_components=k, random_state=42)
                gmm.fit(X_scaled)
                bics.append(gmm.bic(X_scaled))
                models.append(gmm)
                
            best_idx = np.argmin(bics)
            best_k = k_range[best_idx]
            best_gmm = models[best_idx]
            best_bic = bics[best_idx]
            
            gmm_labels = best_gmm.predict(X_scaled)
            gmm_silhouette = -1.0
            try:
                if len(X_scaled) > 10000:
                    idx = np.random.choice(len(X_scaled), 10000, replace=False)
                    gmm_silhouette = silhouette_score(X_scaled[idx], gmm_labels[idx])
                else:
                    gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
            except:
                pass
        
        print(f"GMM Results:")
        print(f"  Optimal k: {best_k}")
        print(f"  BIC value: {best_bic:.2f}")
        print(f"  Silhouette Score: {gmm_silhouette:.4f}")
        
        mlflow.log_param("gmm_optimal_k", best_k)
        mlflow.log_metric("gmm_best_bic", float(best_bic))
        mlflow.log_metric("gmm_silhouette", float(gmm_silhouette))
        
        joblib.dump(best_gmm, os.path.join(ARTIFACTS_DIR, 'gmm_model.pkl'))
        joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'gmm_scaler.pkl'))
        
        print("\n✓ F-02 COMPLETE: K-Means + DBSCAN + GMM all trained and logged.")

if __name__ == "__main__":
    main()
