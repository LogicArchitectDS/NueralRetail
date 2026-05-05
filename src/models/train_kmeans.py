import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

def main():
    print("Loading data...")
    # Load data
    data_path = "data/features/churn_features.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_parquet(data_path)
    
    # Extract features
    features = ['Frequency', 'Monetary']
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Required column '{feature}' not found in the dataset.")
            
    X = df[features]
    
    print("Preprocessing data...")
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hyperparameter tuning (Amdox F-02 requirement)
    best_k = None
    best_db_score = float('inf')
    best_model = None
    best_silhouette = -1
    
    print("Starting hyperparameter tuning (k=6 to 10)...")
    for k in range(6, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        sil_score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        
        print(f"k={k}: Silhouette Score = {sil_score:.4f}, Davies-Bouldin Index = {db_score:.4f}")
        
        # Select k with lowest Davies-Bouldin Index
        if db_score < best_db_score:
            best_db_score = db_score
            best_k = k
            best_model = kmeans
            best_silhouette = sil_score
            
    print(f"\nOptimal k selected: {best_k}")
    print(f"Lowest Davies-Bouldin Index: {best_db_score:.4f}")
    print(f"Corresponding Silhouette Score: {best_silhouette:.4f}\n")
    
    print("Saving artifacts...")
    # Serialization
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/kmeans_scaler.pkl")
    joblib.dump(best_model, "models/kmeans_model.pkl")
    print("Saved kmeans_scaler.pkl and kmeans_model.pkl to models/")
    
    print("Enriching data and saving...")
    # Data Enrichment
    df['Cluster'] = best_model.labels_
    
    output_path = "data/features/segmented_customers.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved segmented data to {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
