import mlflow
import os

def test_mlflow():
    mlflow.set_tracking_uri("http://localhost:5002")
    mlflow.set_experiment("NeuralRetail-Phase2")
    
    with mlflow.start_run(run_name="Feature-Engineering-Verification"):
        mlflow.log_param("phase", 2)
        mlflow.log_metric("rfm_count", 90000) # Dummy metric
        print("Successfully logged to MLflow at http://localhost:5002")

if __name__ == "__main__":
    test_mlflow()
