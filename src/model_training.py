import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import json
import sys
import os
from datetime import datetime

def train_model(data_path, version):
    """Train model with hyperparameter tuning"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(f"stock_prediction_v{version}")
    
    with mlflow.start_run():
        print(f"Training model v{version} at {datetime.now()}")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check target distribution
        target_dist = df['target'].value_counts()
        print(f"Target distribution: {target_dist.to_dict()}")
        
        # Prepare features
        features = ['rolling_avg_10', 'volume_sum_10']
        X = df[features].fillna(method='ffill').fillna(0)
        y = df['target']
        
        print(f"Feature shapes: {X.shape}")
        print(f"Missing values: {X.isnull().sum().to_dict()}")
        
        # Time-based split (80% train, 20% test)
        # Sort by timestamp to maintain temporal order
        df_sorted = df.sort_values('timestamp')
        split_idx = int(len(df_sorted) * 0.8)
        
        train_indices = df_sorted.index[:split_idx]
        test_indices = df_sorted.index[split_idx:]
        
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Train period: {df_sorted.loc[train_indices, 'timestamp'].min()} to {df_sorted.loc[train_indices, 'timestamp'].max()}")
        print(f"Test period: {df_sorted.loc[test_indices, 'timestamp'].min()} to {df_sorted.loc[test_indices, 'timestamp'].max()}")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50],
            'max_depth': [10],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }
        
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Detailed metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        mlflow.log_metric("precision_0", class_report['0']['precision'])
        mlflow.log_metric("recall_0", class_report['0']['recall'])
        mlflow.log_metric("precision_1", class_report['1']['precision'])
        mlflow.log_metric("recall_1", class_report['1']['recall'])
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        try:
            mlflow.register_model(model_uri, f"stock_predictor_v{version}")
            print(f"Model registered successfully as stock_predictor_v{version}")
        except Exception as e:
            print(f"Model registration failed: {e}")
        
        # Save metrics for CML
        metrics = {
            "version": version,
            "accuracy": float(accuracy),
            "best_params": grid_search.best_params_,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "feature_importance": {
                features[i]: float(best_model.feature_importances_[i]) 
                for i in range(len(features))
            }
        }
        
        with open(f'metrics_v{version}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Model v{version} - Accuracy: {accuracy:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Feature importance: {dict(zip(features, best_model.feature_importances_))}")
        
        return best_model, accuracy

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "0"
    data_path = f"data/processed/processed_v{version}.csv"
    
    if os.path.exists(data_path):
        train_model(data_path, version)
    else:
        print(f"Data file {data_path} not found. Run preprocessing first.")
