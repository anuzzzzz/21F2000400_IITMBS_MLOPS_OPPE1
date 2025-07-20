#!/bin/bash
set -e

echo "🚀 Starting MLOps Pipeline for Stock Prediction"
echo "=============================================="

# Add data to DVC (version both datasets)
echo "📦 Setting up DVC data versioning..."
dvc add data/v0
dvc add data/v1

# Setup Feast feature store
echo "🏪 Setting up Feast feature store..."
cd feast_features
feast apply
cd ..

# Run preprocessing for v0
echo "🔄 Processing v0 data (AARTIIND + ABCAPITAL)..."
python3 src/data_preprocessing.py 0

# Check if v0 data was created
if [ -f "data/processed/processed_v0.csv" ]; then
    echo "✅ v0 data preprocessing completed"
    
    # Train v0 model
    echo "🤖 Training v0 model..."
    python3 src/model_training.py 0
    echo "✅ v0 model training completed"
else
    echo "❌ v0 data preprocessing failed"
    exit 1
fi

# Run preprocessing for v1
echo "🔄 Processing v1 data (All 5 stocks)..."
python3 src/data_preprocessing.py 1

# Check if v1 data was created
if [ -f "data/processed/processed_v1.csv" ]; then
    echo "✅ v1 data preprocessing completed"
    
    # Train v1 model
    echo "🤖 Training v1 model..."
    python3 src/model_training.py 1
    echo "✅ v1 model training completed"
else
    echo "❌ v1 data preprocessing failed"
    exit 1
fi

# Run tests
echo "🧪 Running tests..."
python3 -m pytest tests/ -v

# Update Feast with v1 data
echo "🏪 Updating Feast with v1 data..."
cd feast_features
feast apply
cd ..

# Commit everything to Git
echo "📝 Committing to Git..."
git add .
git commit -m "Complete MLOps pipeline with v0 and v1 models

- Data preprocessing with time-aware gap filling
- Feature engineering (rolling_avg_10, volume_sum_10)
- Model training with hyperparameter tuning
- MLflow experiment tracking and model registry
- DVC data versioning with GCS remote storage
- Feast feature store integration
- Comprehensive testing suite
- CI/CD with CML reporting"

# Push DVC data to remote storage (BONUS)
echo "☁️ Pushing data to GCS remote storage..."
dvc push

echo ""
echo "🎉 Pipeline completed successfully!"
echo "=================================="
echo "📊 View MLflow UI: mlflow ui"
echo "🔍 Check Feast features: cd feast_features && feast feature-views list"
echo "📈 View metrics: cat metrics_v0.json && cat metrics_v1.json"
echo "🌐 Push to GitHub: git push origin main"
