# Stock Price Movement Prediction - MLOps Pipeline

[![CI/CD Pipeline](https://github.com/anuzzzzz/21F2000400_IITMBS_MLOPS_OPPE1/actions/workflows/ci.yml/badge.svg)](https://github.com/anuzzzzz/21F2000400_IITMBS_MLOPS_OPPE1/actions/workflows/ci.yml)

## Project Overview

This project implements a complete MLOps pipeline for predicting stock price movements using technical analysis. The system predicts whether a stock will trade higher 5 minutes later based on 10-minute historical data features.

### Problem Statement
- **Objective**: Binary classification to predict if stock price will be higher in 5 minutes
- **Features**: 10-minute rolling average and volume sum
- **Data**: NSE minute-level OHLCV data (2018-2021)
- **Approach**: Two-iteration model training (v0: 2 stocks → v1: 5 stocks)

## Architecture

### MLOps Components
- **Data Versioning**: DVC with Google Cloud Storage remote
- **Feature Store**: Feast for feature management
- **Experiment Tracking**: MLflow for model versioning and metrics
- **CI/CD**: GitHub Actions with automated testing
- **Model Registry**: MLflow model registry with versioned models

### Technology Stack
- **Language**: Python 3.9
- **ML Framework**: scikit-learn (Random Forest Classifier)
- **Data Processing**: pandas, numpy
- **Cloud Platform**: Google Cloud Platform
- **Container**: Docker (for deployment)
- **Orchestration**: Kubernetes (GKE)

## Dataset

### Data Sources
```
v0/ (First Iteration)
├── AARTIIND__EQ__NSE__NSE__MINUTE.csv
└── ABCAPITAL__EQ__NSE__NSE__MINUTE.csv

v1/ (Second Iteration - Added)
├── ABFRL__EQ__NSE__NSE__MINUTE.csv
├── ADANIENT__EQ__NSE__NSE__MINUTE.csv
└── ADANIGAS__EQ__NSE__NSE__MINUTE.csv
```

### Data Processing
- **Time-series handling**: Proper timestamp sorting and gap filling
- **Trading hours**: 9:15 AM - 3:30 PM (Monday-Friday)
- **Gap augmentation**: Forward-fill missing minutes with previous data
- **Train/Test split**: 80/20 temporal split maintaining chronological order

## Features

### Engineered Features
1. **rolling_avg_10**: 10-minute moving average of close price
2. **volume_sum_10**: Total volume traded over 10 minutes

### Target Variable
- **Binary classification**: 1 if stock closes higher 5 minutes later, 0 otherwise

## Model Performance

### Version 0 (2 stocks: AARTIIND + ABCAPITAL)
- **Accuracy**: 53.13%
- **Training samples**: 576,300
- **Test samples**: 144,075
- **Feature importance**: rolling_avg_10 (51.6%), volume_sum_10 (48.4%)

### Version 1 (5 stocks: All v0 + v1 data)
- **Accuracy**: 48.79%
- **Training samples**: 1,374,886
- **Test samples**: 343,722
- **Feature importance**: rolling_avg_10 (51.2%), volume_sum_10 (48.8%)

## Project Structure

```
21F2000400_IITMBS_MLOPS_OPPE1/
├── src/
│   ├── data_preprocessing.py    # Time-series data processing and feature engineering
│   └── model_training.py        # ML model training with hyperparameter tuning
├── tests/
│   └── test_features.py         # Feature validation tests
├── feast_features/
│   ├── feature_store.yaml       # Feast configuration
│   └── features.py              # Feature definitions
├── data/
│   ├── v0/                      # Version 0 stock data
│   ├── v1/                      # Version 1 stock data
│   └── processed/               # Processed datasets (ignored in git)
├── mlruns/                      # MLflow experiment tracking
├── .github/workflows/
│   └── ci.yml                   # CI/CD pipeline configuration
├── dvc.yaml                     # DVC pipeline definition
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation & Setup

### Prerequisites
- Python 3.9+
- Google Cloud Platform account
- Git and DVC

### Local Setup
```bash
# Clone repository
git clone https://github.com/anuzzzzz/21F2000400_IITMBS_MLOPS_OPPE1.git
cd 21F2000400_IITMBS_MLOPS_OPPE1

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init

# Configure GCS remote (optional)
dvc remote add -d gcs_storage gs://your-bucket-name
```

### Data Setup
```bash
# Get data from source repository
git clone https://github.com/IITMBSMLOps/oppe_data.git
cd oppe_data && git checkout MAY_2025_OPPE_1

# Copy data files to appropriate directories
cp oppe_data/v0/* data/v0/
cp oppe_data/v1/* data/v1/
```

## Usage

### Running the Pipeline

#### Complete Pipeline Execution
```bash
# Run full pipeline (preprocessing + training for both versions)
./run_complete_pipeline.sh
```

#### Individual Steps
```bash
# Data preprocessing
python src/data_preprocessing.py 0  # v0 data
python src/data_preprocessing.py 1  # v1 data

# Model training
python src/model_training.py 0     # v0 model
python src/model_training.py 1     # v1 model

# Run tests
python -m pytest tests/ -v

# Setup Feast
cd feast_features && feast apply
```

### MLflow Tracking
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Access at: http://localhost:5000
```

### DVC Operations
```bash
# Add data to version control
dvc add data/v0 data/v1

# Push to remote storage
dvc push

# Pull from remote storage
dvc pull

# Check pipeline status
dvc status
```

## Testing

### Feature Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_features.py::test_rolling_average_feature -v
```

### Test Coverage
- Rolling average feature validation
- Volume sum feature validation
- Target variable creation
- Data sorting verification

## CI/CD Pipeline

### GitHub Actions Workflow
- **Trigger**: Push to master branch
- **Steps**:
  1. Install dependencies
  2. Run feature tests
  3. Generate CML report
  4. Upload artifacts

### Automated Testing
- Feature engineering validation
- Data preprocessing checks
- Model performance tracking
- Pipeline status reporting

## Deployment

### Model Registry
Models are automatically registered in MLflow:
- `stock_predictor_v0`: Version 0 model
- `stock_predictor_v1`: Version 1 model

### Docker Deployment (Future Enhancement)
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
EXPOSE 8000
CMD ["python", "src/api.py"]
```

## Results & Insights

### Key Findings
1. **Feature Importance**: Both price trends and volume patterns contribute equally
2. **Model Performance**: Version 0 outperformed Version 1, suggesting potential overfitting with more diverse data
3. **Data Quality**: Proper time-series handling crucial for realistic performance
4. **Pipeline Robustness**: Automated testing ensures feature consistency

### Performance Analysis
- **Best Model**: Version 0 with 53.13% accuracy
- **Feature Balance**: Nearly equal importance between price and volume features
- **Temporal Validation**: Proper train/test split maintains chronological order

## Future Enhancements

### Technical Improvements
- [ ] Advanced feature engineering (technical indicators)
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Real-time data streaming
- [ ] A/B testing framework

### MLOps Enhancements
- [ ] Model monitoring and drift detection
- [ ] Automated retraining pipeline
- [ ] Multi-environment deployment (dev/staging/prod)
- [ ] Performance benchmarking suite

## Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request
5. Automated CI/CD validation

### Code Standards
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Use meaningful commit messages

## License

This project is developed for educational purposes as part of the IIT Madras MLOps course.

## Contact

**Student**: 21F2000400  
**Course**: MLOps - IIT Madras  
**Term**: May 2025  

## Acknowledgments

- IIT Madras MLOps Course Team
- NSE for providing market data
- Open source MLOps community

---

**Status**: ✅ All MLOps requirements completed successfully
**Last Updated**: July 2025
