# 🏨 Hotel Reservation Prediction with MLFlow, Jenkins & GCP

[![MLFlow](https://img.shields.io/badge/MLFlow-1.30.0-%23d9ead3?logo=mlflow)](https://mlflow.org/)
[![Jenkins](https://img.shields.io/badge/Jenkins-CI/CD-blue?logo=jenkins)](https://jenkins.io/)
[![GCP](https://img.shields.io/badge/Google_Cloud-Cloud%20Run%20%7C%20GCS-yellow?logo=google-cloud)](https://cloud.google.com/)

End-to-end machine learning pipeline for predicting hotel reservation cancellations. Features experiment tracking with MLFlow, automated CI/CD with Jenkins, and production deployment on Google Cloud Platform.

##  Project Overview

Predicts likelihood of hotel booking cancellations using historical reservation data. Key components:
- **MLFlow**: Experiment tracking, model registry & artifact storage
- **Jenkins**: CI/CD pipeline for automated testing & deployment
- **GCP**: Cloud Run for model serving + Cloud Storage for artifacts
- **Scikit-Learn/XGBoost**: Machine learning model training


##  Key Features
- **Automated ML Pipeline**:
  - Data preprocessing and feature engineering
  - Model training with multiple algorithms
  - Performance evaluation and model selection
- **MLFlow Integration**:
  - Experiment tracking with parameters/metrics
  - Model versioning and registry
  - Artifact logging (plots, preprocessors)
- **Jenkins CI/CD**:
  - Automated testing on code commits
  - Model retraining pipeline triggers
  - Deployment to staging/production
- **GCP Deployment**:
  - Docker container deployment via Cloud Run
  - REST API for real-time predictions
  - Google Cloud Storage for model artifacts

##  Getting Started

### Prerequisites
- Python 3.8+
- Docker
- Jenkins server
- GCP account (with Cloud Run & GCS enabled)
- MLFlow tracking server

### Installation
```bash
# Clone repository
git clone https://github.com/zakkou/Hotel-Reservation-Prediction-with-MLFlow-Jenkins-and-GCP-Deployment.git
cd Hotel-Reservation-Prediction-with-MLFlow-Jenkins-and-GCP-Deployment

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp-key.json"
```
