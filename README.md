# Heart Disease Prediction MLOps Pipeline

## Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) pipeline for predicting heart disease using multiple machine learning models. The workflow includes data preprocessing, model training, model comparison, experiment tracking, model versioning, API deployment, containerization, and continuous integration.

The objective of this project is to build a reproducible and production-ready machine learning pipeline by integrating industry-standard MLOps tools.

---

## Features

- End-to-end machine learning pipeline
- Data preprocessing and train-test split
- Training and comparison of multiple machine learning models
- Automatic best model selection
- Experiment tracking using MLflow
- Model Registry using MLflow
- Data and pipeline versioning using DVC
- REST API deployment using FastAPI
- Docker containerization
- Continuous Integration using GitHub Actions
- Automated testing using Pytest

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API Framework | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | Pytest |

---

# Project Structure

```text
HeartDisease-MLOps/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│
├── models/
│
├── screenshots/
│
├── src/
│   ├── app.py
│   ├── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
│
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Machine Learning Pipeline

The project follows a complete MLOps workflow.

## 1. Data Preparation

- Load the Heart Disease dataset
- Handle missing values
- Split data into training and testing datasets
- Save processed datasets using DVC

---

## 2. Model Training

Three machine learning models are trained independently.

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Each model is evaluated on the validation dataset.

---

## 3. Model Comparison

The performance of all trained models is compared using validation accuracy.

| Model | Accuracy | Status |
|--------|---------:|--------|
| Logistic Regression | 0.8689 | Evaluated |
| Random Forest | **0.8852** | Selected |
| XGBoost | 0.8525 | Evaluated |

The model with the highest accuracy is automatically selected and saved as:

```text
models/best_model.joblib
```

---

## 4. Model Evaluation

The selected model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

Evaluation metrics are stored in:

```text
metrics.json
```

and logged to MLflow.

---

## 5. DVC Pipeline

Pipeline stages:

- prepare
- train
- evaluate

Execute the pipeline:

```bash
dvc repro
```

Check pipeline status:

```bash
dvc status
```

---

## 6. MLflow Experiment Tracking

Launch MLflow:

```bash
mlflow ui
```

MLflow is used for:

- Experiment Tracking
- Parameter Logging
- Metric Logging
- Model Artifact Logging
- Model Registry

---

## 7. FastAPI Deployment

Start the API server:

```bash
uvicorn src.app:app --reload
```

Swagger Documentation:

```
http://localhost:8000/docs
```

Prediction Endpoint:

```
POST /predict
```

---

## 8. Docker

Build Docker Image

```bash
docker build -t heart-disease-mlops .
```

Run Docker Container

```bash
docker run -p 8000:8000 heart-disease-mlops
```

---

## 9. Continuous Integration

GitHub Actions automatically performs:

- Dependency Installation
- DVC Pipeline Execution
- Automated Testing
- Build Validation

---

# Results

## Model Comparison

The following screenshot shows the training of three machine learning models and the automatic selection of the best-performing model.

<img width="675" height="562" alt="Model Comparison" src="https://github.com/user-attachments/assets/4d0671df-dd98-456a-9040-d2d51964c5a0" />


---

## MLflow Experiment Tracking

MLflow records all experiments, parameters, metrics, and model artifacts.

<img width="1117" height="596" alt="MLFlow" src="https://github.com/user-attachments/assets/dbd7989b-288d-49f0-b52e-1aedb255addd" />


---

## Registered Model

The best-performing model is registered using MLflow Model Registry.

<img width="820" height="522" alt="Registered Models" src="https://github.com/user-attachments/assets/63e0d4f2-299d-4d72-b679-eb1449d8455b" />


---

## DVC Pipeline Tracking

The complete machine learning pipeline is tracked and reproduced using DVC.

<img width="454" height="28" alt="DVC status" src="https://github.com/user-attachments/assets/525c5024-8b05-4083-9bc4-c5f0a2f85cca" />


---

## GitHub Actions

Continuous Integration pipeline successfully executes the DVC workflow and automated tests.

<img width="1119" height="518" alt="Git Hub Action" src="https://github.com/user-attachments/assets/886a724d-b231-42ea-829f-5e1c958d8d09" />


---

## FastAPI Prediction

Prediction API exposed through FastAPI with interactive Swagger documentation.

<img width="1084" height="588" alt="FastAPI" src="https://github.com/user-attachments/assets/9189cd7d-0616-4584-8677-3112a1e19972" />


---

## Docker Deployment

Application containerized using Docker.

<img width="808" height="47" alt="Docker Container" src="https://github.com/user-attachments/assets/6060ccd4-dba8-48f7-a056-6b681631c5f8" />


---

# Future Improvements

- Hyperparameter Optimization
- Model Monitoring
- Data Drift Detection
- Automated Model Retraining
- Kubernetes Deployment
- Cloud Deployment (AWS/Azure/GCP)
- CI/CD Deployment to Cloud Platforms

---

# Author

**Kanishka Rajesh**



GitHub: https://github.com/Kanishka-Rajesh
