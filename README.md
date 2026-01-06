# 🫀 Heart Disease Prediction – End-to-End MLOps Pipeline

## 📌 Project Overview
This project implements a complete **MLOps workflow** for predicting the risk of heart disease using patient health data.  
It demonstrates industry-aligned practices including experiment tracking, CI/CD automation, containerization, Kubernetes deployment, and monitoring.

**Dataset:** UCI Heart Disease Dataset  
**Problem:** Binary classification (presence / absence of heart disease)

---

## 🧰 Technology Stack
- Python 3.11
- Scikit-learn
- MLflow (experiment tracking & model registry)
- FastAPI (model serving)
- Pytest (unit testing)
- GitHub Actions (CI/CD)
- Docker (containerization)
- Kubernetes (Docker Desktop / Minikube)
- Prometheus-compatible metrics

---

## 📁 Project Structure
mlops_assign/
│
├── api/ # FastAPI application
│ ├── main.py
│ ├── schema.py
│ └── init.py
│
├── data/processed/
│ └── heart_disease_cleaned.csv
│
├── notebooks/
│ ├── eda.ipynb # Exploratory Data Analysis
│ └── training.ipynb # Model experimentation
│
├── tests/ # Unit tests
│ ├── test_api.py
│ └── test_data.py
│
├── deployment.yaml # Kubernetes Deployment
├── service.yaml # Kubernetes Service
├── Dockerfile
├── train.py # Model training & MLflow logging
├── requirements.txt
├── pytest.ini
├── README.md
└── .github/workflows/ci-cd.yml


## ⚙️ Local Environment Setup

### Create virtual environment
python -m venv venv

### Activate 
Windows: venv\Scripts\activate
Mac: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Data & EDA
data/processed/heart_disease_cleaned.csv
notebooks/eda.ipynb

### Model training
python train.py

### To start ML flow UI
mlflow ui
Access: http://localhost:5000

### Run API locally
http://localhost:5000

### Available endpoints
GET /
POST /predict
Sample body:
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
GET /metrics

### Unit testing
Run pytest

### CI/CD workflow file
.github/workflows/ci-cd.yml

### Docker
docker build -t heart-disease-api:1.0 .
docker run -p 8000:8000 heart-disease-api:1.0
API availale at: http://localhost:8000

### Kubernetes
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get pods
kubectl get svc

### Monitoring & Logging
GET /metrics

### Screenshots
Screenshots of MLflow UI, CI/CD pipeline, Docker containers, Kubernetes pods/services, API responses, and metrics are included in the screenshots/ folder

### Demo video link
https://drive.google.com/file/d/1jgA5vXEBXi3ru5fsX2gO_m0ReMUnMzt_/view?usp=drive_link
