import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from api.schema import HeartDiseaseInput

# ---------------------------
# MLflow configuration
# ---------------------------
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

# mlflow.set_tracking_uri("sqlite:////mlflow/mlflow.db")
# mlflow.set_registry_uri("sqlite:////mlflow/mlflow.db")


# MODEL_URI = "models:/HeartDiseaseModel/2"

MODEL_URI = "models:/HeartDiseaseModel/Production"


# Load model once at startup
model = mlflow.sklearn.load_model(MODEL_URI)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts risk of heart disease using MLflow model",
    version="1.0"
)

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).max()

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
