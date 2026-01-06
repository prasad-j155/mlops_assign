import time
import logging
import pandas as pd
import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

from api.schema import HeartDiseaseInput

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# MLflow Configuration (LOCAL)
# ---------------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/HeartDiseaseModel/Production"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    logger.info("Model loaded successfully from MLflow Production stage.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load model - {e}")
    raise e

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts risk of heart disease using a trained ML model",
    version="1.0"
)

# ---------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of prediction requests"
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Total number of prediction errors"
)

# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "healthy"}

# ---------------------------------------------------------
# Metrics Endpoint
# ---------------------------------------------------------
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())

# ---------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------
@app.post("/predict")
def predict(data: HeartDiseaseInput):
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        input_data = data.dict()
        logger.info(f"Received request: {input_data}")

        df = pd.DataFrame([input_data])

        with REQUEST_LATENCY.time():
            prediction = model.predict(df)[0]
            confidence = model.predict_proba(df).max()

        latency = round(time.time() - start_time, 4)

        logger.info(
            f"Prediction={prediction}, Confidence={confidence:.4f}, Latency={latency}s"
        )

        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "latency_seconds": latency
        }

    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Prediction failed: {str(e)}")
        raise e
