FROM python:3.11-slim

WORKDIR /api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/

COPY notebooks/mlruns /mlruns
COPY notebooks/mlflow.db mlflow.db


EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
