FROM python:3.11-slim

WORKDIR /api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1. Copy necessary code and data
COPY api/ api/
# Copy the data folder (Adjust source path if your local data is elsewhere)
COPY data/processed/heart_disease_cleaned.csv data/processed/heart_disease_cleaned.csv
COPY train.py .

# 2. RUN TRAINING inside the image
# This generates mlflow.db and mlruns with correct Linux paths (/api/mlruns)
RUN python train.py

# 3. Expose and Run
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]