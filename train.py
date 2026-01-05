import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from mlflow.tracking import MlflowClient
#
# ---------------------------------------------------------
# 1. Setup MLflow
# ---------------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Heart Disease")

# ---------------------------------------------------------
# 2. Load and Prepare Data
# ---------------------------------------------------------
# Assuming data is copied to the root folder in Docker
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Feature definitions
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 3. Define Models
# ---------------------------------------------------------
# Logistic Regression Pipeline
log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Random Forest Pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
])

# Helper function to calculate metrics
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

# ---------------------------------------------------------
# 4. Train and Log: Logistic Regression
# ---------------------------------------------------------
print("Training Logistic Regression...")
with mlflow.start_run(run_name="Logistic_Regression") as run_lr:
    mlflow.log_param("model_type", "LogisticRegression")
    
    # Fit
    log_reg_pipeline.fit(X_train, y_train)
    
    # Evaluate
    metrics = get_metrics(log_reg_pipeline, X_test, y_test)
    
    # Log metrics
    for k, v in metrics.items():
        mlflow.log_metric(f"test_{k}", v)
        
    # Log model (but don't register it as the main model yet)
    mlflow.sklearn.log_model(log_reg_pipeline, "model")

# ---------------------------------------------------------
# 5. Train and Log: Random Forest (The Winner)
# ---------------------------------------------------------
print("Training Random Forest...")
with mlflow.start_run(run_name="Random_Forest") as run_rf:
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    
    # Fit
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate
    metrics = get_metrics(rf_pipeline, X_test, y_test)
    
    # Log metrics
    for k, v in metrics.items():
        mlflow.log_metric(f"test_{k}", v)
    
    # Log AND Register this one (since we know it's the best from your notebook)
    mlflow.sklearn.log_model(
        rf_pipeline, 
        artifact_path="model", 
        registered_model_name="HeartDiseaseModel"
    )

# ---------------------------------------------------------
# 6. Transition Winner to Production
# ---------------------------------------------------------
client = MlflowClient()

# Get the latest version of the registered model we just created
latest_version = client.get_latest_versions("HeartDiseaseModel", stages=["None"])[0].version

client.transition_model_version_stage(
    name="HeartDiseaseModel",
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Random Forest (Version {latest_version}) transitioned to Production.")