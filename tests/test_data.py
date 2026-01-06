import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def test_dataset_loading():
    df = pd.read_csv("data/processed/heart_disease_cleaned.csv")
    assert not df.empty
    assert "target" in df.columns


def test_model_training_basic():
    """Test if a simple ML pipeline can be trained without errors"""
    df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert len(preds) == len(y_test)
