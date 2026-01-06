from api.main import model

def test_model_loaded():
    assert model is not None
    assert hasattr(model, "predict")
