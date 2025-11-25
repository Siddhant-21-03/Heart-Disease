from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'rf_pipeline.joblib'


def load_model(path: str | Path = None):
    path = Path(path) if path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train the model first using `train_model.py`.")
    return joblib.load(path)


def predict_from_dict(features: dict, model_path: str | Path = None) -> dict:
    """Given a dict of features, return prediction and probability.

    Args:
        features: mapping feature_name -> value
    Returns:
        dict with `prediction` and `probability`
    """
    model = load_model(model_path)
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    prob = None
    try:
        proba = model.predict_proba(df)[0]
        prob = float(max(proba))
    except Exception:
        prob = None

    return {"prediction": int(pred), "probability": prob}
