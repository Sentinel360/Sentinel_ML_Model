"""ML model loading and individual model predictions."""
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')


def load_model(path):
    """Load a pickled model/scaler/artefact, returning None on failure."""
    try:
        obj = joblib.load(path)
        return obj
    except Exception as e:
        print(f"\u26a0\ufe0f  Failed to load {path}: {str(e)[:80]}")
        return None


def predict_ghana_gb(feature_array, model, scaler):
    """Return probability-of-risky-class from Ghana Gradient Boosting."""
    scaled = scaler.transform([feature_array])
    return float(model.predict_proba(scaled)[0][1])


def predict_porto_if(feature_array, model, scaler):
    """Return sigmoid-normalised anomaly score from Porto Isolation Forest."""
    scaled = scaler.transform([feature_array])
    raw = float(model.decision_function(scaled)[0])
    return 1.0 / (1.0 + np.exp(-raw))
