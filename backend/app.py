from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = None  # <-- IMPORTANT : ne pas charger au démarrage


def get_latest_model():
    """Trouve et charge le modèle avec le numéro de version le plus élevé."""
    artifacts_path = Path("artifacts")
    models = list(artifacts_path.glob("model_v*.joblib"))
    if not models:
        default_model = artifacts_path / "model.joblib"
        if default_model.exists():
            return joblib.load(default_model)
        raise FileNotFoundError("Aucun modèle trouvé dans le dossier artifacts.")

    latest_model_path = sorted(models, key=lambda x: int(x.stem.split("_v")[-1]))[-1]
    return joblib.load(latest_model_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    global model
    if model is None:
        model = get_latest_model()  # <-- chargé seulement si nécessaire

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}