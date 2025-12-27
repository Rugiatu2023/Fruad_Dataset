from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="Fraud Detection API")

# Load model
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    values: list[float]

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.values).reshape(1, -1)
    prediction = int(model.predict(x)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x)[0][1])

    return {
        "prediction": prediction,
        "proba_fraud": proba
    }

