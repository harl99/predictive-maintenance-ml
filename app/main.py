import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import PredictionInput, PredictionOutput


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_pipeline.pkl"

model = joblib.load(MODEL_PATH)

logger.info("Loading ML model from %s", MODEL_PATH)
logger.info("ML model loaded successfully")

app = FastAPI(
    title="Predictive Maintenance ML API",
    description="API for industrial machine failure prediction using a trained Random Forest model.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance ML API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    logger.info("Received prediction request")

    input_df = pd.DataFrame([{
        "Type": input_data.type,
        "Air temperature [K]": input_data.air_temperature,
        "Process temperature [K]": input_data.process_temperature,
        "Rotational speed [rpm]": input_data.rotational_speed,
        "Torque [Nm]": input_data.torque,
        "Tool wear [min]": input_data.tool_wear,
    }])

    probability = model.predict_proba(input_df)[0][1]

    threshold = 0.3
    prediction = int(probability >= threshold)

    if probability >= 0.7:
        risk_label = "High Risk"
    elif probability >= 0.3:
        risk_label = "Medium Risk"
    else:
        risk_label = "Low Risk"

    logger.info(
        "Prediction completed | probability=%s | prediction=%s | risk=%s",
        probability,
        prediction,
        risk_label,
    )

    return PredictionOutput(
        prediction=prediction,
        failure_probability=round(float(probability), 4),
        risk_label=risk_label,
    )
