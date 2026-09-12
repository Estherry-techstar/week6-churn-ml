import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "models" / "churn_model.joblib")
metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
THRESHOLD = metadata["threshold"]

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn.",
    version="1.0.0",
)

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: YesNo
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNo
    OnlineBackup: YesNo
    DeviceProtection: YesNo
    TechSupport: YesNo
    StreamingTV: YesNo
    StreamingMovies: YesNo
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_band: str
    threshold: float


def risk_band(p: float) -> str:
    if p < 0.20:
        return "low"
    if p < 0.40:
        return "medium"
    if p < 0.65:
        return "high"
    return "very high"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "model_type": metadata["model_type"],
        "threshold": THRESHOLD,
        "metrics": metadata["metrics"],
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    try:
        row = pd.DataFrame([customer.model_dump()])[metadata["columns"]]
        probability = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return Prediction(
        churn_probability=round(probability, 4),
        will_churn=probability >= THRESHOLD,
        risk_band=risk_band(probability),
        threshold=THRESHOLD,
    )