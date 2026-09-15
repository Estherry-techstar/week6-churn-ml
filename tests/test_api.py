import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 85.0,
}

LOW_RISK = {
    **HIGH_RISK, "tenure": 65, "Contract": "Two year", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "TechSupport": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 60.0, "TotalCharges": 3900.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info_has_metrics():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "roc_auc" in r.json()["metrics"]


def test_predict_returns_expected_schema():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability", "will_churn", "risk_band", "threshold"}
    assert isinstance(body["will_churn"], bool)


def test_probability_in_valid_range():
    for payload in (HIGH_RISK, LOW_RISK):
        p = client.post("/predict", json=payload).json()["churn_probability"]
        assert 0.0 <= p <= 1.0


def test_high_risk_scores_above_low_risk():
    """Behavioural test: the model must rank risk sensibly, whatever the exact numbers."""
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]
    assert high > low


def test_will_churn_agrees_with_threshold():
    body = client.post("/predict", json=HIGH_RISK).json()
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold"])


def test_predictions_are_deterministic():
    a = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    b = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    assert a == b


@pytest.mark.parametrize("field,value", [
    ("Contract", "Annual"),
    ("gender", "Other"),
    ("InternetService", "Satellite"),
    ("tenure", -5),
    ("MonthlyCharges", -10.0),
    ("SeniorCitizen", 7),
])
def test_invalid_values_rejected(field, value):
    r = client.post("/predict", json={**HIGH_RISK, field: value})
    assert r.status_code == 422


def test_missing_field_rejected():
    payload = {k: v for k, v in HIGH_RISK.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422