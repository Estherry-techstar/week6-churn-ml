# Churn Prediction API — Week 6

Predicts the probability that a telecom customer will cancel their subscription,
and serves that prediction through a FastAPI `/predict` endpoint.

Built on the IBM Telco Customer Churn dataset: 7,043 customers, 19 features,
26.5% churn rate.

**Headline result:** logistic regression, ROC AUC **0.8424** on held-out data
(cross-validated 0.8451 ± 0.0133). At the shipped decision threshold of 0.25 the
model catches **81% of churners** at 50% precision.

See [`reports/evaluation.md`](reports/evaluation.md) for the full evaluation,
including why recall was chosen over accuracy and where the model is weak.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the raw data into `data/`:

```bash
curl -L -o data/churn_raw.csv \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
```

## Running the pipeline

Run from the project root, in this order:

```bash
python src/explore.py         # inspect raw data quality
python src/clean.py           # -> data/churn_clean.csv
python src/train_baseline.py  # dummy vs logistic regression
python src/evaluate.py        # metrics, thresholds, cost model, cross-validation
python src/experiments.py     # model comparison and overfitting demonstration
python src/train_final.py     # -> models/churn_model.joblib + metadata.json
```

## Running the API

```bash
uvicorn src.api:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check, confirms the model loaded |
| GET | `/model-info` | Model type, decision threshold, held-out metrics |
| POST | `/predict` | Churn probability for one customer |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 85.0
  }'
```

```json
{
  "churn_probability": 0.6943,
  "will_churn": true,
  "risk_band": "very high",
  "threshold": 0.25
}
```

Invalid categories are rejected with a `422` before reaching the model, so an
unknown value can never be silently encoded as all-zeros and produce a
confident-looking but meaningless prediction.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

21 tests covering the API contract, input validation, and the data-cleaning
guarantees. CI (`.github/workflows/ci.yml`) rebuilds the cleaned dataset and
retrains the model from source on every push, so the whole pipeline is verified
as reproducible rather than trusting the committed model file.

Tests assert *properties*, not exact predictions — a high-risk customer must
score above a low-risk one, probabilities must fall in [0, 1] — so retraining
does not break the suite for cosmetic reasons.

## Project structure

```
.
├── data/
│   ├── churn_raw.csv          # as downloaded
│   └── churn_clean.csv        # produced by src/clean.py
├── models/
│   ├── churn_model.joblib     # fitted sklearn Pipeline (preprocessing + model)
│   └── metadata.json          # threshold, metrics, valid categories
├── reports/
│   └── evaluation.md          # written evaluation
├── src/
│   ├── explore.py
│   ├── clean.py
│   ├── train_baseline.py
│   ├── evaluate.py
│   ├── experiments.py
│   ├── train_final.py
│   └── api.py
├── tests/
│   ├── test_api.py
│   └── test_data.py
└── .github/workflows/ci.yml
```

## Design notes

**The whole Pipeline is persisted, not just the estimator.** The `.joblib` file
contains the fitted `StandardScaler`, the fitted `OneHotEncoder` and the model
coefficients in a single object. The API therefore never reimplements
preprocessing, which removes an entire class of training/serving skew bugs.

**Metrics are reported from a model that never saw the test set**; the shipped
model is then retrained on all 7,043 rows. Reporting and deployment are
different jobs.

**The endpoint returns a probability, not just a boolean.** Downstream consumers
can apply their own threshold without a retrain, and `risk_band` lets a
retention team triage by severity instead of treating every flagged customer
identically.