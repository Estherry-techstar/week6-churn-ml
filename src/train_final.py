import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

THRESHOLD = 0.25

df = pd.read_csv("data/churn_clean.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical = [c for c in X.columns if c not in numeric]


def build_pipeline():
    return Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical),
        ])),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


# 1. Honest metrics come from a model that never saw the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

eval_model = build_pipeline().fit(X_train, y_train)
proba = eval_model.predict_proba(X_test)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

metrics = {
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
    "threshold": THRESHOLD,
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "true_negatives": int(tn), "false_positives": int(fp),
    "false_negatives": int(fn), "true_positives": int(tp),
    "test_set_size": int(len(y_test)),
}
print("Held-out metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# 2. Ship a model trained on ALL the data
final_model = build_pipeline().fit(X, y)
joblib.dump(final_model, "models/churn_model.joblib")

metadata = {
    "model_type": "LogisticRegression",
    "threshold": THRESHOLD,
    "metrics": metrics,
    "columns": list(X.columns),
    "numeric_columns": numeric,
    "valid_categories": {c: sorted(X[c].unique().tolist()) for c in categorical},
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved -> models/churn_model.joblib")
print("Saved -> models/metadata.json")

# 3. Verify the round trip
reloaded = joblib.load("models/churn_model.joblib")
sample = pd.DataFrame([X.iloc[0].to_dict()])
print(f"\nReload check — probability for row 0: {reloaded.predict_proba(sample)[0, 1]:.4f}")