import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data/churn_clean.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"]

# SPLIT FIRST — before any scaling or encoding
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")
print(f"Churn rate — train {y_train.mean():.4f}, test {y_test.mean():.4f}")

numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical = [c for c in X.columns if c not in numeric]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical),
])

# --- Baseline 1: the do-nothing model ---
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

print("\n" + "=" * 55)
print("DUMMY — always predicts 'no churn'")
print("=" * 55)
print(f"Accuracy: {accuracy_score(y_test, dummy_pred):.4f}")
print(confusion_matrix(y_test, dummy_pred))
print(classification_report(y_test, dummy_pred,
                            target_names=["Stay", "Churn"], zero_division=0))

# --- Baseline 2: logistic regression ---
model = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("=" * 55)
print("LOGISTIC REGRESSION")
print("=" * 55)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))

n_feat = model.named_steps["pre"].transform(X_train).shape[1]
print(f"Features after encoding: {n_feat} (from {X_train.shape[1]} columns)")