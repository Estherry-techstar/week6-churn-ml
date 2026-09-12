import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/churn_clean.csv")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def make_pipeline(model, X):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical),
        ])),
        ("clf", model),
    ])


def evaluate(model, data, name):
    X = data.drop(columns=["Churn"])
    y = data["Churn"]
    scores = cross_val_score(make_pipeline(model, X), X, y, cv=cv, scoring="roc_auc")
    print(f"{name:<38} {scores.mean():.4f} +/- {scores.std():.4f}")


print("=" * 60)
print("DOES FEATURE ENGINEERING HELP?")
print("=" * 60)
evaluate(LogisticRegression(max_iter=1000, random_state=42), df, "logreg / raw features")

eng = df.copy()
# Average spend per month of tenure — reveals plan changes over time
eng["avg_monthly"] = np.where(eng["tenure"] > 0,
                              eng["TotalCharges"] / eng["tenure"],
                              eng["MonthlyCharges"])
# Ratio of current to historical average: >1 means a recent price rise
eng["price_change"] = (eng["MonthlyCharges"] / eng["avg_monthly"].replace(0, np.nan)).fillna(1.0)
# How many products the customer holds — more products, stickier customer
services = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"]
eng["n_services"] = sum(eng[c].isin(["Yes", "DSL", "Fiber optic"]).astype(int) for c in services)
eng["is_new"] = (eng["tenure"] <= 6).astype(int)

evaluate(LogisticRegression(max_iter=1000, random_state=42), eng, "logreg / + engineered")

print("\n" + "=" * 60)
print("DOES A FANCIER MODEL HELP?")
print("=" * 60)
evaluate(DecisionTreeClassifier(random_state=42), df, "decision tree (unconstrained)")
evaluate(DecisionTreeClassifier(max_depth=5, random_state=42), df, "decision tree (max_depth=5)")
evaluate(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), df, "random forest")
evaluate(RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                random_state=42, n_jobs=-1), df, "random forest (leaf>=5)")
evaluate(HistGradientBoostingClassifier(random_state=42), df, "hist gradient boosting")
evaluate(LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42), df, "logreg (class_weight=balanced)")

print("\n" + "=" * 60)
print("WHAT OVERFITTING ACTUALLY LOOKS LIKE")
print("=" * 60)
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

for name, clf in [
    ("tree (unconstrained)", DecisionTreeClassifier(random_state=42)),
    ("tree (max_depth=5)", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("logistic regression", LogisticRegression(max_iter=1000, random_state=42)),
]:
    p = make_pipeline(clf, X).fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, p.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, p.predict_proba(X_test)[:, 1])
    print(f"{name:<22} train {train_auc:.4f}  test {test_auc:.4f}  gap {train_auc-test_auc:+.4f}")