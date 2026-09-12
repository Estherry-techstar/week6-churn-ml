import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

df = pd.read_csv("data/churn_clean.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical = [c for c in X.columns if c not in numeric]

model = Pipeline([
    ("pre", ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical),
    ])),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)

print("=" * 58)
print("OVERFITTING CHECK")
print("=" * 58)
tr, te = model.score(X_train, y_train), model.score(X_test, y_test)
print(f"Train accuracy: {tr:.4f}")
print(f"Test  accuracy: {te:.4f}")
print(f"Gap:            {tr - te:+.4f}")

proba = model.predict_proba(X_test)[:, 1]
print(f"\nROC AUC: {roc_auc_score(y_test, proba):.4f}")

print("\n" + "=" * 58)
print("THRESHOLD SWEEP")
print("=" * 58)
print("thr    precision  recall   f1     flagged  missed")
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"{t:.2f}   {precision_score(y_test, pred, zero_division=0):.3f}      "
          f"{recall_score(y_test, pred):.3f}    "
          f"{f1_score(y_test, pred):.3f}   {tp+fp:5d}    {fn:5d}")

print("\n" + "=" * 58)
print("COST MODEL")
print("=" * 58)
OFFER_COST = 50      # retention discount per customer contacted
CUSTOMER_VALUE = 770 # annual revenue from a retained customer
SAVE_RATE = 0.30     # fraction of contacted churners actually retained
print(f"Offer ${OFFER_COST} | customer worth ${CUSTOMER_VALUE} | save rate {SAVE_RATE:.0%}\n")
print("thr   offers  saved  lost    net cost")
best = None
for t in np.arange(0.10, 0.85, 0.05):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cost = (tp + fp) * OFFER_COST + (fn + tp * (1 - SAVE_RATE)) * CUSTOMER_VALUE
    print(f"{t:.2f}  {tp+fp:6d}  {tp*SAVE_RATE:5.0f}  {fn+tp*(1-SAVE_RATE):5.0f}  ${cost:9,.0f}")
    if best is None or cost < best[1]:
        best = (t, cost)
do_nothing = y_test.sum() * CUSTOMER_VALUE
print(f"\nBest threshold: {best[0]:.2f} at ${best[1]:,.0f}")
print(f"Do nothing:     ${do_nothing:,.0f}")
print(f"Saving:         ${do_nothing - best[1]:,.0f}")

print("\n" + "=" * 58)
print("5-FOLD CROSS-VALIDATION")
print("=" * 58)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"AUC per fold:      {np.round(auc, 4)}")
print(f"AUC mean {auc.mean():.4f} +/- {auc.std():.4f}")
print(f"Accuracy per fold: {np.round(acc, 4)}")
print(f"Acc mean {acc.mean():.4f} +/- {acc.std():.4f}")