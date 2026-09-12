import pandas as pd

df = pd.read_csv("data/churn_raw.csv")
print(f"Loaded: {df.shape}")

# 1. Drop the identifier — one unique value per row, zero predictive content
df = df.drop(columns=["customerID"])

# 2. Force TotalCharges to numeric. errors="coerce" turns unparseable
#    values into NaN instead of raising, so we can see and handle them.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_bad = df["TotalCharges"].isnull().sum()
print(f"TotalCharges unparseable: {n_bad}")

# 3. Those rows all have tenure=0: new customers, never billed.
#    The correct value is 0, NOT the median.
df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = 0.0

# 4. Target to 0/1 so metrics treat "Yes" as the positive class
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# 5. Collapse redundant third categories
service_cols = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
for col in service_cols:
    df[col] = df[col].replace(
        {"No internet service": "No", "No phone service": "No"}
    )

# Verify
print("\nDtypes after cleaning:")
print(df.dtypes.value_counts())
print(f"\nAny nulls left: {df.isnull().sum().sum()}")
print(f"Churn rate: {df['Churn'].mean():.4f}")
print(f"Final shape: {df.shape}")

df.to_csv("data/churn_clean.csv", index=False)
print("\nSaved -> data/churn_clean.csv")