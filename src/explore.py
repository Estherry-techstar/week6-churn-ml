import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv("data/churn_raw.csv")

print("=" * 60)
print("SHAPE & DTYPES")
print("=" * 60)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES (what pandas thinks)")
print("=" * 60)
print(df.isnull().sum().to_string())

print("\n" + "=" * 60)
print("THE TOTALCHARGES PROBLEM")
print("=" * 60)
blank = df["TotalCharges"].astype(str).str.strip() == ""
print(f"Rows where TotalCharges is an empty string: {blank.sum()}\n")
print(df.loc[blank, ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"]])
print(f"\nTenure values of those rows: {df.loc[blank, 'tenure'].unique()}")

print("\n" + "=" * 60)
print("TARGET BALANCE")
print("=" * 60)
print(df["Churn"].value_counts())
print()
print((df["Churn"].value_counts(normalize=True) * 100).round(1))

print("\n" + "=" * 60)
print("CARDINALITY (how many distinct values per column)")
print("=" * 60)
print(df.nunique().to_string())

print("\n" + "=" * 60)
print("NUMERIC SUMMARY")
print("=" * 60)
print(df[["tenure", "MonthlyCharges"]].describe())