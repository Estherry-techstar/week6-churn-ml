import pandas as pd
import sklearn
import fastapi

df = pd.read_csv("data/churn_raw.csv")

print(f"pandas       {pd.__version__}")
print(f"scikit-learn {sklearn.__version__}")
print(f"fastapi      {fastapi.__version__}")
print(f"\nShape: {df.shape}")
print(f"\nChurn breakdown:\n{df['Churn'].value_counts()}")