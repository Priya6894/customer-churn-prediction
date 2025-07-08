import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Step 1: Load the original dataset
df = pd.read_csv("data/Telco-Big.csv")

# Step 2: Drop rows with missing values
df.dropna(inplace=True)

# Step 3: Encode all categorical (object) columns
encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col])

# Step 4: Ensure the 'data' folder exists and save the cleaned file
os.makedirs("data", exist_ok=True)
df.to_csv("data/telco_cleaned.csv", index=False)

print("✅ Cleaned data saved to: data/telco_cleaned.csv")
