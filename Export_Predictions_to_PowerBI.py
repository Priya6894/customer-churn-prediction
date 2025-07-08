import pandas as pd
import joblib
import os

# Step 1: Load the cleaned data
df = pd.read_csv("data/telco_cleaned.csv")

# Step 2: Load the trained model
model = joblib.load("outputs/churn_model.pkl")

# Step 3: Prepare input features
X = df.drop("Churn", axis=1)

# Step 4: Make churn predictions
df["Churn_Predicted"] = model.predict(X)

# Step 5: Save predictions for Power BI
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/churn_predictions.csv", index=False)

print("✅ churn_predictions.csv saved to: outputs/churn_predictions.csv")
