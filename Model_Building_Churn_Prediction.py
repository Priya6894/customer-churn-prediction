import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Step 1: Load cleaned data
df = pd.read_csv("data/telco_cleaned.csv")

# Step 2: Define features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate performance
y_pred = model.predict(X_test)
print("✅ Model trained. Evaluation report:\n")
print(classification_report(y_test, y_pred))

# Step 6: Save model to outputs folder
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/churn_model.pkl")
print
