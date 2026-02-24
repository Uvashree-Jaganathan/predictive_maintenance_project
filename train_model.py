import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/sensor_data.csv")

X = df.drop("failure", axis=1)
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, preds))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved")