import pandas as pd
import joblib

df = pd.read_csv("data/sensor_data.csv")

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

X = df.drop("failure", axis=1)
X_scaled = scaler.transform(X)

predictions = model.predict(X_scaled)

reactive_downtime = df["failure"].sum() * 10
predictive_downtime = predictions.sum() * 6

reduction = (reactive_downtime - predictive_downtime) / reactive_downtime

print("Reactive downtime:", reactive_downtime)
print("Predictive downtime:", predictive_downtime)
print("Reduction:", reduction * 100, "%")