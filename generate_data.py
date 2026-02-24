import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_samples = 5000

temperature = np.random.normal(70, 10, n_samples)
vibration = np.random.normal(0.03, 0.01, n_samples)
pressure = np.random.normal(30, 5, n_samples)
rpm = np.random.normal(1500, 300, n_samples)

failure = (
    (temperature > 85) |
    (vibration > 0.05) |
    (pressure > 40) |
    (rpm > 2000)
).astype(int)

df = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "rpm": rpm,
    "failure": failure
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/sensor_data.csv", index=False)

print("Dataset generated")