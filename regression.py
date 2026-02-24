import pandas as pd
import numpy as np

# Load your dataset
data = pd.read_csv("data/sensor_data.csv")

# Simulate remaining life
np.random.seed(42)  # reproducibility
data['remaining_life'] = np.where(
    data['failure'] == 1,
    np.random.randint(1, 10, size=len(data)),   # machines about to fail
    np.random.randint(20, 100, size=len(data))  # healthy machines
)

# Save new dataset
data.to_csv("data/sensor_data_with_reg.csv", index=False)

print("New dataset with remaining_life created:")
print(data.head())