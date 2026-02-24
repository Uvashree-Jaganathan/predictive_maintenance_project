import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset with RUL
data = pd.read_csv("data/sensor_data_with_reg.csv")

# Features
X = data[['temperature','vibration','pressure','rpm']]

# Regression target
y = data['remaining_life']

# Scale features
scaler_reg = StandardScaler()
X_scaled = scaler_reg.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train RandomForest Regressor
reg_model = RandomForestRegressor()
reg_model.fit(X_train, y_train)

# Evaluate
predictions = reg_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save model and scaler
joblib.dump(reg_model, "models/regression_model.pkl")
joblib.dump(scaler_reg, "models/scaler_reg.pkl")

print("Regression model training complete!")