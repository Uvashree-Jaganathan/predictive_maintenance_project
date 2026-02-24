# Predictive Maintenance System

## Project Structure
```text
predictive_maintenance_project/
├── data/
│   ├── sensor_data.csv               # Classification dataset
│   └── sensor_data_with_reg.csv      # Regression dataset
├── models/
│   ├── model.pkl                     # Classification model
│   ├── scaler.pkl                    # Classification scaler
│   ├── regression_model.pkl          # Regression model
│   └── scaler_reg.pkl                # Regression scaler
├── app.py                            # Streamlit dashboard
├── generate_data.py                  # Generates synthetic datasets
├── regression.py                     # Regression prediction functions
├── simulate_downtime.py              # Simulates downtime reduction
├── train_model.py                    # Trains classification model
├── train_regression.py               # Trains regression model
└── requirements.txt                  # Python dependencies