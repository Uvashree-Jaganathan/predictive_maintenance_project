# Predictive Maintenance System

## Project Structure
```predictive_maintenance_project/```
│
├── ````data/```
│ ├── ````sensor_data.csv````           # Classification dataset
│ └── ```sensor_data_with_reg.csv```  # Regression dataset
├── ```models/```
│ ├── ```` model.pkl  ```               # Classification model
│ ├── ```scaler.pkl   ```             # Classification scaler
│ ├── ```regression_model.pkl ```     # Regression model
│ └── ```scaler_reg.pkl ```           # Regression scaler
├── ```generate_data.py   ```         # Generates synthetic datasets
├── ```train_model.py  ```            # Trains classification model
├── ```train_regression.py ```        # Trains regression model
├── ```simulate_downtime.py  ```      # Simulates downtime reduction
├── ```regression.py    ```           # Regression prediction functions
├── ```app.py ```                     # Streamlit dashboard (classification + regression)
└── ```requirements.txt    ```        # Python dependencies
