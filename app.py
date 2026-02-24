import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")
st.title("🔧 AI-Powered Predictive Maintenance Dashboard")

# =========================
# Load Models & Scalers
# =========================
# Classification
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Regression
reg_model = joblib.load("models/regression_model.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")

# =========================
# Sidebar Mode Selection
# =========================
mode = st.sidebar.selectbox("Select Mode", ["Manual Simulation", "Upload Dataset"])

# ==============================
# 🔹 MANUAL SIMULATION MODE
# ==============================
if mode == "Manual Simulation":
    st.subheader("Machine Condition Simulation")

    # Input sliders
    temperature = st.slider("Temperature", 30, 120, 70)
    vibration = st.slider("Vibration", 0.0, 0.1, 0.03)
    pressure = st.slider("Pressure", 10, 60, 30)
    rpm = st.slider("RPM", 500, 3000, 1500)

    if st.button("Predict"):
        # Use DataFrame to avoid scaler warnings
        X = pd.DataFrame([[temperature, vibration, pressure, rpm]],
                         columns=['temperature','vibration','pressure','rpm'])

        # Classification Prediction
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        # Regression Prediction
        X_reg_scaled = scaler_reg.transform(X)
        rul_prediction = reg_model.predict(X_reg_scaled)[0]

        # Display Metrics
        st.metric("Failure Probability", f"{prob:.2%}")
        st.metric("Predicted Remaining Life (Hours)", f"{rul_prediction:.2f}")

        if prediction == 1:
            st.error("⚠️ High Failure Risk Detected")
        else:
            st.success("✅ Machine Operating Normally")

        # Simple Visualization
        fig, ax = plt.subplots()
        ax.bar(["Failure Probability", "Remaining Life"], [prob, rul_prediction], color=['red','green'])
        ax.set_ylabel("Value")
        plt.tight_layout()
        st.pyplot(fig)

# ==============================
# 🔹 DATASET UPLOAD MODE
# ==============================
else:
    st.subheader("Upload Machine Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(data)

        features = ['temperature','vibration','pressure','rpm']

        # Classification
        X_scaled = scaler.transform(data[features])
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        # Regression
        X_reg_scaled = scaler_reg.transform(data[features])
        rul_predictions = reg_model.predict(X_reg_scaled)

        # Add predictions to dataframe
        data["Failure Prediction"] = predictions
        data["Failure Probability"] = probabilities
        data["Predicted Remaining Life"] = rul_predictions

        st.write("### Prediction Results")
        st.dataframe(data)

        # Visualizations
        st.write("### Failure Probability Distribution")
        fig1, ax1 = plt.subplots()
        ax1.hist(probabilities, bins=20, color='red', alpha=0.7)
        ax1.set_xlabel("Failure Probability")
        ax1.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig1)

        st.write("### Predicted Remaining Life Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(rul_predictions, bins=20, color='green', alpha=0.7)
        ax2.set_xlabel("Remaining Life (Hours)")
        ax2.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig2)

        # Classification metrics
        if "failure" in data.columns:
            y_true = data["failure"]
            acc = accuracy_score(y_true, predictions)
            st.write(f"### Classification Accuracy: {acc:.2f}")

            cm = confusion_matrix(y_true, predictions)
            st.write("Confusion Matrix")
            st.write(cm)

            st.text("Classification Report")
            st.text(classification_report(y_true, predictions))

        # Regression metrics (if true remaining life column exists)
        if "remaining_life" in data.columns:
            y_true_rul = data["remaining_life"]
            mae = mean_absolute_error(y_true_rul, rul_predictions)
            mse = mean_squared_error(y_true_rul, rul_predictions)
            r2 = r2_score(y_true_rul, rul_predictions)

            st.write("### Regression Metrics")
            st.write(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")