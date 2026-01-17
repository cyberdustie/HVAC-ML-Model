import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
from CoolProp.HumidAirProp import HAPropsSI

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AI HVAC Performance Predictor",
    layout="wide",
    page_icon="❄️"
)

# ==============================
# LOAD MODELS & SCALERS
# ==============================
@st.cache_resource
def load_assets():
    models = joblib.load("models/models.pkl")
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    return models, scaler_X, scaler_y

models, scaler_X, scaler_y = load_assets()

targets = ["W_c", "P_cond", "P_e", "Q_e", "m_r", "m_s", "Q_cond"]

# ==============================
# PSYCHROMETRICS
# ==============================
def humidity_ratio(Tdb, Twb, P=101325):
    return HAPropsSI("W", "Tdb", Tdb + 273.15, "Twb", Twb + 273.15, "P", P)

def enthalpy(Tdb, Twb, P=101325):
    return HAPropsSI("H", "Tdb", Tdb + 273.15, "Twb", Twb + 273.15, "P", P) / 1000

# ==============================
# SAVE PREDICTIONS
# ==============================
def save_prediction(row):
    os.makedirs("data", exist_ok=True)
    file_path = "data/prediction_history.csv"

    df = pd.DataFrame([row])
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("🔧 User Inputs")

DBT = st.sidebar.number_input("Dry Bulb Temperature (°C)", 0.0, 60.0, 25.0)
WBT = st.sidebar.number_input("Wet Bulb Temperature (°C)", 0.0, 60.0, 18.0)
RSH = st.sidebar.number_input("Room Sensible Heat (RSH)", 0.1, 500.0, 120.0)
RLH = st.sidebar.number_input("Room Latent Heat (RLH)", 0.1, 500.0, 80.0)

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost", "ANN", "CatBoost"]
)

predict_btn = st.sidebar.button("🚀 Run Prediction")

# ==============================
# MAIN DASHBOARD
# ==============================
st.title("❄️ AI-Based HVAC Performance Prediction Dashboard")
st.markdown("**Advanced ML-driven prediction with psychrometric intelligence**")

if predict_btn:
    RSHF = RSH / (RSH + RLH)
    w = humidity_ratio(DBT, WBT)
    h = enthalpy(DBT, WBT)

    X = scaler_X.transform([[h, w, RSHF]])

    if model_name == "CatBoost":
     y_scaled = np.column_stack([
        models["CatBoost"][t].predict(X) for t in targets
     ])

    else:
        y_scaled = models[model_name].predict(X)

    y = scaler_y.inverse_transform(y_scaled)[0]

    # ==============================
    # KPI DISPLAY
    # ========================streamlit======
    st.subheader("📊 Predicted Outputs")

    cols = st.columns(len(targets))
    for col, t, val in zip(cols, targets, y):
        col.metric(t, f"{val:.3f}")

    # ==============================
    # FEATURE VISUALIZATION
    # ==============================
    st.subheader("📈 Input Feature Space")

    fig, ax = plt.subplots()
    ax.bar(["Enthalpy (h)", "Humidity Ratio (w)", "RSHF"], [h, w, RSHF])
    ax.set_ylabel("Value")
    st.pyplot(fig)

    # ==============================
    # SAVE DATA
    # ==============================
    save_prediction({
        "timestamp": datetime.now(),
        "DBT": DBT,
        "WBT": WBT,
        "RSH": RSH,
        "RLH": RLH,
        "h": h,
        "w": w,
        "RSHF": RSHF,
        "model": model_name,
        **{t: v for t, v in zip(targets, y)}
    })

    st.success("✅ Prediction saved for future training!")

# ==============================
# HISTORY & ANALYTICS
# ==============================
st.subheader("📚 Prediction History")

if os.path.exists("data/prediction_history.csv"):
    hist = pd.read_csv("data/prediction_history.csv")
    st.dataframe(hist.tail(20), use_container_width=True)

    st.subheader("📉 Trend Analysis")
    fig2, ax2 = plt.subplots()
    for t in targets:
        ax2.plot(hist[t], label=t)
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("No prediction history available yet.")