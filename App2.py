# ============================================================
# AI HVAC PERFORMANCE PREDICTOR
# TRAINING + STREAMLIT APP (SINGLE FILE)
# ============================================================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import joblib
from datetime import datetime

import streamlit as st
import matplotlib.pyplot as plt

from CoolProp.HumidAirProp import HAPropsSI

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "models"
DATA_DIR = "data"

FEATURES = ["h", "w", "RSHF"]
TARGETS = ["W_c", "P_cond", "P_e", "Q_e", "m_r", "m_s", "Q_cond"]


# ============================================================
# PSYCHROMETRICS
# ============================================================

def humidity_ratio(Tdb, Twb, P=101325):
    return HAPropsSI("W", "Tdb", Tdb + 273.15, "Twb", Twb + 273.15, "P", P)

def enthalpy(Tdb, Twb, P=101325):
    return HAPropsSI("H", "Tdb", Tdb + 273.15, "Twb", Twb + 273.15, "P", P) / 1000


# ============================================================
# TRAIN MODELS (AUTO IF NOT FOUND)
# ============================================================

def train_and_save_models():

    st.info("🔄 Training models for first-time setup...")

    df = pd.read_csv(
        r"Sample data for ML Model - Copy.csv"
    )

    df["RSHF"] = df["RSH"] / (df["RSH"] + df["RLH"])
    df["w"] = df.apply(lambda r: humidity_ratio(r["DBT"], r["WBT"]), axis=1)
    df["h"] = df.apply(lambda r: enthalpy(r["DBT"], r["WBT"]), axis=1)

    X = df[FEATURES].values
    y = df[TARGETS].values

    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)

    X = scaler_X.transform(X)
    y = scaler_y.transform(y)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------ MODELS ------------------

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(objective="reg:squarederror", n_estimators=200)
    xgb.fit(X_train, y_train)

    cat_models = {}
    for i, t in enumerate(TARGETS):
        m = CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.1,
            verbose=0
        )
        m.fit(X_train, y_train[:, i])
        cat_models[t] = m

    ann = Sequential([
        Input(shape=(3,)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(len(TARGETS))
    ])
    ann.compile(optimizer=Adam(0.001), loss="mse")
    ann.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

    models = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "ANN": ann,
        "CatBoost": cat_models
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(models, f"{MODEL_DIR}/models.pkl")
    joblib.dump(scaler_X, f"{MODEL_DIR}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{MODEL_DIR}/scaler_y.pkl")

    st.success("✅ Models trained & saved successfully!")


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_assets():
    models = joblib.load(f"{MODEL_DIR}/models.pkl")
    scaler_X = joblib.load(f"{MODEL_DIR}/scaler_X.pkl")
    scaler_y = joblib.load(f"{MODEL_DIR}/scaler_y.pkl")
    return models, scaler_X, scaler_y


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="AI HVAC Performance Predictor",
    layout="wide",
    page_icon="❄️"
)

st.title("❄️ AI-Based HVAC Performance Prediction Dashboard")
st.markdown("**Unified ML + Psychrometrics HVAC Engine**")

# ---------- AUTO TRAIN ----------
if not os.path.exists(f"{MODEL_DIR}/models.pkl"):
    train_and_save_models()

models, scaler_X, scaler_y = load_assets()

# ---------- SIDEBAR ----------
st.sidebar.header("🔧 Input Conditions")

DBT = st.sidebar.number_input("Dry Bulb Temp (°C)", 0.0, 60.0, 25.0)
WBT = st.sidebar.number_input("Wet Bulb Temp (°C)", 0.0, 60.0, 18.0)
RSH = st.sidebar.number_input("Room Sensible Heat", 0.1, 500.0, 120.0)
RLH = st.sidebar.number_input("Room Latent Heat", 0.1, 500.0, 80.0)

model_name = st.sidebar.selectbox(
    "ML Model",
    ["Random Forest", "XGBoost", "ANN", "CatBoost"]
)

predict_btn = st.sidebar.button("🚀 Run Prediction")


# ============================================================
# PREDICTION
# ============================================================

if predict_btn:

    RSHF = RSH / (RSH + RLH)
    w = humidity_ratio(DBT, WBT)
    h = enthalpy(DBT, WBT)

    X = scaler_X.transform([[h, w, RSHF]])

    if model_name == "CatBoost":
        y_scaled = np.column_stack([
            models["CatBoost"][t].predict(X) for t in TARGETS
        ])
    else:
        y_scaled = models[model_name].predict(X)

    y = scaler_y.inverse_transform(y_scaled)[0]

    # ---------- OUTPUT ----------
    st.subheader("📊 Predicted Performance")

    cols = st.columns(len(TARGETS))
    for c, t, v in zip(cols, TARGETS, y):
        c.metric(t, f"{v:.3f}")

    # ---------- FEATURE BAR ----------
    st.subheader("🔍 Feature Space")

    fig, ax = plt.subplots()
    ax.bar(["h", "w", "RSHF"], [h, w, RSHF])
    st.pyplot(fig)

    # ---------- SAVE ----------
    os.makedirs(DATA_DIR, exist_ok=True)

    row = {
        "timestamp": datetime.now(),
        "DBT": DBT,
        "WBT": WBT,
        "RSH": RSH,
        "RLH": RLH,
        "h": h,
        "w": w,
        "RSHF": RSHF,
        "model": model_name,
        **{t: v for t, v in zip(TARGETS, y)}
    }

    file = f"{DATA_DIR}/prediction_history.csv"
    pd.DataFrame([row]).to_csv(
        file, mode="a", header=not os.path.exists(file), index=False
    )

    st.success("✅ Prediction saved!")


# ============================================================
# HISTORY
# ============================================================

st.subheader("📈 Prediction History")

if os.path.exists(f"{DATA_DIR}/prediction_history.csv"):
    hist = pd.read_csv(f"{DATA_DIR}/prediction_history.csv")
    st.dataframe(hist.tail(20), use_container_width=True)

    fig2, ax2 = plt.subplots()
    for t in TARGETS:
        ax2.plot(hist[t], label=t)
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("No historical data yet.")

