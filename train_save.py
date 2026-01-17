import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import joblib
from CoolProp.HumidAirProp import HAPropsSI

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# =========================
# LOAD DATA (CHANGE PATH IF NEEDED)
# =========================
df = pd.read_csv("C:\ReseaRCH\AI in HVAC\Sample data for ML Model - Copy.csv")

# =========================
# PSYCHROMETRICS
# =========================
def humidity_ratio(Tdb, Twb, P=101325):
    return HAPropsSI("W", "Tdb", Tdb + 273.15, "Twb", Twb + 273.15, "P", P)

def enthalpy(Tdb, Twb, P=101325):
    return HAPropsSI("H", "Tdb", Tdb + 273.15, "Twb", Twb + 273.15, "P", P) / 1000

df["RSHF"] = df["RSH"] / (df["RSH"] + df["RLH"])
df["w"] = df.apply(lambda r: humidity_ratio(r["DBT"], r["WBT"]), axis=1)
df["h"] = df.apply(lambda r: enthalpy(r["DBT"], r["WBT"]), axis=1)

features = ["h", "w", "RSHF"]
targets = ["W_c", "P_cond", "P_e", "Q_e", "m_r", "m_s", "Q_cond"]

X = df[features].values
y = df[targets].values

scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)

X = scaler_X.transform(X)
y = scaler_y.transform(y)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# MODELS
# =========================
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

xgb = XGBRegressor(objective="reg:squarederror", n_estimators=200)
xgb.fit(X_train, y_train)

cat_models = {}
for i, t in enumerate(targets):
    m = CatBoostRegressor(iterations=300, depth=8, learning_rate=0.1, verbose=0)
    m.fit(X_train, y_train[:, i])
    cat_models[t] = m

def cat_predict(X):
    return np.column_stack([cat_models[t].predict(X) for t in targets])

ann = Sequential([
    Input(shape=(3,)),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(7)
])
ann.compile(optimizer=Adam(0.001), loss="mse")
ann.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

models = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "ANN": ann,
    "CatBoost": cat_models   # ✅ NOT cat_predict  # SAVE MODELS, NOT FUNCTION
}


# =========================
# SAVE EVERYTHING
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(models, "models/models.pkl")
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

print("✅ MODELS SAVED SUCCESSFULLY")
