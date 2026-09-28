import pandas as pd
import numpy as np
import joblib
import json
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


print("scikit-learn version:", sklearn.__version__)


# ==========================================
# PATHS
# ==========================================

DATA_PATH = "backend_data/energy_data.csv"
MODEL_DIR = "saved_models"


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["TimeStamp"]
)

df = df.sort_values(
    "TimeStamp"
).reset_index(drop=True)

print("Dataset shape:", df.shape)


# ==========================================
# LOAD FEATURE CONFIGURATION
# ==========================================

with open(
    f"{MODEL_DIR}/feature_config.json",
    "r"
) as f:
    config = json.load(f)


feature_order = config["features"]


# ==========================================
# TARGET
# ==========================================

target = "electricity_demand_MW"

X = df[feature_order].copy()
y = df[target].copy()


# ==========================================
# CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

split = int(len(X) * 0.8)

X_train = X.iloc[:split].copy()
X_test = X.iloc[split:].copy()

y_train = y.iloc[:split].copy()
y_test = y.iloc[split:].copy()


print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# LINEAR REGRESSION
# ==========================================

print("\nTraining Linear Regression...")

lr_model = LinearRegression()

lr_model.fit(
    X_train,
    y_train
)


# ==========================================
# RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)


# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(
    lr_model,
    f"{MODEL_DIR}/lr_model.pkl"
)

joblib.dump(
    rf_model,
    f"{MODEL_DIR}/rf_model.pkl"
)


print("\nModels saved successfully!")

print(f"{MODEL_DIR}/lr_model.pkl")
print(f"{MODEL_DIR}/rf_model.pkl")


# ==========================================
# TEST LOADING
# ==========================================

print("\nTesting model loading...")

test_lr = joblib.load(
    f"{MODEL_DIR}/lr_model.pkl"
)

test_rf = joblib.load(
    f"{MODEL_DIR}/rf_model.pkl"
)

print("Linear Regression loading: OK")
print("Random Forest loading: OK")


# ==========================================
# TEST PREDICTION
# ==========================================

lr_sample = test_lr.predict(
    X_test.iloc[:5]
)

rf_sample = test_rf.predict(
    X_test.iloc[:5]
)


print("\nSample LR predictions:")
print(lr_sample)

print("\nSample RF predictions:")
print(rf_sample)


print("\nDeployment model preparation complete!")