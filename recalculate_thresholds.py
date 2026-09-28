import pandas as pd
import numpy as np
import joblib
import json

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


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
# FEATURES AND TARGET
# ==========================================

X = df[feature_order].copy()

y = df["electricity_demand_MW"].copy()


# ==========================================
# CALIBRATION PERIOD
# ==========================================

calibration_start = 62236
calibration_end = 62956

calibration_hours = (
    calibration_end - calibration_start
)

print("\nCalibration period:")
print("Start index:", calibration_start)
print("End index:", calibration_end)
print("Hours:", calibration_hours)

print(
    "Timestamp start:",
    df.iloc[calibration_start]["TimeStamp"]
)

print(
    "Timestamp end:",
    df.iloc[calibration_end - 1]["TimeStamp"]
)


# ==========================================
# CALIBRATION TRAINING DATA
# ==========================================

X_cal_train = X.iloc[
    :calibration_start
].copy()

y_cal_train = y.iloc[
    :calibration_start
].copy()

print("\nCalibration training data:")
print("X:", X_cal_train.shape)
print("y:", y_cal_train.shape)


# ==========================================
# TRAIN CALIBRATION LR
# ==========================================

print("\nTraining calibration Linear Regression...")

lr_cal = LinearRegression()

lr_cal.fit(
    X_cal_train,
    y_cal_train
)


# ==========================================
# TRAIN CALIBRATION RF
# ==========================================

print("Training calibration Random Forest...")

rf_cal = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_cal.fit(
    X_cal_train,
    y_cal_train
)


print("Calibration models trained successfully!")


# ==========================================
# FEATURE GENERATION FUNCTION
# ==========================================

def create_features(history, future_row):

    feature_values = {}

    # --------------------------------------
    # Lag features
    # --------------------------------------

    feature_values["lag_1"] = history.iloc[-1]

    feature_values["lag_24"] = history.iloc[-24]

    feature_values["lag_168"] = history.iloc[-168]


    # --------------------------------------
    # Rolling features
    # --------------------------------------

    feature_values["rolling_mean_24"] = (
        history.iloc[-24:].mean()
    )

    feature_values["rolling_mean_168"] = (
        history.iloc[-168:].mean()
    )


    # --------------------------------------
    # Calendar features
    # --------------------------------------

    feature_values["hour"] = (
        future_row["hour"]
    )

    feature_values["day_of_week"] = (
        future_row["day_of_week"]
    )

    feature_values["is_weekend"] = (
        future_row["is_weekend"]
    )

    feature_values["month"] = (
        future_row["month"]
    )

    feature_values["quarter"] = (
        future_row["quarter"]
    )

    feature_values["day_of_month"] = (
        future_row["day_of_month"]
    )

    feature_values["week_of_year"] = (
        future_row["week_of_year"]
    )


    # --------------------------------------
    # Holiday features
    # --------------------------------------

    feature_values["is_central_holiday"] = (
        future_row["is_central_holiday"]
    )

    feature_values["is_major_festival"] = (
        future_row["is_major_festival"]
    )


    # --------------------------------------
    # Cyclical features
    # --------------------------------------

    hour = future_row["hour"]

    day_of_week = future_row["day_of_week"]

    month = future_row["month"]


    feature_values["hour_sin"] = (
        np.sin(
            2 * np.pi * hour / 24
        )
    )

    feature_values["hour_cos"] = (
        np.cos(
            2 * np.pi * hour / 24
        )
    )

    feature_values["day_sin"] = (
        np.sin(
            2 * np.pi * day_of_week / 7
        )
    )

    feature_values["day_cos"] = (
        np.cos(
            2 * np.pi * day_of_week / 7
        )
    )

    feature_values["month_sin"] = (
        np.sin(
            2 * np.pi * month / 12
        )
    )

    feature_values["month_cos"] = (
        np.cos(
            2 * np.pi * month / 12
        )
    )


    # --------------------------------------
    # Weather features
    # --------------------------------------

    for feature in feature_order:

        if (
            feature not in feature_values
            and feature in future_row.index
        ):

            feature_values[feature] = (
                future_row[feature]
            )


    # --------------------------------------
    # Create final feature dataframe
    # --------------------------------------

    feature_df = pd.DataFrame(
        [feature_values],
        columns=feature_order
    )

    return feature_df.astype(float)


# ==========================================
# INITIAL HISTORY
# ==========================================

initial_history = df.loc[
    calibration_start - 168:
    calibration_start - 1,
    "electricity_demand_MW"
].reset_index(drop=True)


# ==========================================
# SEPARATE HISTORIES
# ==========================================

lr_history = initial_history.copy()

rf_history = initial_history.copy()


# ==========================================
# STORAGE
# ==========================================

lr_predictions = []

rf_predictions = []

actual_values = []

timestamps = []


# ==========================================
# RECURSIVE CALIBRATION FORECAST
# ==========================================

print("\nGenerating 720-hour calibration forecasts...")


for i in range(
    calibration_start,
    calibration_end
):

    future_row = df.iloc[i]


    # --------------------------------------
    # Linear Regression features
    # --------------------------------------

    lr_features = create_features(
        lr_history,
        future_row
    )


    # --------------------------------------
    # Random Forest features
    # --------------------------------------

    rf_features = create_features(
        rf_history,
        future_row
    )


    # --------------------------------------
    # Predictions
    # --------------------------------------

    lr_prediction = lr_cal.predict(
        lr_features
    )[0]

    rf_prediction = rf_cal.predict(
        rf_features
    )[0]


    lr_prediction = float(
        lr_prediction
    )

    rf_prediction = float(
        rf_prediction
    )


    # --------------------------------------
    # Store results
    # --------------------------------------

    actual = float(
        future_row["electricity_demand_MW"]
    )

    lr_predictions.append(
        lr_prediction
    )

    rf_predictions.append(
        rf_prediction
    )

    actual_values.append(
        actual
    )

    timestamps.append(
        future_row["TimeStamp"]
    )


    # --------------------------------------
    # Recursive feedback
    # --------------------------------------

    lr_history = pd.concat(
        [
            lr_history,
            pd.Series([lr_prediction])
        ],
        ignore_index=True
    )

    lr_history = lr_history.iloc[-168:]


    rf_history = pd.concat(
        [
            rf_history,
            pd.Series([rf_prediction])
        ],
        ignore_index=True
    )

    rf_history = rf_history.iloc[-168:]


print("Calibration forecasting completed!")


# ==========================================
# CREATE CALIBRATION RESULTS
# ==========================================

calibration_results = pd.DataFrame({

    "TimeStamp": timestamps,

    "Actual_Demand_MW": actual_values,

    "LR_Forecast_MW": lr_predictions,

    "RF_Forecast_MW": rf_predictions

})


# ==========================================
# CALCULATE RESIDUALS
# ==========================================

calibration_results[
    "LR_Residual_MW"
] = (
    calibration_results["Actual_Demand_MW"]
    -
    calibration_results["LR_Forecast_MW"]
)


calibration_results[
    "RF_Residual_MW"
] = (
    calibration_results["Actual_Demand_MW"]
    -
    calibration_results["RF_Forecast_MW"]
)


# ==========================================
# ABSOLUTE RESIDUALS
# ==========================================

calibration_results[
    "LR_Absolute_Residual_MW"
] = (
    calibration_results["LR_Residual_MW"]
    .abs()
)


calibration_results[
    "RF_Absolute_Residual_MW"
] = (
    calibration_results["RF_Residual_MW"]
    .abs()
)


# ==========================================
# CALCULATE 95% THRESHOLDS
# ==========================================

lr_threshold = np.percentile(
    calibration_results[
        "LR_Absolute_Residual_MW"
    ],
    95
)


rf_threshold = np.percentile(
    calibration_results[
        "RF_Absolute_Residual_MW"
    ],
    95
)


# ==========================================
# PRINT THRESHOLDS
# ==========================================

print("\n")
print("==========================================")
print("NEW DEPLOYMENT ANOMALY THRESHOLDS")
print("==========================================")

print(
    f"LR 95th percentile threshold: "
    f"{lr_threshold:.2f} MW"
)

print(
    f"RF 95th percentile threshold: "
    f"{rf_threshold:.2f} MW"
)


# ==========================================
# RESIDUAL SUMMARY
# ==========================================

print("\n")
print("LINEAR REGRESSION CALIBRATION RESIDUALS")
print("=======================================")

print(
    calibration_results[
        "LR_Residual_MW"
    ].describe()
)


print("\n")
print("RANDOM FOREST CALIBRATION RESIDUALS")
print("===================================")

print(
    calibration_results[
        "RF_Residual_MW"
    ].describe()
)


# ==========================================
# UPDATE FEATURE CONFIGURATION
# ==========================================

config[
    "lr_anomaly_threshold_MW"
] = float(lr_threshold)

config[
    "rf_anomaly_threshold_MW"
] = float(rf_threshold)


with open(
    f"{MODEL_DIR}/feature_config.json",
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


print("\nFeature configuration updated successfully!")


# ==========================================
# SAVE CALIBRATION RESULTS
# ==========================================

calibration_results.to_csv(
    "backend_data/calibration_results.csv",
    index=False
)


print(
    "Calibration results saved to:"
)

print(
    "backend_data/calibration_results.csv"
)


# ==========================================
# FINAL CHECK
# ==========================================

print("\n")
print("==========================================")
print("THRESHOLD RECALCULATION COMPLETE")
print("==========================================")

print(
    f"LR threshold: {lr_threshold:.2f} MW"
)

print(
    f"RF threshold: {rf_threshold:.2f} MW"
)

print(
    "feature_config.json updated."
)

print(
    "Ready for FastAPI deployment."
)