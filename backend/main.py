
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
from backend.classification import classify_peak_demand

# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Electricity Demand Forecasting API",
    description="24-hour electricity demand forecasting using Linear Regression and Random Forest",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "backend_data"
)


# ==========================================
# LOAD MODELS
# ==========================================

# Linear Regression is small, so load it at startup
lr_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "lr_model.pkl"
    )
)

# Random Forest is large, so load it only when needed
rf_model = None


def get_rf_model():

    global rf_model

    if rf_model is None:
        rf_model = joblib.load(
            os.path.join(
                MODEL_DIR,
                "rf_model.pkl"
            )
        )

    return rf_model

# ==========================================
# LOAD CONFIGURATION
# ==========================================

with open(
    os.path.join(
        MODEL_DIR,
        "feature_config.json"
    ),
    "r"
) as f:

    feature_config = json.load(f)


feature_order = feature_config["features"]

lr_threshold = feature_config[
    "lr_anomaly_threshold_MW"
]

rf_threshold = feature_config[
    "rf_anomaly_threshold_MW"
]


# ==========================================
# LOAD DATA
# ==========================================

DATA_PATH = os.path.join(
    DATA_DIR,
    "energy_data.csv"
)

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "TimeStamp",
        "electricity_demand_MW",

        "hour",
        "day_of_week",
        "is_weekend",
        "month",
        "quarter",
        "day_of_month",
        "week_of_year",
        "is_central_holiday",
        "is_major_festival",

        "northern_t2m",
        "northern_rh2m",
        "northern_prectotcorr",
        "northern_ws10m",

        "western_t2m",
        "western_rh2m",
        "western_prectotcorr",
        "western_ws10m",

        "southern_t2m",
        "southern_rh2m",
        "southern_prectotcorr",
        "southern_ws10m",

        "eastern_t2m",
        "eastern_rh2m",
        "eastern_prectotcorr",
        "eastern_ws10m",

        "north_eastern_t2m",
        "north_eastern_rh2m",
        "north_eastern_prectotcorr",
        "north_eastern_ws10m"
    ],
    parse_dates=["TimeStamp"]
)

df = df.sort_values(
    "TimeStamp"
).reset_index(drop=True)


# ==========================================
# REQUEST SCHEMA
# ==========================================

class ForecastRequest(BaseModel):

    forecast_start: str

    model: str = "both"


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Electricity Demand Forecasting API is running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models": [
            "linear_regression",
            "random_forest"
        ],
        "data_rows": len(df)
    }


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/models")
def models():

    return {
        "models": [
            {
                "name": "linear_regression",
                "display_name": "Linear Regression",
                "anomaly_threshold_MW": lr_threshold
            },
            {
                "name": "random_forest",
                "display_name": "Random Forest",
                "anomaly_threshold_MW": rf_threshold
            }
        ]
    }


# ==========================================
# FEATURE GENERATION
# ==========================================

def create_forecast_features(
    history,
    future_row
):

    feature_values = {}

    # -------------------------
    # Lag features
    # -------------------------

    feature_values["lag_1"] = history.iloc[-1]

    feature_values["lag_24"] = history.iloc[-24]

    feature_values["lag_168"] = history.iloc[-168]


    # -------------------------
    # Rolling features
    # -------------------------

    feature_values["rolling_mean_24"] = (
        history.iloc[-24:].mean()
    )

    feature_values["rolling_mean_168"] = (
        history.iloc[-168:].mean()
    )


    # -------------------------
    # Calendar features
    # -------------------------

    feature_values["hour"] = future_row["hour"]

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


    # -------------------------
    # Holiday features
    # -------------------------

    feature_values["is_central_holiday"] = (
        future_row["is_central_holiday"]
    )

    feature_values["is_major_festival"] = (
        future_row["is_major_festival"]
    )


    # -------------------------
    # Cyclical features
    # -------------------------

    hour = future_row["hour"]

    day_of_week = future_row["day_of_week"]

    month = future_row["month"]


    feature_values["hour_sin"] = (
        np.sin(2 * np.pi * hour / 24)
    )

    feature_values["hour_cos"] = (
        np.cos(2 * np.pi * hour / 24)
    )

    feature_values["day_sin"] = (
        np.sin(2 * np.pi * day_of_week / 7)
    )

    feature_values["day_cos"] = (
        np.cos(2 * np.pi * day_of_week / 7)
    )

    feature_values["month_sin"] = (
        np.sin(2 * np.pi * month / 12)
    )

    feature_values["month_cos"] = (
        np.cos(2 * np.pi * month / 12)
    )


    # -------------------------
    # Weather features
    # -------------------------

    for feature in feature_order:

        if (
            feature not in feature_values
            and feature in future_row.index
        ):

            feature_values[feature] = (
                future_row[feature]
            )


    # -------------------------
    # Final feature dataframe
    # -------------------------

    feature_df = pd.DataFrame(
        [feature_values],
        columns=feature_order
    )

    return feature_df.astype(float)


# ==========================================
# FORECAST FUNCTION
# ==========================================

def generate_forecast(
    start_timestamp,
    model_name
):

    # --------------------------------------
    # Locate requested timestamp
    # --------------------------------------

    matching_rows = df[
        df["TimeStamp"] == start_timestamp
    ]

    if len(matching_rows) == 0:

        raise HTTPException(
            status_code=404,
            detail="Timestamp not found in dataset."
        )


    start_index = matching_rows.index[0]


    # --------------------------------------
    # Need 168 hours history
    # --------------------------------------

    if start_index < 168:

        raise HTTPException(
            status_code=400,
            detail="Not enough historical data before this timestamp."
        )


    # --------------------------------------
    # Need 24 future observations
    # --------------------------------------

    if start_index + 24 > len(df):

        raise HTTPException(
            status_code=400,
            detail="Not enough data for a complete 24-hour forecast."
        )


    # --------------------------------------
    # Initial demand history
    # --------------------------------------

    history = df.loc[
        start_index - 168:start_index - 1,
        "electricity_demand_MW"
    ].reset_index(drop=True)


    predictions = []


    # --------------------------------------
    # Recursive 24-hour forecast
    # --------------------------------------

    for step in range(24):

        future_row = df.iloc[
            start_index + step
        ]


        feature_df = create_forecast_features(
            history,
            future_row
        )


        # -------------------------------
        # Prediction
        # -------------------------------

        if model_name == "linear_regression":

            prediction = lr_model.predict(
                feature_df
            )[0]

        elif model_name == "random_forest":

            model = get_rf_model()

            prediction = model.predict(
                feature_df
            )[0]

        else:

            raise HTTPException(
                status_code=400,
                detail="Invalid model."
            )


        prediction = float(prediction)

        predictions.append(prediction)


        # -------------------------------
        # Recursive feedback
        # -------------------------------

        history = pd.concat(
            [
                history,
                pd.Series([prediction])
            ],
            ignore_index=True
        )

        history = history.iloc[-168:]


    # --------------------------------------
    # Build results
    # --------------------------------------

    results = []

    threshold = (
        lr_threshold
        if model_name == "linear_regression"
        else rf_threshold
    )


    for step in range(24):

        row = df.iloc[
            start_index + step
        ]

        actual = float(
            row["electricity_demand_MW"]
        )

        forecast = predictions[step]

        residual = actual - forecast

        absolute_residual = abs(residual)

        anomaly = (
            absolute_residual > threshold
        )


        if anomaly:

            if residual > 0:

                anomaly_type = "High Demand"

            else:

                anomaly_type = "Low Demand"

        else:

            anomaly_type = "Normal"


        results.append(
            {
                "timestamp": row[
                    "TimeStamp"
                ].strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "actual_demand_MW": actual,

                "forecasted_demand_MW": forecast,

                "residual_MW": residual,

                "absolute_residual_MW":
                    absolute_residual,

                "anomaly": bool(anomaly),

                "anomaly_type":
                    anomaly_type
            }
        )


    return results


# ==========================================
# FORECAST ENDPOINT
# ==========================================

@app.post("/forecast")
def forecast(
    request: ForecastRequest
):

    try:

        start_timestamp = pd.to_datetime(
            request.forecast_start
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid forecast_start format."
        )


    requested_model = request.model.lower()


    # ======================================
    # BOTH MODELS
    # ======================================

    if requested_model == "both":

        lr_results = generate_forecast(
            start_timestamp,
            "linear_regression"
        )

        rf_results = generate_forecast(
            start_timestamp,
            "random_forest"
        )

        return {
            "forecast_start":
                start_timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "forecast_horizon_hours": 24,

            "lr": lr_results,

            "rf": rf_results
        }


    # ======================================
    # SINGLE MODEL
    # ======================================

    if requested_model not in [
        "linear_regression",
        "random_forest"
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "model must be "
                "'linear_regression', "
                "'random_forest', or 'both'."
            )
        )


    results = generate_forecast(
        start_timestamp,
        requested_model
    )


    return {
        "forecast_start":
            start_timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "model": requested_model,

        "forecast_horizon_hours": 24,

        "results": results
    }
        # ==========================================
# PEAK DEMAND CLASSIFICATION ENDPOINT
# ==========================================

class PeakClassificationRequest(BaseModel):

    classification_date: str


@app.post("/peak-classification")
def peak_classification(
    request: PeakClassificationRequest
):

    try:

        result = classify_peak_demand(
            request.classification_date
        )

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )