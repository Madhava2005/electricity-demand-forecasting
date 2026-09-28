import os
import joblib
import pandas as pd


# ==========================================
# CLASSIFICATION MODEL
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
# LOAD MODEL
# ==========================================

classification_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "peak_demand_classification_model.pkl"
    )
)


# ==========================================
# CLASSIFICATION FEATURES
# ==========================================

classification_features = [
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_max_7"
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
    parse_dates=["TimeStamp"]
)

df = df.sort_values(
    "TimeStamp"
).reset_index(drop=True)


# ==========================================
# CREATE DAILY PEAK DATASET
# ==========================================

daily_peak = (
    df.groupby(
        df["TimeStamp"].dt.date
    )["electricity_demand_MW"]
    .max()
    .reset_index()
)

daily_peak.columns = [
    "date",
    "daily_peak_demand_MW"
]

daily_peak["date"] = pd.to_datetime(
    daily_peak["date"]
)


# ==========================================
# CREATE CALENDAR FEATURES
# ==========================================

daily_peak["day_of_week"] = (
    daily_peak["date"].dt.dayofweek
)

daily_peak["day_of_month"] = (
    daily_peak["date"].dt.day
)

daily_peak["month"] = (
    daily_peak["date"].dt.month
)

daily_peak["quarter"] = (
    daily_peak["date"].dt.quarter
)

daily_peak["week_of_year"] = (
    daily_peak["date"].dt.isocalendar().week.astype(int)
)

daily_peak["is_weekend"] = (
    daily_peak["day_of_week"] >= 5
).astype(int)


# ==========================================
# PREVIOUS DEMAND FEATURES
# ==========================================

daily_peak["lag_1"] = (
    daily_peak["daily_peak_demand_MW"].shift(1)
)

daily_peak["lag_7"] = (
    daily_peak["daily_peak_demand_MW"].shift(7)
)

daily_peak["rolling_mean_7"] = (
    daily_peak["daily_peak_demand_MW"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily_peak["rolling_max_7"] = (
    daily_peak["daily_peak_demand_MW"]
    .shift(1)
    .rolling(7)
    .max()
)


# ==========================================
# CLASSIFY DATE
# ==========================================

def classify_peak_demand(target_date):

    target_date = pd.to_datetime(
        target_date
    ).normalize()

    matching_rows = daily_peak[
        daily_peak["date"] == target_date
    ]

    if len(matching_rows) == 0:
        raise ValueError(
            "Date not found in dataset."
        )

    row = matching_rows.iloc[0]

    # Make sure all required features exist
    feature_values = {}

    for feature in classification_features:
        feature_values[feature] = row[feature]

    feature_df = pd.DataFrame(
        [feature_values],
        columns=classification_features
    )

    # Check for insufficient history
    if feature_df.isnull().any().any():
        raise ValueError(
            "Not enough historical data "
            "to calculate classification features."
        )

    prediction = classification_model.predict(
        feature_df
    )[0]

    prediction = int(prediction)

    if prediction == 1:
        classification = "Peak"
    else:
        classification = "Normal"

    return {
        "date": target_date.strftime(
            "%Y-%m-%d"
        ),
        "prediction": prediction,
        "classification": classification
    }