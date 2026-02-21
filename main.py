from typing import List, Optional
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# ML FUNCTIONS
# =========================

FEATURES = [
    "day_of_week", "is_weekend", "month", "day",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_std_3",
    "rolling_mean_7", "rolling_std_7",
    "cumsum_7",
]


def aggregate_expenses(df: pd.DataFrame):
    """
    Build continuous daily time series + features.
    Returns: (series_df, msg)
      - series_df index: daily dates
      - columns: daily_expense + engineered features
      - msg: None if OK, else error string
    """
    if df is None or df.empty:
        return pd.DataFrame(), "No data."

    df = df.copy()

    if "type" in df.columns:
        df = df[df["type"] == "expense"]

    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return pd.DataFrame(), "No data."

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])

    if df.empty:
        return pd.DataFrame(), "No data."

    daily = df.groupby("date")["amount"].sum().sort_index()
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")

    series_df = daily.reindex(full_range, fill_value=0.0).to_frame("daily_expense")

    # calendar features
    series_df["day_of_week"] = series_df.index.dayofweek
    series_df["is_weekend"] = series_df["day_of_week"].isin([5, 6]).astype(int)
    series_df["month"] = series_df.index.month
    series_df["day"] = series_df.index.day

    # lags
    series_df["lag_1"] = series_df["daily_expense"].shift(1)
    series_df["lag_2"] = series_df["daily_expense"].shift(2)
    series_df["lag_3"] = series_df["daily_expense"].shift(3)

    # rollings (shift(1) = only past info)
    series_df["rolling_mean_3"] = series_df["daily_expense"].rolling(3).mean().shift(1)
    series_df["rolling_std_3"] = series_df["daily_expense"].rolling(3).std(ddof=0).shift(1)

    series_df["rolling_mean_7"] = series_df["daily_expense"].rolling(7).mean().shift(1)
    series_df["rolling_std_7"] = series_df["daily_expense"].rolling(7).std(ddof=0).shift(1)

    series_df["cumsum_7"] = series_df["daily_expense"].rolling(7).sum().shift(1)

    # safe fill for rolling std (first rows)
    series_df["rolling_std_3"] = series_df["rolling_std_3"].fillna(0.0)
    series_df["rolling_std_7"] = series_df["rolling_std_7"].fillna(0.0)

    # drop rows that still have NaNs (lags/means require history)
    series_df = series_df.dropna()

    # need enough rows to be meaningful
    if len(series_df) < 10:
        return pd.DataFrame(), "No data."

    return series_df, None


def train_random_forest(series_df: pd.DataFrame):
    X = series_df[FEATURES].copy()
    y = series_df["daily_expense"].astype(float)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    return model


def evaluate_random_forest(model, X_eval, y_eval, fallback_tol=1.2):
    y_eval = np.asarray(y_eval, dtype=float)

    baseline = np.full(shape=y_eval.shape, fill_value=float(np.mean(y_eval)), dtype=float)
    baseline_mae = mean_absolute_error(y_eval, baseline)

    preds = model.predict(X_eval)

    mae = mean_absolute_error(y_eval, preds)
    rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
    r2 = float(r2_score(y_eval, preds))

    accepted = (mae <= baseline_mae * fallback_tol) or (r2 > 0)

    return {
        "model": "RandomForest",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "accepted": bool(accepted),
    }


def _make_feature_row(temp_df: pd.DataFrame, d: pd.Timestamp) -> dict:
    tail7 = temp_df["daily_expense"].tail(7)
    tail3 = temp_df["daily_expense"].tail(3)

    return {
        "day_of_week": d.dayofweek,
        "is_weekend": int(d.dayofweek in [5, 6]),
        "month": d.month,
        "day": d.day,
        "lag_1": float(temp_df["daily_expense"].iloc[-1]) if len(temp_df) >= 1 else 0.0,
        "lag_2": float(temp_df["daily_expense"].iloc[-2]) if len(temp_df) >= 2 else 0.0,
        "lag_3": float(temp_df["daily_expense"].iloc[-3]) if len(temp_df) >= 3 else 0.0,
        "rolling_mean_3": float(tail3.mean()) if len(tail3) else 0.0,
        "rolling_std_3": float(tail3.std(ddof=0)) if len(tail3) > 1 else 0.0,
        "rolling_mean_7": float(tail7.mean()) if len(tail7) else 0.0,
        "rolling_std_7": float(tail7.std(ddof=0)) if len(tail7) > 1 else 0.0,
        "cumsum_7": float(tail7.sum()) if len(tail7) else 0.0,
    }


def predict_future(model, series_df, days=30):
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["daily_expense_pred"])

    last_date = series_df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")

    temp_df = series_df[["daily_expense"]].copy()  # keep only target series
    preds = []

    for d in future_dates:
        row = _make_feature_row(temp_df, d)
        X_pred = pd.DataFrame([row])[FEATURES]

        pred = float(model.predict(X_pred)[0])
        pred = max(0.0, pred)

        preds.append(pred)
        temp_df.loc[d, "daily_expense"] = pred  # feed prediction back

    return pd.DataFrame({"daily_expense_pred": preds}, index=future_dates)


def predict_all_horizons_multi(transactions_df: pd.DataFrame, days: int = 30):
    """
    Returns: (message, next_day, next_week, next_month, metrics_list_or_none)
    """
    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0, None

    df = transactions_df.copy()

    if "type" in df.columns:
        df = df[df["type"] == "expense"]

    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return "No expense data available.", 0.0, 0.0, 0.0, None

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])

    if df.empty or float(df["amount"].sum()) == 0.0:
        return "No expense data available.", 0.0, 0.0, 0.0, None

    daily = df.groupby("date")["amount"].sum().sort_index()

    # simple fallbacks (if ML not usable)
    fallback_day = round(float(daily.iloc[-1]), 2)
    fallback_week = round(float(daily.tail(7).mean() * 7), 2)
    fallback_month = round(float(daily.tail(30).mean() * 30), 2)

    series_df, msg = aggregate_expenses(df)
    if msg:
        return "Insufficient data. Using averages.", fallback_day, fallback_week, fallback_month, None

    model = train_random_forest(series_df)

    X_eval = series_df[FEATURES]
    y_eval = series_df["daily_expense"]

    metrics = evaluate_random_forest(model, X_eval, y_eval)

    if not metrics["accepted"]:
        return "Random Forest not reliable. Using averages.", fallback_day, fallback_week, fallback_month, [metrics]

    preds_df = predict_future(model, series_df, days=days)
    if preds_df.empty:
        return "Prediction failed. Using averages.", fallback_day, fallback_week, fallback_month, [metrics]

    # horizons
    next_day = round(float(preds_df.iloc[0]["daily_expense_pred"]), 2)
    next_week = round(float(preds_df.head(7)["daily_expense_pred"].sum()), 2)
    next_month = round(float(preds_df.head(min(30, len(preds_df)))["daily_expense_pred"].sum()), 2)

    return "Machine Learning model used: Random Forest (historical fit).", next_day, next_week, next_month, [metrics]


# =========================
# FASTAPI LAYER
# =========================

app = FastAPI(title="SmartBudget ML API", version="1.0.1")

# ✅ CORS for Flutter Web (edit origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domains when deploying
    allow_credentials=False,  # keep False when using ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionIn(BaseModel):
    date: str = Field(..., description="ISO date string, e.g. 2026-02-21")
    amount: float
    type: Optional[str] = Field(default="expense")
    description: Optional[str] = None
    category: Optional[str] = None


class PredictRequest(BaseModel):
    transactions: List[TransactionIn]
    days: int = Field(default=30, ge=1, le=365)


class MetricsOut(BaseModel):
    model: str
    mae: float
    rmse: float
    r2: float
    accepted: bool


class PredictResponse(BaseModel):
    message: str
    next_day: float
    next_week: float
    next_month: float
    metrics: Optional[List[MetricsOut]] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([t.model_dump() for t in req.transactions])

    msg, next_day, next_week, next_month, metrics = predict_all_horizons_multi(df, days=req.days)

    return {
        "message": msg,
        "next_day": float(next_day),
        "next_week": float(next_week),
        "next_month": float(next_month),
        "metrics": metrics,
    }