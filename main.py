from typing import List, Optional, Tuple, Dict, Any
from datetime import timedelta, date
import calendar

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
MIN_UNIQUE_DAYS_FOR_ML = 30
ONE_TIME_FLOOR = 300.0
ONE_TIME_ONLY_IF_SINGLE = True

# =========================
# BASIC HELPERS & SPIKE SAFE
# =========================
def _daily_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df.groupby("date")["amount"].sum().sort_index()

def _avg_calendar_window(daily: pd.Series, anchor: pd.Timestamp, window_days: int) -> float:
    if window_days <= 0 or daily is None or daily.empty:
        return 0.0
    anchor = pd.Timestamp(anchor).normalize()
    idx = pd.date_range(anchor - pd.Timedelta(days=window_days - 1), anchor, freq="D")
    filled = daily.reindex(idx, fill_value=0.0)
    return float(filled.mean())

def _robust_estimate_small_sample(vals: np.ndarray) -> float:
    s = pd.Series(vals.astype(float))
    if s.empty:
        return 0.0
    med = float(s.median())
    if med <= 0:
        nz = s[s > 0]
        return float(nz.mean()) if len(nz) else 0.0
    if len(s) < 4:
        return med
    filtered = s[s <= med * 8.0]
    if len(filtered) == 0:
        return med
    return float(filtered.median())

def heuristic_v3_spike_safe(daily: pd.Series) -> Tuple[float, float, float]:
    if daily is None or daily.empty:
        return 0.0, 0.0, 0.0

    daily = daily.sort_index().astype(float)
    unique_days = int(daily.index.nunique())
    anchor = daily.index.max()

    if unique_days == 1:
        x = float(daily.iloc[0])
        return round(x, 2), round(x * 7, 2), round(x * 30, 2)

    if unique_days < 7:
        est = _robust_estimate_small_sample(daily.values)
        return round(est, 2), round(est * 7, 2), round(est * 30, 2)

    avg7 = _avg_calendar_window(daily, anchor, 7)

    if unique_days >= 30:
        avg30 = _avg_calendar_window(daily, anchor, 30)
        return round(avg7, 2), round(avg7 * 7, 2), round(avg30 * 30, 2)

    return round(avg7, 2), round(avg7 * 7, 2), round(avg7 * 30, 2)

def mark_one_time_transactions(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    df = raw.copy()
    floor_th = float(ONE_TIME_FLOOR)
    df["is_one_time"] = False

    # RM300-only rule (single transaction >= floor)
    if ONE_TIME_ONLY_IF_SINGLE:
        df.loc[df["amount"] >= floor_th, "is_one_time"] = True
    else:
        daily_total = _daily_series(df)
        big_days = daily_total[daily_total >= floor_th].index
        df.loc[df["date"].isin(big_days), "is_one_time"] = True

    return df, (floor_th, 0.0)

# =========================
# TRAINING WINDOW (FIXED)
# =========================
def _prev_six_full_months_window(anchor_start: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Train on the 6 full months BEFORE the anchor month.
    Example: anchor_start=2026-03-01 -> window=2025-09-01 .. 2026-02-28
    """
    anchor_start = pd.Timestamp(anchor_start).normalize()
    train_start = (anchor_start - pd.DateOffset(months=6)).replace(day=1)
    train_end = anchor_start - pd.Timedelta(days=1)
    return train_start, train_end

# ======================================================
# EXPANDED ML ENGINE & EVALUATION
# ======================================================
def aggregate_expenses(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    if df is None or df.empty:
        return pd.DataFrame(), "No data."

    df = df.copy()
    daily = df.groupby("date")["amount"].sum().sort_index()
    if daily.empty:
        return pd.DataFrame(), "No data."

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    series_df = daily.reindex(full_range, fill_value=0.0).to_frame("daily_expense")

    series_df["day_of_week"] = series_df.index.dayofweek
    series_df["is_weekend"] = series_df["day_of_week"].isin([5, 6]).astype(int)
    series_df["month"] = series_df.index.month
    series_df["day"] = series_df.index.day

    series_df["lag_1"] = series_df["daily_expense"].shift(1)
    series_df["lag_2"] = series_df["daily_expense"].shift(2)
    series_df["lag_3"] = series_df["daily_expense"].shift(3)

    series_df["rolling_mean_3"] = series_df["daily_expense"].rolling(3).mean().shift(1)
    series_df["rolling_std_3"] = series_df["daily_expense"].rolling(3).std(ddof=0).shift(1).fillna(0.0)
    series_df["rolling_mean_7"] = series_df["daily_expense"].rolling(7).mean().shift(1)
    series_df["rolling_std_7"] = series_df["daily_expense"].rolling(7).std(ddof=0).shift(1).fillna(0.0)
    series_df["cumsum_7"] = series_df["daily_expense"].rolling(7).sum().shift(1)

    series_df.dropna(inplace=True)
    if len(series_df) < 10:
        return pd.DataFrame(), "No data."

    return series_df, None

def train_rf_only(series_df: pd.DataFrame) -> Tuple[RandomForestRegressor, List[str]]:
    X = series_df.drop(columns=["daily_expense"])
    y = series_df["daily_expense"]
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

def evaluate_rf(model: RandomForestRegressor, X_eval: pd.DataFrame, y_eval: pd.Series, fallback_tol: float = 1.2) -> pd.DataFrame:
    baseline = np.full_like(y_eval, float(np.mean(y_eval)), dtype=float)
    baseline_mae = mean_absolute_error(y_eval, baseline)

    preds = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, preds)
    rmse = np.sqrt(mean_squared_error(y_eval, preds))
    r2 = r2_score(y_eval, preds)

    accepted = (mae <= baseline_mae * fallback_tol) or (r2 > 0)

    return pd.DataFrame([{
        "model": "RandomForest",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "accepted": bool(accepted)
    }])

def predict_future(model: RandomForestRegressor, features: List[str], series_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["daily_expense_pred"])

    last_date = series_df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")
    temp_df = series_df.copy()
    predictions = []

    for d in future_dates:
        tail7 = temp_df["daily_expense"].tail(7)
        tail3 = temp_df["daily_expense"].tail(3)

        row = {
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
            "cumsum_7": float(tail7.sum()) if len(tail7) else 0.0
        }

        X_pred = pd.DataFrame([row])
        for f in features:
            if f not in X_pred.columns:
                X_pred[f] = 0.0
        X_pred = X_pred[features]

        pred = max(0.0, float(model.predict(X_pred)[0]))
        predictions.append(pred)
        temp_df.loc[d, "daily_expense"] = pred

    return pd.DataFrame({"daily_expense_pred": predictions}, index=future_dates)

# =========================
# ORCHESTRATION (MERGED CALENDAR + ML)
# =========================
def predict_all_horizons_multi(
    transactions_df: pd.DataFrame,
    days: int = 30,
    anchor_year: Optional[int] = None,
    anchor_month: Optional[int] = None
):
    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0, []

    raw = transactions_df.copy()
    if "type" in raw.columns:
        raw["type"] = raw["type"].astype(str).str.strip().str.lower()
        raw = raw[raw["type"] == "expense"]

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    raw = raw.dropna(subset=["date", "amount"])
    raw = raw[raw["amount"] > 0]

    if raw.empty:
        return "No expense data available.", 0.0, 0.0, 0.0, []

    today_ts = pd.Timestamp(date.today()).normalize()
    latest = raw["date"].max()

    # anchor month selection
    if anchor_year is not None and anchor_month is not None:
        ay, am = int(anchor_year), int(anchor_month)
    else:
        ay, am = today_ts.year, today_ts.month

    last_day = calendar.monthrange(ay, am)[1]
    anchor_start = pd.Timestamp(date(ay, am, 1))
    anchor_end = pd.Timestamp(date(ay, am, last_day))

    spent_so_far = float(
        raw[(raw["date"] >= anchor_start) & (raw["date"] <= min(today_ts, anchor_end))]["amount"].sum()
    )

    start_pred = max(today_ts + pd.Timedelta(days=1), anchor_start)
    remaining_days = 0 if start_pred > anchor_end else int((anchor_end - start_pred).days) + 1

    # ✅ FIX: training data based on anchor month (previous 6 full months)
    train_start, train_end = _prev_six_full_months_window(anchor_start)
    df6_anchor = raw[(raw["date"] >= train_start) & (raw["date"] <= train_end)]

    # fallback safety if too small
    if len(df6_anchor) >= 10:
        df = df6_anchor
    else:
        cutoff = latest - pd.DateOffset(months=6)
        df6_latest = raw[raw["date"] >= cutoff]
        df = df6_latest if len(df6_latest) >= 10 else raw

    df_marked, _ = mark_one_time_transactions(df)
    df_model = df_marked[~df_marked["is_one_time"]].copy()

    daily_model = _daily_series(df_model) if not df_model.empty else pd.Series(dtype=float)
    fallback_day, fallback_week, _ = heuristic_v3_spike_safe(daily_model)
    unique_days = int(daily_model.index.nunique()) if len(daily_model) else 0

    next_day = float(fallback_day)
    next_week = float(fallback_week)
    predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
    msg = "Using average"
    metrics_list: List[Dict[str, Any]] = []

    if unique_days >= MIN_UNIQUE_DAYS_FOR_ML:
        series_df, err = aggregate_expenses(df_model)
        if not err:
            model, features = train_rf_only(series_df)

            metrics_df = evaluate_rf(model, series_df[features], series_df["daily_expense"])
            metrics_list = metrics_df.to_dict(orient="records")

            if bool(metrics_df.iloc[0]["accepted"]):
                last_hist = series_df.index[-1].normalize()
                days_to_anchor_end = max(0, int((anchor_end - last_hist).days))
                days_to_next_week = max(0, int((today_ts + pd.Timedelta(days=7) - last_hist).days))
                needed_days = max(days, days_to_anchor_end, days_to_next_week)

                preds_df = predict_future(model, features, series_df, days=needed_days)

                if not preds_df.empty:
                    msg = "RandomForest-based forecast"
                    real_tomorrow = today_ts + pd.Timedelta(days=1)
                    future_preds = preds_df[preds_df.index >= real_tomorrow]

                    if not future_preds.empty:
                        next_day = round(float(future_preds.iloc[0]["daily_expense_pred"]), 2)
                        next_week = round(float(future_preds.head(7)["daily_expense_pred"].sum()), 2)

                    if remaining_days > 0:
                        preds_in_month = preds_df.loc[(preds_df.index >= start_pred) & (preds_df.index <= anchor_end)]
                        predicted_remaining = float(preds_in_month["daily_expense_pred"].sum())
            else:
                msg = "ML model not reliable (failed MAE/R2). Using averages."

    this_month_forecast = round(spent_so_far + predicted_remaining, 2)

    # Past month = show actual total
    if anchor_end < today_ts:
        msg = "Past month - showing actual total"
        actual = float(raw[(raw["date"] >= anchor_start) & (raw["date"] <= anchor_end)]["amount"].sum())
        this_month_forecast = round(actual, 2)

    return msg, next_day, next_week, this_month_forecast, metrics_list

# =========================
# FASTAPI
# =========================
app = FastAPI(title="SmartBudget ML API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

class TransactionIn(BaseModel):
    date: str = Field(..., description="ISO date string")
    amount: float
    type: Optional[str] = Field(default="expense")
    description: Optional[str] = None
    category: Optional[str] = None

class PredictRequest(BaseModel):
    transactions: List[TransactionIn]
    days: int = Field(default=30, ge=1, le=365)
    anchor_year: Optional[int] = None
    anchor_month: Optional[int] = Field(default=None, ge=1, le=12)

class PredictResponse(BaseModel):
    message: str
    next_day: float
    next_week: float
    next_month: float
    metrics: Optional[List[Dict[str, Any]]] = None

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    msg, next_day, next_week, next_month, metrics_list = predict_all_horizons_multi(
        df,
        days=req.days,
        anchor_year=req.anchor_year,
        anchor_month=req.anchor_month,
    )
    return {
        "message": msg,
        "next_day": float(next_day),
        "next_week": float(next_week),
        "next_month": float(next_month),
        "metrics": metrics_list
    }