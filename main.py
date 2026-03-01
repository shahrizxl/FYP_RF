# smartbudget_ml_api.py
# FULL - FINAL (RM300-only one-time rule + past-month guard + fixed future-month ML alignment)
# ✅ One-time tx detection: amount >= RM300 ONLY -> excludes from modeling
# ✅ Small-data friendly heuristics (works even with 1 record)
# ✅ Calendar-month forecast using anchor_year/anchor_month:
#    this_month = spent_so_far_in_that_month (REAL, includes one-time)
#              + predicted_remaining_until_month_end (habit-only)
# ✅ FIXED: future-month remaining prediction sums ONLY dates inside selected month window
# ✅ NEW: If user selects a PAST month (already complete), return ACTUAL total (no ML forecast)
# ✅ ML engine = SIMPLE RandomForest ONLY (lags + rolling means)

# Run:
#   uvicorn smartbudget_ml_api:app --host 0.0.0.0 --port 8000

from typing import List, Optional, Tuple
from datetime import timedelta, date
import calendar

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor


# =========================
# CONFIG
# =========================
MIN_UNIQUE_DAYS_FOR_ML = 30

# ✅ One-time rule: RM300 only
ONE_TIME_FLOOR = 300.0
ONE_TIME_ONLY_IF_SINGLE = True  # True: mark only the transaction; False: mark whole day if day total >= 300

SPIKE_MULTIPLIER_DEFAULT = 3.0
MIN_DAYS_FOR_STRICT_SPIKES = 14

# RandomForest params (simple engine)
RF_ESTIMATORS = 300
RF_MAX_DEPTH = 12
RANDOM_STATE = 42


# =========================
# BASIC HELPERS
# =========================
def _daily_series(df: pd.DataFrame) -> pd.Series:
    """date -> total amount for that day (only days with data)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df.groupby("date")["amount"].sum().sort_index()


def _avg_calendar_window(daily: pd.Series, anchor: pd.Timestamp, window_days: int) -> float:
    """
    Average over last N CALENDAR days ending at anchor (includes 0-spend days).
    Anchor is latest transaction date in payload (not today's date).
    """
    if window_days <= 0 or daily is None or daily.empty:
        return 0.0

    anchor = pd.Timestamp(anchor).normalize()
    idx = pd.date_range(anchor - pd.Timedelta(days=window_days - 1), anchor, freq="D")
    filled = daily.reindex(idx, fill_value=0.0)
    return float(filled.mean())


def _spike_multiplier_for_n(unique_days: int) -> float:
    """
    Small-data friendly spike multiplier:
      - very small history => relaxed spike detection
      - enough history => default
    """
    if unique_days is None or unique_days <= 0:
        return SPIKE_MULTIPLIER_DEFAULT

    if unique_days < 7:
        return 8.0
    if unique_days < MIN_DAYS_FOR_STRICT_SPIKES:
        return 5.0
    return SPIKE_MULTIPLIER_DEFAULT


def _robust_estimate_small_sample(vals: np.ndarray) -> float:
    """
    Best estimate for 2..6 days:
      - Use median (robust)
      - If >=4, drop extreme outliers using a relaxed threshold
    """
    s = pd.Series(vals.astype(float))
    if s.empty:
        return 0.0

    med = float(s.median())

    if med <= 0:
        nz = s[s > 0]
        return float(nz.mean()) if len(nz) else 0.0

    if len(s) < 4:
        return med

    mult = 8.0
    filtered = s[s <= med * mult]
    if len(filtered) == 0:
        return med
    return float(filtered.median())


def heuristic_v3_spike_safe(daily: pd.Series) -> Tuple[float, float, float]:
    """
    Heuristic (small-data friendly):
      1) Only 1 day => x, x*7, x*30
      2) 2..6 days => robust_estimate * 7/30
      3) 7..29 days => avg(last 7 calendar days incl 0s) => month = avg7*30
      4) >=30 days => day=avg7, week=avg7*7, month=avg30*30
    """
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


# =========================
# ONE-TIME TRANSACTION DETECTION (RM300 ONLY)
# =========================
def mark_one_time_transactions(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """
    Marks one-time transactions using a simple rule:

      - SINGLE transaction is one-time if amount >= RM300

    If ONE_TIME_ONLY_IF_SINGLE = False:
      - Mark the whole day as one-time if that day's total >= RM300

    Returns:
      df_marked with column is_one_time
      (floor_th, dummy)
    """
    df = raw.copy()
    floor_th = float(ONE_TIME_FLOOR)

    df["is_one_time"] = False

    if ONE_TIME_ONLY_IF_SINGLE:
        df.loc[df["amount"] >= floor_th, "is_one_time"] = True
    else:
        daily_total = _daily_series(df)
        big_days = daily_total[daily_total >= floor_th].index
        df.loc[df["date"].isin(big_days), "is_one_time"] = True

    return df, (floor_th, 0.0)


# ======================================================
# ✅ SIMPLE ML ENGINE (RandomForest ONLY)
#    (this is the part you asked to use "inside this")
# ======================================================

def aggregate_expenses_simple(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Simple feature engineering:
      - daily totals (fill missing days with 0)
      - calendar features
      - lags (1..3)
      - rolling mean 3 and 7 (shifted)
    """
    if df is None or df.empty:
        return pd.DataFrame(), "No data."

    df = df.copy()

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.lower()
        df = df[df["type"] == "expense"]

    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return pd.DataFrame(), "No data."

    daily = df.groupby("date")["amount"].sum().sort_index()
    if daily.empty:
        return pd.DataFrame(), "No data."

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    series_df = daily.reindex(full_range, fill_value=0.0).to_frame("daily_expense")

    # calendar
    series_df["day_of_week"] = series_df.index.dayofweek
    series_df["is_weekend"] = series_df["day_of_week"].isin([5, 6]).astype(int)
    series_df["month"] = series_df.index.month
    series_df["day"] = series_df.index.day

    # lags
    series_df["lag_1"] = series_df["daily_expense"].shift(1)
    series_df["lag_2"] = series_df["daily_expense"].shift(2)
    series_df["lag_3"] = series_df["daily_expense"].shift(3)

    # rolling means (shift so we don't leak today)
    series_df["rolling_mean_3"] = series_df["daily_expense"].rolling(3).mean().shift(1)
    series_df["rolling_mean_7"] = series_df["daily_expense"].rolling(7).mean().shift(1)

    series_df.dropna(inplace=True)

    # needs enough rows after lags/rollings
    if len(series_df) < 10:
        return pd.DataFrame(), "Not enough history after features."

    return series_df, None


def train_random_forest_simple(series_df: pd.DataFrame) -> Tuple[RandomForestRegressor, List[str]]:
    X = series_df.drop(columns=["daily_expense"])
    y = series_df["daily_expense"].astype(float)

    model = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model, X.columns.tolist()


def predict_future_simple(
    model: RandomForestRegressor,
    features: List[str],
    series_df: pd.DataFrame,
    days: int = 30
) -> pd.DataFrame:
    """
    Recursive predictions using last values.
    Updates only daily_expense for future dates.
    """
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
            "rolling_mean_7": float(tail7.mean()) if len(tail7) else 0.0,
        }

        X_pred = pd.DataFrame([row])

        # align columns exactly
        for f in features:
            if f not in X_pred.columns:
                X_pred[f] = 0.0
        X_pred = X_pred[features]

        pred = float(model.predict(X_pred)[0])
        pred = max(0.0, pred)

        predictions.append(pred)
        temp_df.loc[d, "daily_expense"] = pred

    return pd.DataFrame({"daily_expense_pred": predictions}, index=future_dates)


# =========================
# MAIN PREDICT (hybrid + calendar-month)
# =========================
def predict_all_horizons_multi(
    transactions_df: pd.DataFrame,
    days: int = 30,
    anchor_year: Optional[int] = None,
    anchor_month: Optional[int] = None,
):
    """
    Returns: (message, next_day, next_week, next_month)

    next_month = calendar-month forecast for anchor month:
       spent_so_far (REAL, includes one-time)
       + predicted_remaining (habit-only)

    Past month behavior:
       if anchor month already ended before latest tx date -> return ACTUAL total (no forecast)
    """
    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0

    raw = transactions_df.copy()

    # keep expense only
    if "type" in raw.columns:
        raw["type"] = raw["type"].astype(str).str.strip().str.lower()
        raw = raw[raw["type"] == "expense"]

    if raw.empty or "date" not in raw.columns or "amount" not in raw.columns:
        return "No expense data available.", 0.0, 0.0, 0.0

    # parse date/amount safely
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")

    raw = raw.dropna(subset=["date", "amount"])
    raw = raw[raw["amount"] > 0]

    if raw.empty or float(raw["amount"].sum()) == 0.0:
        return "No expense data available.", 0.0, 0.0, 0.0

    latest = raw["date"].max()

    # Decide anchor month (selected month from Flutter)
    if anchor_year is not None and anchor_month is not None:
        ay, am = int(anchor_year), int(anchor_month)
    else:
        ay, am = int(latest.year), int(latest.month)

    # anchor month boundaries
    last_day = calendar.monthrange(ay, am)[1]
    anchor_start = pd.Timestamp(date(ay, am, 1))
    anchor_end = pd.Timestamp(date(ay, am, last_day))

    # ✅ Past month guard: month is already completed and fully known
    if anchor_end < latest:
        actual = float(raw[(raw["date"] >= anchor_start) & (raw["date"] <= anchor_end)]["amount"].sum())
        return "Past month - showing actual total", 0.0, 0.0, round(actual, 2)

    # Real spent so far in anchor month (includes one-time)
    spent_so_far_end = min(latest, anchor_end)
    spent_so_far = float(
        raw[(raw["date"] >= anchor_start) & (raw["date"] <= spent_so_far_end)]["amount"].sum()
    )

    # prediction start inside selected month
    start_pred = max((latest + pd.Timedelta(days=1)).normalize(), anchor_start)

    if start_pred > anchor_end:
        remaining_days = 0
    else:
        remaining_days = int((anchor_end - start_pred).days) + 1

    # Use last 6 months if available for detection/modeling
    cutoff = latest - pd.DateOffset(months=6)
    df6 = raw[raw["date"] >= cutoff]
    df = df6 if len(df6) >= 10 else raw

    # One-time marking + exclude from model
    df_marked, (floor_th, _dummy) = mark_one_time_transactions(df)
    one_time_count = int(df_marked["is_one_time"].sum())

    df_model = df_marked[~df_marked["is_one_time"]].copy()

    daily_model = _daily_series(df_model) if not df_model.empty else pd.Series(dtype=float)
    fallback_day, fallback_week, _fallback_month30 = heuristic_v3_spike_safe(daily_model)

    unique_days = int(daily_model.index.nunique()) if len(daily_model) else 0

    # Cold start / small data => calendar-month from daily fallback
    if unique_days < MIN_UNIQUE_DAYS_FOR_ML:
        predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
        this_month_forecast = round(spent_so_far + predicted_remaining, 2)

        msg = "Using average (cold start)"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx (>= {floor_th:.0f})"
        return msg, float(fallback_day), float(fallback_week), float(this_month_forecast)

    # ✅ ML FEATURES (SIMPLE ENGINE)
    series_df, err = aggregate_expenses_simple(df_model)
    if err:
        predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
        this_month_forecast = round(spent_so_far + predicted_remaining, 2)

        msg = f"Using average ({err})"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx (>= {floor_th:.0f})"
        return msg, float(fallback_day), float(fallback_week), float(this_month_forecast)

    model, features = train_random_forest_simple(series_df)

    # ✅ predict far enough to reach anchor_end
    last_hist = series_df.index[-1].normalize()
    days_to_anchor_end = int((anchor_end - last_hist).days)
    needed_days = max(days, max(0, days_to_anchor_end))

    preds_df = predict_future_simple(model, features, series_df, days=needed_days)

    if preds_df.empty:
        predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
        this_month_forecast = round(spent_so_far + predicted_remaining, 2)

        msg = "Using average (prediction empty)"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx (>= {floor_th:.0f})"
        return msg, float(fallback_day), float(fallback_week), float(this_month_forecast)

    next_day = round(float(preds_df.iloc[0]["daily_expense_pred"]), 2)
    next_week = round(float(preds_df.head(7)["daily_expense_pred"].sum()), 2)

    # ✅ sum ONLY predictions inside the selected month window
    if remaining_days > 0:
        preds_in_month = preds_df.loc[(preds_df.index >= start_pred) & (preds_df.index <= anchor_end)]
        predicted_remaining = float(preds_in_month["daily_expense_pred"].sum())
    else:
        predicted_remaining = 0.0

    this_month_forecast = round(spent_so_far + predicted_remaining, 2)

    msg = "ML used (one-time excluded)"
    if one_time_count > 0:
        msg += f" - excluded {one_time_count} one-time tx (>= {floor_th:.0f})"

    return msg, float(next_day), float(next_week), float(this_month_forecast)


# =========================
# FASTAPI (single app)
# =========================
app = FastAPI(title="SmartBudget ML API", version="2.6.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

    # supports your Flutter month view
    anchor_year: Optional[int] = None
    anchor_month: Optional[int] = Field(default=None, ge=1, le=12)


class PredictResponse(BaseModel):
    message: str
    next_day: float
    next_week: float
    next_month: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([t.model_dump() for t in req.transactions])

    msg, next_day, next_week, next_month = predict_all_horizons_multi(
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
    }