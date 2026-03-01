# smartbudget_ml_api.py (FULL - corrected: small-data friendly + calendar-month forecast)
# ✅ Detects large SINGLE transactions as "one-time"
# ✅ Excludes one-time tx from prediction modeling (heuristic + ML)
# ✅ Small-data friendly: spike rules AUTO-RELAX when history is short
# ✅ Calendar-month forecast (THIS month of latest tx date):
#    this_month_forecast = spent_so_far_this_month (REAL, includes one-time)
#                         + predicted_remaining_until_month_end (habit, one-time removed)
#
# Run:
#   uvicorn smartbudget_ml_api:app --host 0.0.0.0 --port 8000

from typing import List, Optional, Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor


# =========================
# CONFIG
# =========================
MIN_UNIQUE_DAYS_FOR_ML = 30        # ML when >=30 spend-days (after one-time removal)
MIN_ROWS_AFTER_FE = 60            # rows after feature engineering

# One-time transaction detection (transaction-level)
ONE_TIME_MULTIPLIER = 5.0         # one-time if tx_amount >= median_daily_total * 5
ONE_TIME_FLOOR = 301.0            # also require >= RM300
ONE_TIME_ONLY_IF_SINGLE = True    # only flag big single tx; not the whole day

# Spike handling (daily-level for stability)
SPIKE_MULTIPLIER_DEFAULT = 3.0
WINSOR_MULTIPLIER = 5.0
MIN_DAYS_FOR_STRICT_SPIKES = 14

# RandomForest
RF_ESTIMATORS = 220
RF_MAX_DEPTH = 14
RANDOM_STATE = 42


# =========================
# FEATURES (spike-aware)
# =========================
FEATURES = [
    "day_of_week", "is_weekend", "month", "day",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_std_3",
    "rolling_mean_7", "rolling_std_7",
    "cumsum_7",
    "is_spike_prev",
    "spike_count_7",
    "spike_count_30",
    "max_7",
    "max_30",
]


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
    Anchor should be latest transaction date in payload for best behavior.
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
      - very small history => relaxed spike detection (avoid over-flagging)
      - enough history => stricter default
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
    Best estimate for 2..6 spend-days:
      - Use median (robust)
      - If >=4 points: trim extreme outliers lightly
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


def heuristic_v3_spike_safe(daily: pd.Series, anchor: Optional[pd.Timestamp] = None) -> Tuple[float, float, float]:
    """
    Heuristic (small-data friendly), ANCHORED to latest payload date if provided:
      1) 1 spend-day => x, x*7, x*30
      2) 2..6 spend-days => robust_estimate * 7/30
      3) 7..29 spend-days => avg(last 7 calendar days incl 0s) => day=avg7, week=avg7*7, month=avg7*30
      4) >=30 spend-days => day=avg7, week=avg7*7, month=avg30*30

    NOTE: "month" here is 30-day style. Calendar-month forecast is computed separately.
    """
    if daily is None or daily.empty:
        return 0.0, 0.0, 0.0

    daily = daily.sort_index().astype(float)
    spend_days = int(daily.index.nunique())

    if anchor is None:
        anchor = daily.index.max()
    else:
        anchor = pd.Timestamp(anchor).normalize()

    if spend_days == 1:
        x = float(daily.iloc[0])
        return round(x, 2), round(x * 7, 2), round(x * 30, 2)

    if spend_days < 7:
        est = _robust_estimate_small_sample(daily.values)
        return round(est, 2), round(est * 7, 2), round(est * 30, 2)

    avg7 = _avg_calendar_window(daily, anchor, 7)

    if spend_days >= 30:
        avg30 = _avg_calendar_window(daily, anchor, 30)
        return round(avg7, 2), round(avg7 * 7, 2), round(avg30 * 30, 2)

    return round(avg7, 2), round(avg7 * 7, 2), round(avg7 * 30, 2)


def _winsorize_daily(series: pd.Series) -> pd.Series:
    """Cap extreme daily values for ML stability: cap = median * WINSOR_MULTIPLIER."""
    s = series.astype(float).copy()
    med = float(s.median()) if len(s) else 0.0
    if med > 0:
        cap = med * WINSOR_MULTIPLIER
        return s.clip(upper=cap)
    return s


def _compute_spike_flags(daily: pd.Series) -> pd.Series:
    """
    Spike day flag:
      spike if daily > median * multiplier
    multiplier auto-relaxed when history is short to avoid false spikes.
    """
    if daily is None or daily.empty:
        return pd.Series(dtype=int)

    s = daily.astype(float)
    med = float(s.median()) if len(s) else 0.0
    if med <= 0:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index)

    mult = _spike_multiplier_for_n(int(s.index.nunique()))
    return (s > (med * mult)).astype(int)


def _month_end(anchor: pd.Timestamp) -> pd.Timestamp:
    """End date of anchor's calendar month (normalized)."""
    a = pd.Timestamp(anchor).normalize()
    return (a + pd.offsets.MonthEnd(0)).normalize()


def _remaining_days_after_anchor_to_month_end(anchor: pd.Timestamp) -> int:
    """
    Remaining days AFTER anchor day until month end.
    If anchor is last day => 0
    """
    a = pd.Timestamp(anchor).normalize()
    end = _month_end(a)
    return max(0, int((end - a).days))


def _spent_so_far_in_anchor_month(df_any: pd.DataFrame, anchor: pd.Timestamp) -> float:
    """Sum of df_any amounts within anchor's calendar month up to anchor date."""
    if df_any is None or df_any.empty:
        return 0.0
    a = pd.Timestamp(anchor).normalize()
    m_start = pd.Timestamp(a.year, a.month, 1)
    mask = (df_any["date"] >= m_start) & (df_any["date"] <= a)
    return float(df_any.loc[mask, "amount"].sum())


# =========================
# ONE-TIME TRANSACTION DETECTION
# =========================

def mark_one_time_transactions(raw: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Marks large SINGLE transactions as one-time.
    Threshold = max(ONE_TIME_FLOOR, median_daily_total * ONE_TIME_MULTIPLIER)

    Returns:
      df_marked with column: is_one_time (bool)
      threshold used (float)
    """
    df = raw.copy()
    df["is_one_time"] = False

    daily_total = _daily_series(df)
    med_daily = float(daily_total.median()) if len(daily_total) else 0.0
    threshold = max(float(ONE_TIME_FLOOR), float(med_daily * ONE_TIME_MULTIPLIER))

    if ONE_TIME_ONLY_IF_SINGLE:
        df.loc[df["amount"] >= threshold, "is_one_time"] = True
    else:
        big_days = daily_total[daily_total >= threshold].index
        df.loc[df["date"].isin(big_days), "is_one_time"] = True

    return df, float(threshold)


# =========================
# ML PIPELINE (spike-aware)
# =========================

def aggregate_expenses_spike_aware(df_model: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Build continuous daily time series + spike-aware features from df_model.
    df_model should already exclude one-time transactions.
    """
    if df_model is None or df_model.empty:
        return pd.DataFrame(), "No data."

    df = df_model.copy()

    if "type" in df.columns:
        df = df[df["type"] == "expense"]

    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return pd.DataFrame(), "No data."

    daily_raw = df.groupby("date")["amount"].sum().sort_index()
    if daily_raw.empty:
        return pd.DataFrame(), "No data."

    full_range = pd.date_range(daily_raw.index.min(), daily_raw.index.max(), freq="D")
    series_df = daily_raw.reindex(full_range, fill_value=0.0).to_frame("daily_expense")

    # spike flags based on raw daily (before winsorization)
    series_df["is_spike"] = _compute_spike_flags(series_df["daily_expense"])

    # winsorize target for stable training + recursion
    series_df["daily_expense"] = _winsorize_daily(series_df["daily_expense"])

    # calendar
    series_df["day_of_week"] = series_df.index.dayofweek
    series_df["is_weekend"] = series_df["day_of_week"].isin([5, 6]).astype(int)
    series_df["month"] = series_df.index.month
    series_df["day"] = series_df.index.day

    # lags
    series_df["lag_1"] = series_df["daily_expense"].shift(1)
    series_df["lag_2"] = series_df["daily_expense"].shift(2)
    series_df["lag_3"] = series_df["daily_expense"].shift(3)

    # rollings (past only)
    series_df["rolling_mean_3"] = series_df["daily_expense"].rolling(3).mean().shift(1)
    series_df["rolling_std_3"] = series_df["daily_expense"].rolling(3).std(ddof=0).shift(1)

    series_df["rolling_mean_7"] = series_df["daily_expense"].rolling(7).mean().shift(1)
    series_df["rolling_std_7"] = series_df["daily_expense"].rolling(7).std(ddof=0).shift(1)

    series_df["cumsum_7"] = series_df["daily_expense"].rolling(7).sum().shift(1)

    # spike-aware (past only)
    series_df["is_spike_prev"] = series_df["is_spike"].shift(1).fillna(0).astype(int)
    series_df["spike_count_7"] = series_df["is_spike"].rolling(7).sum().shift(1).fillna(0).astype(int)
    series_df["spike_count_30"] = series_df["is_spike"].rolling(30).sum().shift(1).fillna(0).astype(int)

    series_df["max_7"] = series_df["daily_expense"].rolling(7).max().shift(1)
    series_df["max_30"] = series_df["daily_expense"].rolling(30).max().shift(1)

    # fill std
    series_df["rolling_std_3"] = series_df["rolling_std_3"].fillna(0.0)
    series_df["rolling_std_7"] = series_df["rolling_std_7"].fillna(0.0)

    series_df = series_df.dropna()

    if len(series_df) < MIN_ROWS_AFTER_FE:
        return pd.DataFrame(), "No data."

    return series_df, None


def train_random_forest(series_df: pd.DataFrame) -> RandomForestRegressor:
    X = series_df[FEATURES].copy()
    y = series_df["daily_expense"].astype(float)

    model = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X, y)
    return model


def _make_feature_row(temp_df: pd.DataFrame, d: pd.Timestamp) -> dict:
    tail7 = temp_df["daily_expense"].tail(7)
    tail3 = temp_df["daily_expense"].tail(3)

    spike_tail7 = temp_df["is_spike"].tail(7) if "is_spike" in temp_df.columns else pd.Series([0])
    spike_tail30 = temp_df["is_spike"].tail(30) if "is_spike" in temp_df.columns else pd.Series([0])

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

        "is_spike_prev": int(temp_df["is_spike"].iloc[-1]) if "is_spike" in temp_df.columns and len(temp_df) else 0,
        "spike_count_7": int(spike_tail7.sum()) if len(spike_tail7) else 0,
        "spike_count_30": int(spike_tail30.sum()) if len(spike_tail30) else 0,

        "max_7": float(tail7.max()) if len(tail7) else 0.0,
        "max_30": float(temp_df["daily_expense"].tail(30).max()) if len(temp_df) else 0.0,
    }


def predict_future(model: RandomForestRegressor, series_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if series_df is None or series_df.empty or days <= 0:
        return pd.DataFrame(columns=["daily_expense_pred"])

    last_date = series_df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")

    temp_df = series_df[["daily_expense", "is_spike"]].copy()
    preds = []

    # hard cap to prevent runaway recursion
    recent_med = float(temp_df["daily_expense"].tail(60).median()) if len(temp_df) else 0.0
    hard_cap = (recent_med * WINSOR_MULTIPLIER) if recent_med > 0 else None

    spike_mult = _spike_multiplier_for_n(int(series_df.index.nunique()))

    for d in future_dates:
        row = _make_feature_row(temp_df, d)
        X_pred = pd.DataFrame([row])[FEATURES]

        pred = float(model.predict(X_pred)[0])
        pred = max(0.0, pred)
        if hard_cap is not None:
            pred = min(pred, hard_cap)

        preds.append(pred)
        temp_df.loc[d, "daily_expense"] = pred

        # update spike flag using rolling median
        roll_med = float(temp_df["daily_expense"].tail(60).median()) if len(temp_df) else 0.0
        temp_df.loc[d, "is_spike"] = 1 if (roll_med > 0 and pred > roll_med * spike_mult) else 0

    return pd.DataFrame({"daily_expense_pred": preds}, index=future_dates)


# =========================
# MAIN PREDICT (hybrid) - CALENDAR MONTH FORECAST
# =========================

def predict_all_horizons_multi(transactions_df: pd.DataFrame, days: int = 30):
    """
    Returns: (message, next_day, next_week, this_month_forecast)

    ✅ One-time tx removed from modeling (habit)
    ✅ But month "spent so far" includes one-time (real month total)
    ✅ this_month_forecast is ONLY for the calendar month of the latest tx date
    """
    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0

    raw = transactions_df.copy()

    # keep only expenses
    if "type" in raw.columns:
        raw = raw[raw["type"] == "expense"]

    if raw.empty or "date" not in raw.columns or "amount" not in raw.columns:
        return "No expense data available.", 0.0, 0.0, 0.0

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    raw = raw.dropna(subset=["date", "amount"])

    if raw.empty or float(raw["amount"].sum()) == 0.0:
        return "No expense data available.", 0.0, 0.0, 0.0

    # last 6 months window based on latest date in payload
    latest = raw["date"].max()
    cutoff = latest - pd.DateOffset(months=6)
    df6 = raw[raw["date"] >= cutoff]
    df = df6 if len(df6) >= 10 else raw

    # mark one-time transactions
    df_marked, one_time_threshold = mark_one_time_transactions(df)
    one_time_count = int(df_marked["is_one_time"].sum())

    # REAL spent so far THIS month (includes one-time)
    spent_so_far_real = _spent_so_far_in_anchor_month(df_marked, latest)

    # Model-only df (one-time removed)
    df_model = df_marked[df_marked["is_one_time"] == False].copy()

    # Heuristic anchored to latest payload date (important!)
    daily_model = _daily_series(df_model)
    fallback_day, fallback_week, _fallback_month30 = heuristic_v3_spike_safe(daily_model, anchor=latest)

    # Remaining days to month end (after latest day)
    remaining_days = _remaining_days_after_anchor_to_month_end(latest)

    spend_days = int(daily_model.index.nunique()) if len(daily_model) else 0

    # ========== Cold start ==========
    if spend_days < MIN_UNIQUE_DAYS_FOR_ML:
        forecast_this_month = spent_so_far_real + (fallback_day * remaining_days)
        msg = "Using average (cold start)"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx from model (>= {one_time_threshold:.0f})"
        return msg, float(fallback_day), float(fallback_week), round(float(forecast_this_month), 2)

    # ========== Try ML ==========
    series_df, err = aggregate_expenses_spike_aware(df_model)
    if err:
        forecast_this_month = spent_so_far_real + (fallback_day * remaining_days)
        msg = "Using average (not enough history)"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx from model (>= {one_time_threshold:.0f})"
        return msg, float(fallback_day), float(fallback_week), round(float(forecast_this_month), 2)

    model = train_random_forest(series_df)

    # Predict enough days to cover:
    # - next_day (1)
    # - next_week (7)
    # - remaining days in THIS calendar month
    need_days = max(7, remaining_days, int(days))
    need_days = min(365, max(1, need_days))

    preds_df = predict_future(model, series_df, days=need_days)

    if preds_df.empty:
        forecast_this_month = spent_so_far_real + (fallback_day * remaining_days)
        msg = "Using average (prediction empty)"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx from model (>= {one_time_threshold:.0f})"
        return msg, float(fallback_day), float(fallback_week), round(float(forecast_this_month), 2)

    next_day = round(float(preds_df.iloc[0]["daily_expense_pred"]), 2)
    next_week = round(float(preds_df.head(7)["daily_expense_pred"].sum()), 2)

    month_end = _month_end(latest)
    pred_remaining = (
        float(preds_df.loc[preds_df.index <= month_end, "daily_expense_pred"].sum())
        if remaining_days > 0 else 0.0
    )

    forecast_this_month = round(float(spent_so_far_real + pred_remaining), 2)

    msg = "ML used (one-time excluded from model)"
    if one_time_count > 0:
        msg += f" - excluded {one_time_count} one-time tx from model (>= {one_time_threshold:.0f})"

    return msg, float(next_day), float(next_week), float(forecast_this_month)


# =========================
# FASTAPI
# =========================

app = FastAPI(title="SmartBudget ML API", version="2.3.2")

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


class PredictResponse(BaseModel):
    message: str
    next_day: float
    next_week: float
    next_month: float  # IMPORTANT: this is actually THIS MONTH forecast (calendar month of latest tx)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    msg, next_day, next_week, next_month = predict_all_horizons_multi(df, days=req.days)

    return {
        "message": msg,
        "next_day": float(next_day),
        "next_week": float(next_week),
        "next_month": float(next_month),
    }