# smartbudget_ml_api.py (FULL - corrected + safer anchor-month math)
# ✅ One-time tx detection (>= RM300 OR >= typical_daily*mult) -> excludes from modeling
# ✅ Small-data friendly heuristics
# ✅ Calendar-month forecast using anchor_year/anchor_month:
#    this_month = spent_so_far_in_that_month (REAL, includes one-time)
#              + predicted_remaining_until_month_end (habit-only)
# ✅ FIXED: if user selects a FUTURE month, remaining-days prediction starts at anchor_start
#          (prevents predicting days outside the selected month)
#
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
# CONFIG (good defaults)
# =========================
MIN_UNIQUE_DAYS_FOR_ML = 30
MIN_ROWS_AFTER_FE = 60

ONE_TIME_MULTIPLIER = 5.0
ONE_TIME_FLOOR = 300.0            # RM300+ means >= 300
ONE_TIME_ONLY_IF_SINGLE = True
ONE_TIME_USE_OR_RULE = True       # OR rule (>= floor OR >= multiplier threshold)

SPIKE_MULTIPLIER_DEFAULT = 3.0
WINSOR_MULTIPLIER = 5.0
MIN_DAYS_FOR_STRICT_SPIKES = 14

RF_ESTIMATORS = 220
RF_MAX_DEPTH = 14
RANDOM_STATE = 42

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
    multiplier auto-relaxed for small histories.
    """
    if daily is None or daily.empty:
        return pd.Series(dtype=int)

    s = daily.astype(float)
    med = float(s.median()) if len(s) else 0.0
    if med <= 0:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index)

    mult = _spike_multiplier_for_n(int(s.index.nunique()))
    return (s > (med * mult)).astype(int)


# =========================
# ONE-TIME TRANSACTION DETECTION
# =========================
def _robust_typical_daily(daily_total: pd.Series) -> float:
    """
    Robust 'typical daily' for one-time threshold:
    - median -> trim days > median*8 -> median again (if enough remains)
    """
    if daily_total is None or daily_total.empty:
        return 0.0

    s = daily_total.astype(float).copy()
    base_med = float(s.median())

    if base_med <= 0:
        nz = s[s > 0]
        return float(nz.median()) if len(nz) else 0.0

    trimmed = s[s <= base_med * 8.0]
    if len(trimmed) >= max(3, int(0.6 * len(s))):
        return float(trimmed.median())

    return base_med


def mark_one_time_transactions(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """
    Marks large SINGLE transactions as one-time.

    Two thresholds:
      floor_th = ONE_TIME_FLOOR
      mult_th  = typical_daily_total * ONE_TIME_MULTIPLIER

    If ONE_TIME_USE_OR_RULE:
      one-time if amount >= floor_th OR amount >= mult_th
    Else:
      one-time if amount >= max(floor_th, mult_th)

    Returns:
      df_marked with column is_one_time
      (floor_th, mult_th) for message display
    """
    df = raw.copy()

    daily_total = _daily_series(df)
    typical_daily = _robust_typical_daily(daily_total)

    floor_th = float(ONE_TIME_FLOOR)
    mult_th = float(typical_daily * ONE_TIME_MULTIPLIER)

    df["is_one_time"] = False

    if ONE_TIME_ONLY_IF_SINGLE:
        if ONE_TIME_USE_OR_RULE:
            df.loc[(df["amount"] >= floor_th) | (df["amount"] >= mult_th), "is_one_time"] = True
        else:
            threshold = max(floor_th, mult_th)
            df.loc[df["amount"] >= threshold, "is_one_time"] = True
    else:
        # day-level mode
        if ONE_TIME_USE_OR_RULE:
            big_days = daily_total[(daily_total >= floor_th) | (daily_total >= mult_th)].index
        else:
            threshold = max(floor_th, mult_th)
            big_days = daily_total[daily_total >= threshold].index
        df.loc[df["date"].isin(big_days), "is_one_time"] = True

    return df, (floor_th, mult_th)


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

    series_df["is_spike"] = _compute_spike_flags(series_df["daily_expense"])
    series_df["daily_expense"] = _winsorize_daily(series_df["daily_expense"])

    series_df["day_of_week"] = series_df.index.dayofweek
    series_df["is_weekend"] = series_df["day_of_week"].isin([5, 6]).astype(int)
    series_df["month"] = series_df.index.month
    series_df["day"] = series_df.index.day

    series_df["lag_1"] = series_df["daily_expense"].shift(1)
    series_df["lag_2"] = series_df["daily_expense"].shift(2)
    series_df["lag_3"] = series_df["daily_expense"].shift(3)

    series_df["rolling_mean_3"] = series_df["daily_expense"].rolling(3).mean().shift(1)
    series_df["rolling_std_3"] = series_df["daily_expense"].rolling(3).std(ddof=0).shift(1)

    series_df["rolling_mean_7"] = series_df["daily_expense"].rolling(7).mean().shift(1)
    series_df["rolling_std_7"] = series_df["daily_expense"].rolling(7).std(ddof=0).shift(1)

    series_df["cumsum_7"] = series_df["daily_expense"].rolling(7).sum().shift(1)

    series_df["is_spike_prev"] = series_df["is_spike"].shift(1).fillna(0).astype(int)
    series_df["spike_count_7"] = series_df["is_spike"].rolling(7).sum().shift(1).fillna(0).astype(int)
    series_df["spike_count_30"] = series_df["is_spike"].rolling(30).sum().shift(1).fillna(0).astype(int)

    series_df["max_7"] = series_df["daily_expense"].rolling(7).max().shift(1)
    series_df["max_30"] = series_df["daily_expense"].rolling(30).max().shift(1)

    series_df["rolling_std_3"] = series_df["rolling_std_3"].fillna(0.0)
    series_df["rolling_std_7"] = series_df["rolling_std_7"].fillna(0.0)

    series_df = series_df.dropna()

    if len(series_df) < MIN_ROWS_AFTER_FE:
        return pd.DataFrame(), f"Not enough history after features (have {len(series_df)}, need {MIN_ROWS_AFTER_FE})."

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
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["daily_expense_pred"])

    last_date = series_df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")

    temp_df = series_df[["daily_expense", "is_spike"]].copy()
    preds = []

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

        roll_med = float(temp_df["daily_expense"].tail(60).median()) if len(temp_df) else 0.0
        temp_df.loc[d, "is_spike"] = 1 if (roll_med > 0 and pred > roll_med * spike_mult) else 0

    return pd.DataFrame({"daily_expense_pred": preds}, index=future_dates)


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
    """
    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0

    raw = transactions_df.copy()

    # keep expense only
    if "type" in raw.columns:
        raw["type"] = raw["type"].astype(str).str.strip().str.lower()
        raw = raw[raw["type"] == "expense"]

    # must have date+amount
    if raw.empty or "date" not in raw.columns or "amount" not in raw.columns:
        return "No expense data available.", 0.0, 0.0, 0.0

    # parse date/amount safely
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")

    # drop invalid
    raw = raw.dropna(subset=["date", "amount"])

    # keep positive only
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

    # Real spent so far in anchor month (includes one-time)
    spent_so_far_end = min(latest, anchor_end)
    spent_so_far = float(
        raw[(raw["date"] >= anchor_start) & (raw["date"] <= spent_so_far_end)]["amount"].sum()
    )

    # ✅ FIX: prediction start is inside the selected month
    # - normally: day after latest transaction
    # - but if selected month is FUTURE: start at anchor_start (not before it)
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
    df_marked, (floor_th, mult_th) = mark_one_time_transactions(df)
    one_time_count = int(df_marked["is_one_time"].sum())

    df_model = df_marked[df_marked["is_one_time"] == False].copy()

    daily_model = _daily_series(df_model) if not df_model.empty else pd.Series(dtype=float)
    fallback_day, fallback_week, _fallback_month30 = heuristic_v3_spike_safe(daily_model)

    unique_days = int(daily_model.index.nunique()) if len(daily_model) else 0

    # Cold start / small data => calendar-month from daily fallback
    if unique_days < MIN_UNIQUE_DAYS_FOR_ML:
        predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
        this_month_forecast = round(spent_so_far + predicted_remaining, 2)

        msg = "Using average (cold start)"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx (>= {floor_th:.0f} OR >= {mult_th:.0f})"
        return msg, float(fallback_day), float(fallback_week), float(this_month_forecast)

    # Build ML features
    series_df, err = aggregate_expenses_spike_aware(df_model)
    if err:
        predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
        this_month_forecast = round(spent_so_far + predicted_remaining, 2)

        msg = f"Using average ({err})"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx (>= {floor_th:.0f} OR >= {mult_th:.0f})"
        return msg, float(fallback_day), float(fallback_week), float(this_month_forecast)

    model = train_random_forest(series_df)

    # Need enough future days for month end
    needed_days = max(days, remaining_days)
    preds_df = predict_future(model, series_df, days=needed_days)

    if preds_df.empty:
        predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
        this_month_forecast = round(spent_so_far + predicted_remaining, 2)

        msg = "Using average (prediction empty)"
        if one_time_count > 0:
            msg += f" - excluded {one_time_count} one-time tx"
        return msg, float(fallback_day), float(fallback_week), float(this_month_forecast)

    next_day = round(float(preds_df.iloc[0]["daily_expense_pred"]), 2)
    next_week = round(float(preds_df.head(7)["daily_expense_pred"].sum()), 2)

    predicted_remaining = (
        float(preds_df.head(remaining_days)["daily_expense_pred"].sum())
        if remaining_days > 0
        else 0.0
    )
    this_month_forecast = round(spent_so_far + predicted_remaining, 2)

    msg = "ML used (one-time excluded)"
    if one_time_count > 0:
        msg += f" - excluded {one_time_count} one-time tx (>= {floor_th:.0f} OR >= {mult_th:.0f})"

    return msg, float(next_day), float(next_week), float(this_month_forecast)


# =========================
# FASTAPI (single app)
# =========================
app = FastAPI(title="SmartBudget ML API", version="2.4.2")

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