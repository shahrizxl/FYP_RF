from typing import List, Optional, Tuple
from datetime import timedelta

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor


# =========================
# ML FEATURES
# =========================

FEATURES = [
    "day_of_week", "is_weekend", "month", "day",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_std_3",
    "rolling_mean_7", "rolling_std_7",
    "cumsum_7",
]


# =========================
# HEURISTIC (V2) HELPERS
# =========================

def _daily_series(df: pd.DataFrame) -> pd.Series:
    """date -> total amount for that day (only days with data)."""
    return df.groupby("date")["amount"].sum().sort_index()


def _avg_calendar_window(daily: pd.Series, anchor: pd.Timestamp, window_days: int) -> float:
    """
    Average over last N CALENDAR days ending at anchor (includes 0-spend days).
    Anchor is based on latest transaction date, not today.
    """
    if window_days <= 0 or daily is None or daily.empty:
        return 0.0

    anchor = pd.Timestamp(anchor).normalize()
    idx = pd.date_range(anchor - pd.Timedelta(days=window_days - 1), anchor, freq="D")
    filled = daily.reindex(idx, fill_value=0.0)
    return float(filled.mean())


def heuristic_v2_from_daily(daily: pd.Series) -> Tuple[float, float, float]:
    """
    Your rules:
      1) Only 1 day => x, x*7, x*30
      2) 2..6 days => avg(days_with_data) * 7/30
      3) >=7 days => avg(last 7 calendar days, incl 0s) => day=avg7, week=avg7*7
      4) month => if >=30 days => avg(last 30 calendar days, incl 0s)*30 else avg7*30
    """
    if daily is None or daily.empty:
        return 0.0, 0.0, 0.0

    unique_days = int(daily.index.nunique())
    total = float(daily.sum())
    anchor = daily.index.max()

    # 1) Only 1 day of data
    if unique_days == 1:
        x = float(daily.iloc[0])
        return round(x, 2), round(x * 7, 2), round(x * 30, 2)

    # 2) 2..6 days: average over days WITH data
    if unique_days < 7:
        avg = total / unique_days
        return round(avg, 2), round(avg * 7, 2), round(avg * 30, 2)

    # 3) >=7 days: last 7 CALENDAR days average (includes zeros)
    avg7 = _avg_calendar_window(daily, anchor, 7)

    # 4) month: if >=30 unique days use last 30 CALENDAR avg, else use avg7
    avg30 = _avg_calendar_window(daily, anchor, 30) if unique_days >= 30 else avg7

    return round(avg7, 2), round(avg7 * 7, 2), round(avg30 * 30, 2)


# =========================
# ML PIPELINE
# =========================

def aggregate_expenses(df: pd.DataFrame):
    """
    Build continuous daily time series + features.
    Returns: (series_df, msg)
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

    series_df["rolling_std_3"] = series_df["rolling_std_3"].fillna(0.0)
    series_df["rolling_std_7"] = series_df["rolling_std_7"].fillna(0.0)

    series_df = series_df.dropna()

    # need some history after feature engineering
    if len(series_df) < 10:
        return pd.DataFrame(), "No data."

    return series_df, None


def train_random_forest(series_df: pd.DataFrame):
    X = series_df[FEATURES].copy()
    y = series_df["daily_expense"].astype(float)

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


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


def predict_future(model, series_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["daily_expense_pred"])

    last_date = series_df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")

    temp_df = series_df[["daily_expense"]].copy()
    preds = []

    for d in future_dates:
        row = _make_feature_row(temp_df, d)
        X_pred = pd.DataFrame([row])[FEATURES]

        pred = float(model.predict(X_pred)[0])
        pred = max(0.0, pred)

        preds.append(pred)
        temp_df.loc[d, "daily_expense"] = pred

    return pd.DataFrame({"daily_expense_pred": preds}, index=future_dates)


def predict_all_horizons_multi(transactions_df: pd.DataFrame, days: int = 30):
    """
    Returns: (message, next_day, next_week, next_month)
    ✅ Enforces last 6 months (based on latest date in payload), falls back if too little.
    ✅ Heuristic V2 fallback (your rules) is done in API (Flutter does nothing).
    """
    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0

    raw = transactions_df.copy()

    if "type" in raw.columns:
        raw = raw[raw["type"] == "expense"]

    if raw.empty or "date" not in raw.columns or "amount" not in raw.columns:
        return "No expense data available.", 0.0, 0.0, 0.0

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    raw = raw.dropna(subset=["date", "amount"])

    if raw.empty or float(raw["amount"].sum()) == 0.0:
        return "No expense data available.", 0.0, 0.0, 0.0

    # ✅ last 6 months window based on latest date in payload
    latest = raw["date"].max()
    cutoff = latest - pd.DateOffset(months=6)
    df6 = raw[raw["date"] >= cutoff]

    # fallback if too small
    df = df6 if len(df6) >= 10 else raw

    # ✅ heuristic V2 fallback (always available)
    daily = _daily_series(df)
    fallback_day, fallback_week, fallback_month = heuristic_v2_from_daily(daily)

    # Optional stronger gate: if too few unique days, skip ML
    if daily.index.nunique() < 10:
        return "Using average", fallback_day, fallback_week, fallback_month

    # ✅ try ML
    series_df, msg = aggregate_expenses(df)
    if msg:
        return "Using average", fallback_day, fallback_week, fallback_month

    model = train_random_forest(series_df)

    preds_df = predict_future(model, series_df, days=days)
    if preds_df.empty:
        return "Using average", fallback_day, fallback_week, fallback_month

    next_day = round(float(preds_df.iloc[0]["daily_expense_pred"]), 2)
    next_week = round(float(preds_df.head(7)["daily_expense_pred"].sum()), 2)
    next_month = round(float(preds_df.head(min(30, len(preds_df)))["daily_expense_pred"].sum()), 2)

    return "ML used", next_day, next_week, next_month


# =========================
# FASTAPI LAYER
# =========================

app = FastAPI(title="SmartBudget ML API", version="1.0.3")

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
    next_month: float


@app.get("/health")
def health():
    return {"ok": True}


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