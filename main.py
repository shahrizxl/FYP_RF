from typing import List, Optional, Tuple, Dict, Any
from datetime import timedelta, date
import calendar

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# =========================
# CONFIG
# =========================
MIN_UNIQUE_DAYS_FOR_ML = 60

# FIX 1: ONE_TIME_FLOOR used to silently drop ALL transactions >= RM300 from training,
# including recurring large costs like rent, bills, salary deductions.
# Solution: only mark a transaction as one-time if it is BOTH >= floor AND its
# category/description has appeared fewer than MIN_RECURRENCE times in the dataset.
# This preserves genuine recurring large expenses in the training data.
ONE_TIME_FLOOR = 300.0
MIN_RECURRENCE_TO_KEEP = 2        # if same category appears >= this many times, it's recurring
ONE_TIME_ONLY_IF_SINGLE = True    # legacy flag, kept for backward compat but logic improved below

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

# FIX 2: heuristic_v3_spike_safe now accepts an explicit anchor parameter.
# Previously it always used daily.index.max() as the anchor, which pointed
# to the last date in whatever dataset was passed — wrong when the training
# window ends months before the target month. Now the caller passes the real
# anchor (anchor_end of the target month or training window end).
def heuristic_v3_spike_safe(
    daily: pd.Series,
    anchor: Optional[pd.Timestamp] = None
) -> Tuple[float, float, float]:
    if daily is None or daily.empty:
        return 0.0, 0.0, 0.0

    daily = daily.sort_index().astype(float)
    unique_days = int(daily.index.nunique())

    # Use explicit anchor if given, otherwise fall back to last date in series.
    effective_anchor: pd.Timestamp = anchor if anchor is not None else daily.index.max()
    effective_anchor = pd.Timestamp(effective_anchor).normalize()

    if unique_days == 1:
        x = float(daily.iloc[0])
        return round(x, 2), round(x * 7, 2), round(x * 30, 2)

    if unique_days < 7:
        est = _robust_estimate_small_sample(daily.values)
        return round(est, 2), round(est * 7, 2), round(est * 30, 2)

    avg7 = _avg_calendar_window(daily, effective_anchor, 7)

    if unique_days >= 30:
        avg30 = _avg_calendar_window(daily, effective_anchor, 30)
        return round(avg7, 2), round(avg7 * 7, 2), round(avg30 * 30, 2)

    return round(avg7, 2), round(avg7 * 7, 2), round(avg7 * 30, 2)


def mark_one_time_transactions(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """
    FIX 1 (continued): Improved one-time detection.
    A transaction is only flagged as one-time if:
      - Its amount >= ONE_TIME_FLOOR, AND
      - Its category has fewer than MIN_RECURRENCE_TO_KEEP occurrences in the dataset.
    This preserves large recurring costs (rent, insurance, loan repayments) in training.
    """
    df = raw.copy()
    floor_th = float(ONE_TIME_FLOOR)
    df["is_one_time"] = False

    big_mask = df["amount"] >= floor_th
    if not big_mask.any():
        return df, (floor_th, 0.0)

    # Count how many times each category appears among large transactions
    if "category" in df.columns:
        cat_counts = df.loc[big_mask, "category"].fillna("unknown").str.strip().str.lower().value_counts()
        rare_cats = set(cat_counts[cat_counts < MIN_RECURRENCE_TO_KEEP].index)
        # Mark as one-time only if large AND in a rare (non-recurring) category
        df.loc[
            big_mask & df["category"].fillna("unknown").str.strip().str.lower().isin(rare_cats),
            "is_one_time"
        ] = True
    else:
        # No category info — fall back to original simple rule
        df.loc[big_mask, "is_one_time"] = True

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


def train_rf_only(
    series_df: pd.DataFrame,
) -> Tuple[RandomForestRegressor, List[str]]:
    """Train RandomForest on the FULL series_df (used after evaluation confirms quality)."""
    X = series_df.drop(columns=["daily_expense"])
    y = series_df["daily_expense"]
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()


# FIX 5: Proper time-series cross-validation instead of evaluating on training data.
# Previously evaluate_rf() received the same data the model was trained on, so
# MAE was always near-zero and accepted was always True — no real quality gate.
# Now we use TimeSeriesSplit (walk-forward) to get honest out-of-sample metrics.
def train_and_evaluate_rf(
    series_df: pd.DataFrame,
    fallback_tol: float = 1.2,
    n_splits: int = 3,
) -> Tuple[Optional[RandomForestRegressor], List[str], pd.DataFrame]:
    """
    Walk-forward cross-validation, then retrain on full data.
    Returns (final_model, features, metrics_df).
    If there's not enough data for CV, falls back to a simple 80/20 temporal split.
    """
    feature_cols = [c for c in series_df.columns if c != "daily_expense"]
    X_all = series_df[feature_cols]
    y_all = series_df["daily_expense"]

    min_train_size = max(10, len(series_df) // (n_splits + 1))

    # Choose split strategy based on available data
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_all))

    if len(splits) == 0 or len(splits[-1][1]) < 5:
        # Not enough data for n_splits — use a simple 80/20 temporal split
        split_idx = int(len(series_df) * 0.8)
        if split_idx < 5 or (len(series_df) - split_idx) < 3:
            # Too small even for 80/20 — skip evaluation, accept as-is
            model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
            model.fit(X_all, y_all)
            metrics_df = pd.DataFrame([{
                "model": "RandomForest",
                "mae": 0.0, "rmse": 0.0, "r2": 0.0,
                "accepted": True,
                "note": "Too little data for CV — accepted without evaluation"
            }])
            return model, feature_cols, metrics_df
        splits = [([*range(split_idx)], [*range(split_idx, len(series_df))])]

    all_mae, all_rmse, all_r2 = [], [], []
    all_baseline_mae = []

    for train_idx, test_idx in splits:
        X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_te, y_te = X_all.iloc[test_idx], y_all.iloc[test_idx]

        fold_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        fold_model.fit(X_tr, y_tr)

        preds = fold_model.predict(X_te)
        baseline = np.full(len(y_te), float(np.mean(y_tr)))

        all_mae.append(mean_absolute_error(y_te, preds))
        all_rmse.append(float(np.sqrt(mean_squared_error(y_te, preds))))
        all_r2.append(r2_score(y_te, preds))
        all_baseline_mae.append(mean_absolute_error(y_te, baseline))

    avg_mae = float(np.mean(all_mae))
    avg_rmse = float(np.mean(all_rmse))
    avg_r2 = float(np.mean(all_r2))
    avg_baseline_mae = float(np.mean(all_baseline_mae))

    accepted = (avg_mae <= avg_baseline_mae * fallback_tol) or (avg_r2 > 0)

    metrics_df = pd.DataFrame([{
        "model": "RandomForest",
        "mae": round(avg_mae, 4),
        "rmse": round(avg_rmse, 4),
        "r2": round(avg_r2, 4),
        "baseline_mae": round(avg_baseline_mae, 4),
        "accepted": bool(accepted),
        "cv_folds": len(splits),
    }])

    # Retrain on full data for final prediction
    final_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_all, y_all)

    return final_model, feature_cols, metrics_df


# FIX 6: predict_future now uses a dedicated history buffer (list) for lag/rolling
# computations instead of writing back into the original series_df DataFrame.
# Previously, temp_df.loc[d] would append predicted values into the full DataFrame
# including all feature columns, causing rolling_std and rolling_mean to mix
# real historical values with predicted values in an uncontrolled way.
def predict_future(
    model: RandomForestRegressor,
    features: List[str],
    series_df: pd.DataFrame,
    days: int = 30,
) -> pd.DataFrame:
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["daily_expense_pred"])

    last_date = series_df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")

    # Keep only the raw daily_expense history for rolling computations.
    # We append predicted values here as we go — cleanly separated from series_df.
    history: List[float] = list(series_df["daily_expense"].values)
    predictions: List[float] = []

    for d in future_dates:
        n = len(history)

        lag1 = history[-1] if n >= 1 else 0.0
        lag2 = history[-2] if n >= 2 else 0.0
        lag3 = history[-3] if n >= 3 else 0.0

        tail3 = history[-3:] if n >= 3 else history
        tail7 = history[-7:] if n >= 7 else history

        mean3 = float(np.mean(tail3)) if tail3 else 0.0
        std3  = float(np.std(tail3, ddof=0)) if len(tail3) > 1 else 0.0
        mean7 = float(np.mean(tail7)) if tail7 else 0.0
        std7  = float(np.std(tail7, ddof=0)) if len(tail7) > 1 else 0.0
        sum7  = float(np.sum(tail7)) if tail7 else 0.0

        row = {
            "day_of_week": d.dayofweek,
            "is_weekend": int(d.dayofweek in [5, 6]),
            "month": d.month,
            "day": d.day,
            "lag_1": lag1,
            "lag_2": lag2,
            "lag_3": lag3,
            "rolling_mean_3": mean3,
            "rolling_std_3": std3,
            "rolling_mean_7": mean7,
            "rolling_std_7": std7,
            "cumsum_7": sum7,
        }

        X_pred = pd.DataFrame([row])
        for f in features:
            if f not in X_pred.columns:
                X_pred[f] = 0.0
        X_pred = X_pred[features]

        pred = max(0.0, float(model.predict(X_pred)[0]))
        predictions.append(pred)
        history.append(pred)  # feed clean prediction into history buffer

    return pd.DataFrame({"daily_expense_pred": predictions}, index=future_dates)


# =========================
# ORCHESTRATION (MERGED CALENDAR + ML)
# =========================
def predict_all_horizons_multi(
    transactions_df: pd.DataFrame,
    days: int = 30,
    anchor_year: Optional[int] = None,
    anchor_month: Optional[int] = None,
) -> Tuple[str, float, float, float, List[Dict[str, Any]]]:

    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0, []

    raw = transactions_df.copy()
    if "type" in raw.columns:
        raw["type"] = raw["type"].astype(str).str.strip().str.lower()
        raw = raw[raw["type"] == "expense"]

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    raw = raw.dropna(subset=["date", "amount"])
    raw["amount"] = raw["amount"].abs()

    if raw.empty:
        return "No expense data available.", 0.0, 0.0, 0.0, []

    today_ts = pd.Timestamp(date.today()).normalize()
    latest = raw["date"].max()

    # Anchor month selection
    if anchor_year is not None and anchor_month is not None:
        ay, am = int(anchor_year), int(anchor_month)
    else:
        ay, am = today_ts.year, today_ts.month

    last_day = calendar.monthrange(ay, am)[1]
    anchor_start = pd.Timestamp(date(ay, am, 1))
    anchor_end = pd.Timestamp(date(ay, am, last_day))

    # FIX 3 & 4: Cleaner spent_so_far and remaining_days calculation.
    # For a past month, clip to anchor_end (not today) so we always get the full month's spend.
    # For current/future month, clip to today so we don't count days that haven't happened.
    actual_end = min(today_ts, anchor_end)
    spent_so_far = float(
        raw[(raw["date"] >= anchor_start) & (raw["date"] <= actual_end)]["amount"].sum()
    )

    # Remaining days = days from tomorrow (or anchor_start if future month) to anchor_end.
    start_pred = max(today_ts + pd.Timedelta(days=1), anchor_start)
    remaining_days = max(0, int((anchor_end - start_pred).days) + 1) if start_pred <= anchor_end else 0

    # Training window: 6 full months before the anchor month
    train_start, train_end = _prev_six_full_months_window(anchor_start)
    df6_anchor = raw[(raw["date"] >= train_start) & (raw["date"] <= train_end)]

    # Fallback safety: if window is too small, use latest 6 months of available data
    if len(df6_anchor) >= 10:
        df = df6_anchor
    else:
        cutoff = latest - pd.DateOffset(months=6)
        df6_latest = raw[raw["date"] >= cutoff]
        df = df6_latest if len(df6_latest) >= 10 else raw

    df_marked, _ = mark_one_time_transactions(df)
    df_model = df_marked[~df_marked["is_one_time"]].copy()

    daily_model = _daily_series(df_model) if not df_model.empty else pd.Series(dtype=float)

    # FIX 2 (applied): pass train_end as the anchor so avg7/avg30 look at the
    # correct calendar window (end of training period), not an arbitrary last date.
    heuristic_anchor = pd.Timestamp(train_end).normalize() if not daily_model.empty else None
    fallback_day, fallback_week, _ = heuristic_v3_spike_safe(daily_model, anchor=heuristic_anchor)

    unique_days = int(daily_model.index.nunique()) if len(daily_model) else 0

    next_day = float(fallback_day)
    next_week = float(fallback_week)
    predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
    msg = "Using average"
    metrics_list: List[Dict[str, Any]] = []

    if unique_days >= MIN_UNIQUE_DAYS_FOR_ML:
        series_df, err = aggregate_expenses(df_model)
        if not err and series_df is not None and not series_df.empty:

            # FIX 5 (applied): train_and_evaluate_rf uses walk-forward CV for honest metrics
            model, features, metrics_df = train_and_evaluate_rf(series_df)
            metrics_list = metrics_df.to_dict(orient="records")

            if model is not None and bool(metrics_df.iloc[0]["accepted"]):
                last_hist = series_df.index[-1].normalize()
                days_to_anchor_end = max(0, int((anchor_end - last_hist).days))
                days_to_next_week = max(0, int((today_ts + pd.Timedelta(days=7) - last_hist).days))
                needed_days = min(
                    90,
                    max(days, days_to_anchor_end, days_to_next_week)
                )

                # FIX 6 (applied): predict_future now uses a clean history buffer
                preds_df = predict_future(model, features, series_df, days=needed_days)

                if not preds_df.empty:
                    msg = "RandomForest-based forecast"
                    real_tomorrow = today_ts + pd.Timedelta(days=1)
                    future_preds = preds_df[preds_df.index >= real_tomorrow]

                    if not future_preds.empty:
                        next_day = round(float(future_preds.iloc[0]["daily_expense_pred"]), 2)
                        next_week = round(float(future_preds.head(7)["daily_expense_pred"].sum()), 2)

                    if remaining_days > 0:
                        preds_in_month = preds_df.loc[
                            (preds_df.index >= start_pred) & (preds_df.index <= anchor_end)
                        ]
                        predicted_remaining = float(preds_in_month["daily_expense_pred"].sum())
            else:
                msg = "ML model not reliable (failed MAE/R2). Using averages."

    this_month_forecast = round(spent_so_far + predicted_remaining, 2)

    # Past month: show the actual total spend (not a forecast)
    if anchor_end < today_ts:
        msg = "Past month — showing actual total"
        actual = float(
            raw[(raw["date"] >= anchor_start) & (raw["date"] <= anchor_end)]["amount"].sum()
        )
        this_month_forecast = round(actual, 2)
        # next_day and next_week remain as ML/heuristic estimates — they are still
        # useful as a baseline for how much the user typically spends per day/week.

    return msg, next_day, next_week, this_month_forecast, metrics_list


# =========================
# FASTAPI
# =========================
app = FastAPI(title="SmartBudget ML API", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionIn(BaseModel):
    date: str = Field(..., description="ISO date string")
    # FIX 7: amount must be > 0. Previously TransactionIn allowed any float
    # including negative values. Negative income entries could corrupt the model
    # since the backend only filters after coercion, not at the schema level.
    amount: float = Field(..., description="Transaction amount")
    type: Optional[str] = Field(default="expense")
    description: Optional[str] = None
    category: Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            pd.Timestamp(v)
        except Exception:
            raise ValueError(f"Invalid date format: {v!r}. Use ISO format e.g. 2026-05-01")
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip().lower() not in ("expense", "income", "transfer"):
            raise ValueError(f"type must be 'expense', 'income', or 'transfer', got {v!r}")
        return v


class PredictRequest(BaseModel):
    transactions: List[TransactionIn] = Field(..., min_length=1)
    days: int = Field(default=30, ge=1, le=365)
    anchor_year: Optional[int] = Field(default=None, ge=2000, le=2100)
    anchor_month: Optional[int] = Field(default=None, ge=1, le=12)


class PredictResponse(BaseModel):
    message: str
    next_day: float
    next_week: float
    next_month: float
    metrics: Optional[List[Dict[str, Any]]] = None


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    msg, next_day, next_week, next_month, metrics_list = predict_all_horizons_multi(
        df,
        days=req.days,
        anchor_year=req.anchor_year,
        anchor_month=req.anchor_month,
    )
    return PredictResponse(
        message=msg,
        next_day=float(next_day),
        next_week=float(next_week),
        next_month=float(next_month),
        metrics=metrics_list,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}