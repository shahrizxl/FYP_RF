from typing import List, Optional, Tuple, Dict, Any
from datetime import timedelta, date
import calendar
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import logging

# =========================
# CONFIG
# =========================
MIN_UNIQUE_DAYS_FOR_ML = 60

ONE_TIME_FLOOR = 300.0
MIN_RECURRENCE_TO_KEEP = 2
ONE_TIME_ONLY_IF_SINGLE = True

# =========================
# FEATURE LABEL MAP
# =========================
FEATURE_LABELS: Dict[str, str] = {
    "lag_1":           "Yesterday's spending",
    "lag_2":           "2 days ago spending",
    "lag_3":           "3 days ago spending",
    "rolling_mean_3":  "3-day average spend",
    "rolling_std_3":   "3-day spending volatility",
    "rolling_mean_7":  "7-day average spend",
    "rolling_std_7":   "7-day spending volatility",
    "cumsum_7":        "7-day cumulative spend",
    "day_of_week":     "Day of week",
    "is_weekend":      "Weekend flag",
    "month":           "Month of year",
    "day":             "Day of month",
}


# =========================
# HISTORICAL METRICS STORE
# =========================
# In-memory store mapping prediction_id -> record.
# A record contains the original prediction, and optionally the actual spend
# submitted later via /record_actual.
# In production, replace with a database or persistent cache.
_prediction_history: Dict[str, Dict[str, Any]] = {}
logger = logging.getLogger("smartbudget")
logging.basicConfig(level=logging.INFO)


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


def heuristic_v3_spike_safe(
    daily: pd.Series,
    anchor: Optional[pd.Timestamp] = None,
) -> Tuple[float, float, float]:
    if daily is None or daily.empty:
        return 0.0, 0.0, 0.0

    daily = daily.sort_index().astype(float)
    unique_days = int(daily.index.nunique())
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
    df = raw.copy()
    floor_th = float(ONE_TIME_FLOOR)
    df["is_one_time"] = False

    big_mask = df["amount"] >= floor_th
    if not big_mask.any():
        return df, (floor_th, 0.0)

    if "category" in df.columns:
        cat_counts = (
            df.loc[big_mask, "category"]
            .fillna("unknown")
            .str.strip()
            .str.lower()
            .value_counts()
        )
        rare_cats = set(cat_counts[cat_counts < MIN_RECURRENCE_TO_KEEP].index)
        df.loc[
            big_mask
            & df["category"].fillna("unknown").str.strip().str.lower().isin(rare_cats),
            "is_one_time",
        ] = True
    else:
        df.loc[big_mask, "is_one_time"] = True

    return df, (floor_th, 0.0)


# =========================
# TRAINING WINDOW
# =========================
def _prev_six_full_months_window(anchor_start: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    anchor_start = pd.Timestamp(anchor_start).normalize()
    train_start = (anchor_start - pd.DateOffset(months=6)).replace(day=1)
    train_end = anchor_start - pd.Timedelta(days=1)
    return train_start, train_end


# =========================
# ML ENGINE
# =========================
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
    series_df["rolling_std_3"] = (
        series_df["daily_expense"].rolling(3).std(ddof=0).shift(1).fillna(0.0)
    )
    series_df["rolling_mean_7"] = series_df["daily_expense"].rolling(7).mean().shift(1)
    series_df["rolling_std_7"] = (
        series_df["daily_expense"].rolling(7).std(ddof=0).shift(1).fillna(0.0)
    )
    series_df["cumsum_7"] = series_df["daily_expense"].rolling(7).sum().shift(1)

    series_df.dropna(inplace=True)
    if len(series_df) < 10:
        return pd.DataFrame(), "No data."

    return series_df, None


def train_and_evaluate_rf(
    series_df: pd.DataFrame,
    fallback_tol: float = 1.2,
    n_splits: int = 3,
) -> Tuple[Optional[RandomForestRegressor], List[str], pd.DataFrame]:
    feature_cols = [c for c in series_df.columns if c != "daily_expense"]
    X_all = series_df[feature_cols]
    y_all = series_df["daily_expense"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_all))

    if len(splits) == 0 or len(splits[-1][1]) < 5:
        split_idx = int(len(series_df) * 0.8)
        if split_idx < 5 or (len(series_df) - split_idx) < 3:
            model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
            model.fit(X_all, y_all)
            metrics_df = pd.DataFrame(
                [
                    {
                        "model": "RandomForest",
                        "mae": 0.0,
                        "rmse": 0.0,
                        "r2": 0.0,
                        "accepted": True,
                        "note": "Too little data for CV — accepted without evaluation",
                    }
                ]
            )
            return model, feature_cols, metrics_df
        splits = [([*range(split_idx)], [*range(split_idx, len(series_df))])]

    all_mae, all_rmse, all_r2, all_baseline_mae = [], [], [], []

    for train_idx, test_idx in splits:
        X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_te, y_te = X_all.iloc[test_idx], y_all.iloc[test_idx]

        fold_model = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
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

    metrics_df = pd.DataFrame(
        [
            {
                "model": "RandomForest",
                "mae": round(avg_mae, 4),
                "rmse": round(avg_rmse, 4),
                "r2": round(avg_r2, 4),
                "baseline_mae": round(avg_baseline_mae, 4),
                "accepted": bool(accepted),
                "cv_folds": len(splits),
            }
        ]
    )

    final_model = RandomForestRegressor(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
    )
    final_model.fit(X_all, y_all)

    return final_model, feature_cols, metrics_df


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
        std3 = float(np.std(tail3, ddof=0)) if len(tail3) > 1 else 0.0
        mean7 = float(np.mean(tail7)) if tail7 else 0.0
        std7 = float(np.std(tail7, ddof=0)) if len(tail7) > 1 else 0.0
        sum7 = float(np.sum(tail7)) if tail7 else 0.0

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
        history.append(pred)

    return pd.DataFrame({"daily_expense_pred": predictions}, index=future_dates)


# ============================================================
# NEW: EXPLAINABILITY HELPERS
# ============================================================

def extract_feature_importance(
    model: RandomForestRegressor,
    features: List[str],
    top_n: int = 6,
) -> List[Dict[str, Any]]:
    """
    Extract and rank feature importances from a trained RandomForest.
    Returns top_n features with human-readable labels and normalised % scores.
    """
    importances = model.feature_importances_          # shape: (n_features,)
    total = float(importances.sum()) or 1.0

    ranked = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    result = []
    for feat, imp in ranked:
        label = FEATURE_LABELS.get(feat, feat.replace("_", " ").capitalize())
        result.append(
            {
                "feature": feat,
                "label": label,
                "importance": round(float(imp), 6),
                "importance_pct": round(float(imp) / total * 100, 1),
            }
        )
    return result


def compute_weekend_influence(
    series_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare mean daily spend on weekdays vs weekends to quantify weekend effect.
    Returns a dict with weekday_avg, weekend_avg, and a direction label.
    """
    if series_df is None or series_df.empty or "is_weekend" not in series_df.columns:
        return {}

    weekday_vals = series_df.loc[series_df["is_weekend"] == 0, "daily_expense"]
    weekend_vals = series_df.loc[series_df["is_weekend"] == 1, "daily_expense"]

    weekday_avg = float(weekday_vals.mean()) if len(weekday_vals) else 0.0
    weekend_avg = float(weekend_vals.mean()) if len(weekend_vals) else 0.0

    if weekday_avg > 0:
        pct_diff = (weekend_avg - weekday_avg) / weekday_avg * 100
    else:
        pct_diff = 0.0

    direction = "higher" if pct_diff > 2 else ("lower" if pct_diff < -2 else "similar")

    return {
        "weekday_avg": round(weekday_avg, 2),
        "weekend_avg": round(weekend_avg, 2),
        "pct_diff": round(pct_diff, 1),
        "direction": direction,
        "summary": (
            f"Weekend spending is {abs(pct_diff):.0f}% {direction} than weekdays"
            if direction != "similar"
            else "Weekend and weekday spending are similar"
        ),
    }


def compute_category_impact(
    df_raw: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Summarise total spend by category, returning top_n contributors with % share.
    Works on raw (unfiltered) expense rows so one-time spikes are included for
    user visibility — users can then understand which categories drive their bill.
    """
    if df_raw is None or df_raw.empty or "category" not in df_raw.columns:
        return []

    df = df_raw.copy()
    df["category"] = df["category"].fillna("Uncategorised").str.strip().str.title()
    totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)

    grand_total = float(totals.sum()) or 1.0
    result = []
    for cat, amt in totals.head(top_n).items():
        result.append(
            {
                "category": str(cat),
                "total": round(float(amt), 2),
                "share_pct": round(float(amt) / grand_total * 100, 1),
            }
        )
    return result


def build_explanation_text(
    feature_importances: List[Dict[str, Any]],
    weekend_influence: Dict[str, Any],
    category_impact: List[Dict[str, Any]],
    method: str,
) -> str:
    """
    Compose a plain-English paragraph explaining the forecast drivers.
    """
    if not feature_importances:
        return f"Forecast generated using {method}."

    top_feat = feature_importances[0]
    second_feat = feature_importances[1] if len(feature_importances) > 1 else None

    parts = [
        f"The forecast is driven primarily by {top_feat['label']} "
        f"({top_feat['importance_pct']:.0f}% of model weight)"
    ]

    if second_feat:
        parts.append(
            f"followed by {second_feat['label']} "
            f"({second_feat['importance_pct']:.0f}%)"
        )

    if weekend_influence:
        parts.append(weekend_influence["summary"].lower())

    if category_impact:
        top_cat = category_impact[0]
        parts.append(
            f"and {top_cat['category']} accounts for the largest share of spending "
            f"({top_cat['share_pct']:.0f}%)"
        )

    return ". ".join(parts) + "."


# =========================
# ORCHESTRATION
# =========================
def predict_all_horizons_multi(
    transactions_df: pd.DataFrame,
    days: int = 30,
    anchor_year: Optional[int] = None,
    anchor_month: Optional[int] = None,
) -> Tuple[str, float, float, float, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: (message, next_day, next_week, this_month_forecast, metrics_list, explainability)
    explainability keys: feature_importances, weekend_influence, category_impact, explanation_text
    """
    empty_explain: Dict[str, Any] = {
        "feature_importances": [],
        "weekend_influence": {},
        "category_impact": [],
        "explanation_text": "",
    }

    if transactions_df is None or transactions_df.empty:
        return "No expense data available.", 0.0, 0.0, 0.0, [], empty_explain

    raw = transactions_df.copy()
    if "type" in raw.columns:
        raw["type"] = raw["type"].astype(str).str.strip().str.lower()
        raw = raw[raw["type"] == "expense"]

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    raw = raw.dropna(subset=["date", "amount"])
    raw["amount"] = raw["amount"].abs()

    if raw.empty:
        return "No expense data available.", 0.0, 0.0, 0.0, [], empty_explain

    today_ts = pd.Timestamp(date.today()).normalize()
    latest = raw["date"].max()

    if anchor_year is not None and anchor_month is not None:
        ay, am = int(anchor_year), int(anchor_month)
    else:
        ay, am = today_ts.year, today_ts.month

    last_day = calendar.monthrange(ay, am)[1]
    anchor_start = pd.Timestamp(date(ay, am, 1))
    anchor_end = pd.Timestamp(date(ay, am, last_day))

    actual_end = min(today_ts, anchor_end)
    spent_so_far = float(
        raw[(raw["date"] >= anchor_start) & (raw["date"] <= actual_end)]["amount"].sum()
    )

    start_pred = max(today_ts + pd.Timedelta(days=1), anchor_start)
    remaining_days = (
        max(0, int((anchor_end - start_pred).days) + 1) if start_pred <= anchor_end else 0
    )

    train_start, train_end = _prev_six_full_months_window(anchor_start)
    df6_anchor = raw[(raw["date"] >= train_start) & (raw["date"] <= train_end)]

    if len(df6_anchor) >= 10:
        df = df6_anchor
    else:
        cutoff = latest - pd.DateOffset(months=6)
        df6_latest = raw[raw["date"] >= cutoff]
        df = df6_latest if len(df6_latest) >= 10 else raw

    # Category impact computed on full training window (before one-time filter)
    category_impact = compute_category_impact(df)

    df_marked, _ = mark_one_time_transactions(df)
    df_model = df_marked[~df_marked["is_one_time"]].copy()

    daily_model = _daily_series(df_model) if not df_model.empty else pd.Series(dtype=float)
    heuristic_anchor = pd.Timestamp(train_end).normalize() if not daily_model.empty else None
    fallback_day, fallback_week, _ = heuristic_v3_spike_safe(daily_model, anchor=heuristic_anchor)

    unique_days = int(daily_model.index.nunique()) if len(daily_model) else 0

    next_day = float(fallback_day)
    next_week = float(fallback_week)
    predicted_remaining = float(fallback_day * remaining_days) if remaining_days > 0 else 0.0
    msg = "Using average"
    metrics_list: List[Dict[str, Any]] = []
    feature_importances: List[Dict[str, Any]] = []
    weekend_influence: Dict[str, Any] = {}
    series_df_used: Optional[pd.DataFrame] = None

    if unique_days >= MIN_UNIQUE_DAYS_FOR_ML:
        series_df, err = aggregate_expenses(df_model)
        if not err and series_df is not None and not series_df.empty:
            series_df_used = series_df

            model, features, metrics_df = train_and_evaluate_rf(series_df)
            metrics_list = metrics_df.to_dict(orient="records")

            if model is not None and bool(metrics_df.iloc[0]["accepted"]):
                # --- EXPLAINABILITY: feature importances & weekend effect ---
                feature_importances = extract_feature_importance(model, features)
                weekend_influence = compute_weekend_influence(series_df)

                last_hist = series_df.index[-1].normalize()
                days_to_anchor_end = max(0, int((anchor_end - last_hist).days))
                days_to_next_week = max(
                    0, int((today_ts + pd.Timedelta(days=7) - last_hist).days)
                )
                needed_days = min(90, max(days, days_to_anchor_end, days_to_next_week))

                preds_df = predict_future(model, features, series_df, days=needed_days)

                if not preds_df.empty:
                    msg = "RandomForest-based forecast"
                    real_tomorrow = today_ts + pd.Timedelta(days=1)
                    future_preds = preds_df[preds_df.index >= real_tomorrow]

                    if not future_preds.empty:
                        next_day = round(float(future_preds.iloc[0]["daily_expense_pred"]), 2)
                        next_week = round(
                            float(future_preds.head(7)["daily_expense_pred"].sum()), 2
                        )

                    if remaining_days > 0:
                        preds_in_month = preds_df.loc[
                            (preds_df.index >= start_pred) & (preds_df.index <= anchor_end)
                        ]
                        predicted_remaining = float(
                            preds_in_month["daily_expense_pred"].sum()
                        )
            else:
                msg = "ML model not reliable (failed MAE/R2). Using averages."
        else:
            if series_df_used is None and not daily_model.empty:
                series_df_used = None

    this_month_forecast = round(spent_so_far + predicted_remaining, 2)

    if anchor_end < today_ts:
        msg = "Past month — showing actual total"
        actual = float(
            raw[(raw["date"] >= anchor_start) & (raw["date"] <= anchor_end)]["amount"].sum()
        )
        this_month_forecast = round(actual, 2)

    # Build explainability block
    explanation_text = build_explanation_text(
        feature_importances, weekend_influence, category_impact, msg
    )
    explainability: Dict[str, Any] = {
        "feature_importances": feature_importances,
        "weekend_influence": weekend_influence,
        "category_impact": category_impact,
        "explanation_text": explanation_text,
    }

    return msg, next_day, next_week, this_month_forecast, metrics_list, explainability


# =========================
# FASTAPI
# =========================
app = FastAPI(title="SmartBudget ML API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request / response models ----------

class TransactionIn(BaseModel):
    date: str = Field(..., description="ISO date string")
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
            raise ValueError(
                f"type must be 'expense', 'income', or 'transfer', got {v!r}"
            )
        return v


class PredictRequest(BaseModel):
    transactions: List[TransactionIn] = Field(..., min_length=1)
    days: int = Field(default=30, ge=1, le=365)
    anchor_year: Optional[int] = Field(default=None, ge=2000, le=2100)
    anchor_month: Optional[int] = Field(default=None, ge=1, le=12)


class FeatureImportanceItem(BaseModel):
    feature: str
    label: str
    importance: float
    importance_pct: float


class WeekendInfluence(BaseModel):
    weekday_avg: float
    weekend_avg: float
    pct_diff: float
    direction: str
    summary: str


class CategoryImpact(BaseModel):
    category: str
    total: float
    share_pct: float


class Explainability(BaseModel):
    feature_importances: List[FeatureImportanceItem]
    weekend_influence: Optional[WeekendInfluence]
    category_impact: List[CategoryImpact]
    explanation_text: str


class PredictResponse(BaseModel):
    prediction_id: str                           # NEW — use to record actual later
    message: str
    next_day: float
    next_week: float
    next_month: float
    metrics: Optional[List[Dict[str, Any]]] = None
    explainability: Optional[Explainability] = None


# ---------- historical accuracy models ----------

class RecordActualRequest(BaseModel):
    prediction_id: str
    actual_spend: float = Field(..., ge=0)
    period_label: Optional[str] = None          # e.g. "May 2026"


class AccuracyRecord(BaseModel):
    prediction_id: str
    period_label: Optional[str]
    predicted: float
    actual: Optional[float]
    error: Optional[float]          # actual - predicted
    abs_error: Optional[float]
    pct_error: Optional[float]      # signed %, actual vs predicted
    method: str
    created_at: str


class AccuracyResponse(BaseModel):
    records: List[AccuracyRecord]
    mean_abs_error: Optional[float]
    mean_pct_error: Optional[float]
    total_records: int
    evaluated_records: int          # records that have an actual_spend


# ---------- endpoints ----------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    msg, next_day, next_week, next_month, metrics_list, explainability = (
        predict_all_horizons_multi(
            df,
            days=req.days,
            anchor_year=req.anchor_year,
            anchor_month=req.anchor_month,
        )
    )

    pred_id = str(uuid.uuid4())

    # Store prediction in history
    _prediction_history[pred_id] = {
        "prediction_id": pred_id,
        "period_label": (
            f"{req.anchor_year}-{req.anchor_month:02d}"
            if req.anchor_year and req.anchor_month
            else date.today().strftime("%Y-%m")
        ),
        "predicted": float(next_month),
        "actual": None,
        "method": msg,
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    # Coerce explainability dict -> Pydantic model
    expl_model: Optional[Explainability] = None
    if explainability:
        wi = explainability.get("weekend_influence")
        expl_model = Explainability(
            feature_importances=[
                FeatureImportanceItem(**f) for f in explainability["feature_importances"]
            ],
            weekend_influence=WeekendInfluence(**wi) if wi else None,
            category_impact=[
                CategoryImpact(**c) for c in explainability["category_impact"]
            ],
            explanation_text=explainability["explanation_text"],
        )

    return PredictResponse(
        prediction_id=pred_id,
        message=msg,
        next_day=float(next_day),
        next_week=float(next_week),
        next_month=float(next_month),
        metrics=metrics_list,
        explainability=expl_model,
    )


# @app.post("/record_actual")
# def record_actual(req: RecordActualRequest) -> Dict[str, Any]:
#     """
#     Call this endpoint once the anchor period closes and you know the real spend.
#     Stores the actual alongside the prediction so /accuracy can compute error.
#     """
#     record = _prediction_history.get(req.prediction_id)
#     if record is None:
#         return {"error": f"prediction_id {req.prediction_id!r} not found"}

#     predicted = record["predicted"]
#     actual = float(req.actual_spend)
#     error = actual - predicted
#     abs_error = abs(error)
#     pct_error = (error / predicted * 100) if predicted else None

#     record["actual"] = actual
#     record["error"] = round(error, 2)
#     record["abs_error"] = round(abs_error, 2)
#     record["pct_error"] = round(pct_error, 1) if pct_error is not None else None
#     if req.period_label:
#         record["period_label"] = req.period_label

#     return {
#         "prediction_id": req.prediction_id,
#         "predicted": predicted,
#         "actual": actual,
#         "error": round(error, 2),
#         "pct_error": round(pct_error, 1) if pct_error is not None else None,
#         "message": "Actual spend recorded.",
#     }


@app.get("/accuracy", response_model=AccuracyResponse)
def accuracy() -> AccuracyResponse:
    """
    Return all historical prediction vs actual records with aggregate metrics.
    """
    records = []
    abs_errors = []
    pct_errors = []

    for rec in _prediction_history.values():
        has_actual = rec.get("actual") is not None
        records.append(
            AccuracyRecord(
                prediction_id=rec["prediction_id"],
                period_label=rec.get("period_label"),
                predicted=rec["predicted"],
                actual=rec.get("actual"),
                error=rec.get("error"),
                abs_error=rec.get("abs_error"),
                pct_error=rec.get("pct_error"),
                method=rec.get("method", "unknown"),
                created_at=rec.get("created_at", ""),
            )
        )
        if has_actual:
            if rec.get("abs_error") is not None:
                abs_errors.append(rec["abs_error"])
            if rec.get("pct_error") is not None:
                pct_errors.append(rec["pct_error"])

    # Sort newest-first by created_at
    records.sort(key=lambda r: r.created_at, reverse=True)

    mean_abs = round(float(np.mean(abs_errors)), 2) if abs_errors else None
    mean_pct = round(float(np.mean(pct_errors)), 1) if pct_errors else None

    return AccuracyResponse(
        records=records,
        mean_abs_error=mean_abs,
        mean_pct_error=mean_pct,
        total_records=len(records),
        evaluated_records=len(abs_errors),
    )

def print_accuracy_report(history: Dict[str, Dict[str, Any]]) -> None:
    """
    Print full ML evaluation metrics to terminal logs.
    """

    completed = [
        r for r in history.values()
        if r.get("actual_spend") is not None
    ]

    if not completed:
        logger.info("No completed prediction evaluations yet.")
        return

    actuals = np.array(
        [float(r["actual_spend"]) for r in completed],
        dtype=float,
    )

    preds = np.array(
        [float(r["predicted"]) for r in completed],
        dtype=float,
    )

    abs_errors = np.abs(actuals - preds)

    # ─────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────
    mae = mean_absolute_error(actuals, preds)

    rmse = np.sqrt(mean_squared_error(actuals, preds))

    # avoid divide-by-zero
    nonzero_mask = actuals != 0

    if nonzero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (actuals[nonzero_mask] - preds[nonzero_mask])
                    / actuals[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = 0.0

    try:
        r2 = r2_score(actuals, preds)
    except Exception:
        r2 = 0.0

    # ─────────────────────────────────────────────
    # TERMINAL REPORT
    # ─────────────────────────────────────────────
    logger.info("")
    logger.info("============================================================")
    logger.info(" SMARTBUDGET ML — ACCURACY REPORT ")
    logger.info("============================================================")

    for r in completed[-5:]:
        pred = float(r["predicted"])
        actual = float(r["actual_spend"])
        err = actual - pred
        pct = (err / actual * 100) if actual != 0 else 0.0

        logger.info(
            f"[{r.get('period_label', '-')}] "
            f"Predicted=RM {pred:.2f} | "
            f"Actual=RM {actual:.2f} | "
            f"Error={err:+.2f} ({pct:+.1f}%)"
        )

    logger.info("------------------------------------------------------------")
    logger.info(" AGGREGATE METRICS ")
    logger.info("------------------------------------------------------------")
    logger.info(f"Records Evaluated : {len(completed)}")
    logger.info(f"MAE               : RM {mae:.2f}")
    logger.info(f"RMSE              : RM {rmse:.2f}")
    logger.info(f"MAPE              : {mape:.2f}%")
    logger.info(f"R² Score          : {r2:.4f}")

    logger.info("============================================================")
    logger.info("")

@app.post("/record_actual")
def record_actual(req: RecordActualRequest) -> Dict[str, Any]:
    record = _prediction_history.get(req.prediction_id)
    if record is None:
        return {"error": f"prediction_id {req.prediction_id!r} not found"}

    predicted = record["predicted"]
    actual = float(req.actual_spend)
    error = actual - predicted
    abs_error = abs(error)
    pct_error = (error / predicted * 100) if predicted else None

    record["actual"] = actual
    record["error"] = round(error, 2)
    record["abs_error"] = round(abs_error, 2)
    record["pct_error"] = round(pct_error, 1) if pct_error is not None else None
    if req.period_label:
        record["period_label"] = req.period_label

    print_accuracy_report(_prediction_history)   # ← add this line

    return {
        "prediction_id": req.prediction_id,
        "predicted": predicted,
        "actual": actual,
        "error": round(error, 2),
        "pct_error": round(pct_error, 1) if pct_error is not None else None,
        "message": "Actual spend recorded.",
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}