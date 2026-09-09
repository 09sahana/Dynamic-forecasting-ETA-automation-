import sys
import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config import MODEL_DIR, RAW_DATA_PATH, TARGET_COL, CANCELLED_COL
from src.preprocessing import load_raw_data, preprocess_data

_MODEL_BUNDLE = None

def load_inference_bundle():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        model_path = MODEL_DIR / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Train model first.")
        _MODEL_BUNDLE = joblib.load(model_path)
    return _MODEL_BUNDLE


def log_quantile_crossing(train_no: int | str, station_code: str, p08: float, p50: float, p92: float):
    """
    Component 7.2 Bug B: Log quantile crossing events for chronic/noisy trains.
    Logged to logs/quantile_crossings.jsonl for diagnostics.
    """
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "quantile_crossings.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "train_no": int(train_no) if str(train_no).isdigit() else train_no,
            "station_code": str(station_code),
            "p08": round(float(p08), 2),
            "p50": round(float(p50), 2),
            "p92": round(float(p92), 2),
            "timestamp": datetime.now().isoformat()
        }) + "\n")

def predict_delay_and_eta(
    train_no: int,
    journey_date: str,
    station_code: str,
    raw_df: pd.DataFrame = None,
    previous_station_delay: Optional[float] = None,
    event_type: str = "NONE",
    event_elapsed_min: float = 0.0,
    event_expected_min: float = 0.0,
    **kwargs,
) -> dict:
    """
    Predicts delay, calculated ETA, and canonical prediction intervals [p08, p92] (84% nominal coverage).
    Handles:
    - Cancellation detection
    - Severe delay risk classification
    - Production-grade cold-start fallback with automatic confidence downgrading
    - Live operational event integration (v6 leakage-free: event_elapsed_min + event_expected_min)
    - Anchored cumulative delay integration (v7.3 previous_station_delay)
    - Calibrated confidence scoring

    Parameters
    ----------
    previous_station_delay: live running cumulative delay accumulated since origin departure.
                            If None, falls back to historical raw_df value (batch mode).
    event_type        : e.g. 'CONGESTION', 'TRAIN_HELD', 'NONE' ...
    event_elapsed_min : how long the active event has been running so far (minutes).
                        Pass 0.0 when event_type is 'NONE' or at event start.
    event_expected_min: dispatcher's published duration prior for this incident type.
                        If 0.0 and event_type != 'NONE', auto-filled from lookup table.
    """
    bundle = load_inference_bundle()
    model = bundle.get("model_p50", bundle.get("model"))
    # FIX 2: Canonical interval [p08, p92] (84% coverage) replacing p10/p90
    model_p08 = bundle.get("model_p08", bundle.get("model_p10"))
    model_p92 = bundle.get("model_p92", bundle.get("model_p90"))
    feature_pipeline = bundle["feature_pipeline"]
    feature_cols = bundle["feature_cols"]
    base_feature_cols = bundle.get("base_feature_cols", [c for c in feature_cols if c != 'severe_delay_risk'])
    target_transform = bundle.get("target_transform", "log1p")
    severe_clf = bundle.get("severe_classifier", None)
    conf_model = bundle.get("confidence_model", None)
    conf_cutoffs = bundle.get("confidence_cutoffs", [13.0, 21.5])

    if raw_df is None:
        raw_df = load_raw_data(RAW_DATA_PATH)

    # Filter matching row or trigger cold-start defensive fallback (Fix 3)
    is_cold_start = False
    match = raw_df[(raw_df['train_no'] == train_no) & 
                   (raw_df['journey_date'] == journey_date) & 
                   (raw_df['station_code'] == station_code)]

    if match.empty:
        # Search by train_no and station_code across all dates
        match = raw_df[(raw_df['train_no'] == train_no) & (raw_df['station_code'] == station_code)]
        if match.empty:
            # FIX 3: Cold-start fallback for unseen train_no in production
            is_cold_start = True
            st_match = raw_df[raw_df['station_code'] == station_code]
            if not st_match.empty:
                row = st_match.iloc[0:1].copy()
            else:
                row = raw_df.iloc[0:1].copy()
                row['station_code'] = station_code
            row['train_no'] = train_no
            row['journey_date'] = journey_date
        else:
            row = match.iloc[0:1].copy()
            row['journey_date'] = journey_date
    else:
        row = match.iloc[0:1].copy()

    # Inject live delay signal if provided (v7.3 anchored cumulative delay)
    if previous_station_delay is not None:
        row['previous_station_delay'] = float(previous_station_delay)

    # Inject operational event telemetry (v6 leakage-free fields)
    row['event_type']         = event_type
    row['event_elapsed_min']  = float(event_elapsed_min)
    # Auto-fill expected duration from lookup if caller didn't provide one
    if event_expected_min == 0.0 and event_type != 'NONE':
        from src.event_generator import EXPECTED_DURATION_PRIOR
        row['event_expected_min'] = float(EXPECTED_DURATION_PRIOR.get(event_type, 15))
    else:
        row['event_expected_min'] = float(event_expected_min)

    scheduled_arr_str = str(row['scheduled_arrival'].values[0])

    # Check cancellation status explicitly
    is_cancelled = False
    if CANCELLED_COL in row.columns:
        c_val = row[CANCELLED_COL].values[0]
        is_cancelled = bool(c_val is True or c_val == 1 or str(c_val).lower() == 'true')

    if is_cancelled:
        return {
            "train_no": train_no,
            "journey_date": journey_date,
            "station_code": station_code,
            "scheduled_arrival": scheduled_arr_str,
            "predicted_delay_min": None,
            "predicted_delay_range_min": None,
            "predicted_eta": None,
            "predicted_eta_range": None,
            "confidence": "Low",
            "severe_delay_risk": 0.0,
            "chronic_tier_is_fallback": 1 if is_cold_start else 0,
            "event_type": event_type,
            "event_elapsed_min": float(event_elapsed_min),
            "event_expected_min": float(event_expected_min),
            "is_cancelled": True,
            "status": "CANCELLED (Service not operating)"
        }

    # Preprocess & Feature Engineer single row context
    clean_row = preprocess_data(row, filter_cancelled=False)
    if previous_station_delay is not None:
        clean_row['previous_station_delay'] = float(previous_station_delay)
    clean_row['event_type']         = event_type
    clean_row['event_elapsed_min']  = float(event_elapsed_min)
    clean_row['event_expected_min'] = float(row['event_expected_min'].values[0])
    
    feat_row = feature_pipeline.transform(clean_row)

    # Compute severe delay risk and stack before calling regressors
    severe_delay_risk = 0.0
    if severe_clf is not None:
        severe_delay_risk = round(float(severe_clf.predict_proba(feat_row[base_feature_cols])[0, 1]), 4)
    
    feat_row['severe_delay_risk'] = severe_delay_risk
    X_input = feat_row[feature_cols]

    is_fallback = bool(is_cold_start or (feat_row['chronic_tier_is_fallback'].values[0] == 1))

    # Branch if severe delay risk is critical (>= 0.5)
    if severe_delay_risk >= 0.5:
        return {
            "train_no": train_no,
            "journey_date": journey_date,
            "station_code": station_code,
            "scheduled_arrival": scheduled_arr_str,
            "predicted_delay_min": None,
            "predicted_delay_range_min": [360.0, 360.0],
            "predicted_eta": None,
            "predicted_eta_range": None,
            "confidence": "Low",
            "severe_delay_risk": severe_delay_risk,
            "chronic_tier_is_fallback": 1 if is_fallback else 0,
            "event_type": event_type,
            "event_elapsed_min": float(event_elapsed_min),
            "event_expected_min": float(event_expected_min),
            "is_cancelled": False,
            "status": "Severe delay — magnitude uncertain (>360m data cap boundary)"
        }

    # FIX 1: Point prediction from median regressor (residual correction dropped from default path)
    raw_p50 = float(model.predict(X_input)[0])
    if target_transform == "log1p":
        p50 = float(np.maximum(0, np.expm1(raw_p50)))
    else:
        p50 = float(np.maximum(0, raw_p50))

    # FIX 2: Canonical Quantile interval [p08, p92] (nominal 84% coverage)
    if model_p08 is not None and model_p92 is not None:
        raw_p08 = float(model_p08.predict(X_input)[0])
        raw_p92 = float(model_p92.predict(X_input)[0])
        if target_transform == "log1p":
            p08 = float(np.maximum(0, np.expm1(raw_p08)))
            p92 = float(np.maximum(0, np.expm1(raw_p92)))
        else:
            p08 = float(np.maximum(0, raw_p08))
            p92 = float(np.maximum(0, raw_p92))
    else:
        p08 = max(0.0, p50 - 15.0)
        p92 = p50 + 15.0

    # Live Event Integration: Add active incident duration & uncertainty dynamically
    actual_expected_min = float(row['event_expected_min'].values[0])
    if event_type != 'NONE' and (event_elapsed_min > 0 or actual_expected_min > 0):
        event_impact = max(float(event_elapsed_min), actual_expected_min)
        p50 = p50 + event_impact
        p08 = p08 + float(event_elapsed_min)
        p92 = p92 + event_impact * 1.5

    # Bug B (v7.2): Quantile Crossing Guard and Diagnostic Logging
    if not (p08 <= p50 <= p92):
        log_quantile_crossing(train_no, station_code, p08, p50, p92)
        # Ensure ordered monotonicity: p08 <= p50 <= p92
        sorted_quantiles = sorted([p08, p50, p92])
        p08 = sorted_quantiles[0]
        p50 = sorted_quantiles[1]
        p92 = sorted_quantiles[2]

    p50 = round(p50, 2)
    p08 = round(p08, 2)
    p92 = round(p92, 2)

    # Helper function for ETA calculation
    def calc_eta(arr_str: str, delay_m: float) -> str:
        try:
            arr_t = datetime.strptime(arr_str, "%H:%M:%S")
            return (arr_t + timedelta(minutes=delay_m)).strftime("%H:%M:%S")
        except Exception:
            return f"{arr_str} (+{delay_m} min)"

    predicted_eta = calc_eta(scheduled_arr_str, p50)
    predicted_eta_range = [calc_eta(scheduled_arr_str, p08), calc_eta(scheduled_arr_str, p92)]

    # Calibrated confidence scoring
    if conf_model is not None:
        exp_err = float(conf_model.predict(X_input)[0])
        q33, q66 = conf_cutoffs
        if exp_err <= q33:
            confidence = "High"
        elif exp_err <= q66:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        confidence = "High" if p50 <= 10 else ("Medium" if p50 <= 45 else "Low")

    # Live event uncertainty: active events downgrade High confidence to Medium/Low
    if event_type != 'NONE' and confidence == "High":
        confidence = "Medium"

    # FIX 3: Defensive confidence downgrade for cold-start unseen trains (never produce "High")
    if is_fallback and confidence == "High":
        confidence = "Medium"

    return {
        "train_no": train_no,
        "journey_date": journey_date,
        "station_code": station_code,
        "scheduled_arrival": scheduled_arr_str,
        "predicted_delay_min": p50,
        "predicted_delay_range_min": [p08, p92],
        "interval_width_type": "p08_p92 (84% nominal)",
        "predicted_eta": predicted_eta,
        "predicted_eta_range": predicted_eta_range,
        "confidence": confidence,
        "severe_delay_risk": severe_delay_risk,
        "chronic_tier_is_fallback": 1 if is_fallback else 0,
        "event_type": event_type,
        "event_elapsed_min": float(event_elapsed_min),
        "event_expected_min": float(event_expected_min),
        "is_cancelled": False,
        "status": "ON_TIME" if p50 <= 5 else "DELAYED"
    }

def predict(*args, **kwargs) -> dict:
    """
    Standard prediction interface supporting positional/keyword arguments
    or a single observation dictionary:
        result = predict(current_observation)
    Delegates directly to predict_delay_and_eta.
    """
    if len(args) == 1 and isinstance(args[0], dict):
        return predict_delay_and_eta(**args[0])
    return predict_delay_and_eta(*args, **kwargs)

if __name__ == "__main__":
    raw = load_raw_data(RAW_DATA_PATH)
    valid_rows = raw.dropna(subset=['train_no', 'journey_date', 'station_code'])
    sample_row = valid_rows.iloc[10]
    
    t_no = int(sample_row['train_no'])
    j_date = str(sample_row['journey_date'])
    s_code = str(sample_row['station_code'])
    
    print("Testing sample prediction with current bundle...")
    try:
        res = predict_delay_and_eta(t_no, j_date, s_code, raw_df=raw)
        print("\nPREDICTION RESULT:")
        for k, v in res.items():
            print(f"  {k:26s}: {v}")
    except Exception as e:
        print(f"Prediction failed (model re-train needed for v5): {e}")
