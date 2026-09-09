"""
api/journey_aggregator.py
==========================
Chains predictions across all stations in a train's route
to produce a progressive timetable and destination ETA.
"""

from typing import Dict, List, Optional
import pandas as pd

from src.config import RAW_DATA_PATH
from src.predict import predict_delay_and_eta
from src.preprocessing import load_raw_data

_CACHED_ROUTES: Dict[int, List[Dict]] = {}
_RAW_DF: Optional[pd.DataFrame] = None


def _get_raw_df() -> pd.DataFrame:
    global _RAW_DF
    if _RAW_DF is None:
        _RAW_DF = load_raw_data(RAW_DATA_PATH)
    return _RAW_DF


def get_train_route(train_no: int) -> List[Dict]:
    """Retrieve the ordered static station route for the given train."""
    if train_no in _CACHED_ROUTES:
        return _CACHED_ROUTES[train_no]

    df = _get_raw_df()
    train_rows = df[df["train_no"] == train_no]
    if train_rows.empty:
        # Fallback to train 12952 if unseen train
        train_rows = df[df["train_no"] == 12952]

    unique_stations = (
        train_rows[["seq", "station_code", "station_name", "distance_km", "scheduled_arrival"]]
        .drop_duplicates(subset=["station_code"])
        .sort_values("seq")
        .to_dict(orient="records")
    )
    _CACHED_ROUTES[train_no] = unique_stations
    return unique_stations


def aggregate_journey_eta(
    train_no: int,
    journey_date: str,
    current_station_code: Optional[str] = None,
    current_delay_min: float = 0.0,
    event_type: str = "NONE",
    event_elapsed_min: float = 0.0,
) -> Dict:
    """
    Progressively loops predict() across route halts, chaining predicted delay forward.
    """
    raw_df = _get_raw_df()
    route = get_train_route(train_no)

    if not route:
        return {
            "train_no": train_no,
            "journey_date": journey_date,
            "origin_station": "UNKNOWN",
            "destination_station": "UNKNOWN",
            "total_distance_km": 0.0,
            "stations": [],
            "final_predicted_eta": None,
            "final_predicted_delay_min": None,
        }

    origin = route[0]["station_code"]
    destination = route[-1]["station_code"]
    total_dist = float(route[-1]["distance_km"])

    # Find starting station index
    start_idx = 0
    if current_station_code:
        for idx, stop in enumerate(route):
            if str(stop["station_code"]).upper() == current_station_code.upper():
                start_idx = idx
                break

    station_results = []
    running_delay = float(current_delay_min)

    for i in range(len(route)):
        stop = route[i]
        st_code = stop["station_code"]
        st_name = stop.get("station_name", st_code)
        dist_km = float(stop["distance_km"])
        sched_arr = str(stop["scheduled_arrival"])

        if i < start_idx:
            # Already passed stations
            station_results.append({
                "seq": int(stop["seq"]),
                "station_code": st_code,
                "station_name": st_name,
                "distance_km": dist_km,
                "scheduled_arrival": sched_arr,
                "predicted_delay_min": 0.0,
                "predicted_eta": sched_arr,
                "confidence": "Departed",
                "severe_delay_risk": 0.0,
            })
            continue

        # Upcoming stations: predict with chained delay
        pred = predict_delay_and_eta(
            train_no=train_no,
            journey_date=journey_date,
            station_code=st_code,
            raw_df=raw_df,
            previous_station_delay=running_delay,
            event_type=event_type if i == start_idx else "NONE",
            event_elapsed_min=event_elapsed_min if i == start_idx else 0.0,
        )

        p_delay = pred.get("predicted_delay_min")
        p_eta = pred.get("predicted_eta")
        p_conf = pred.get("confidence", "Medium")
        p_risk = pred.get("severe_delay_risk", 0.0)

        # Chain forward for next stop
        if p_delay is not None:
            running_delay = float(p_delay)

        station_results.append({
            "seq": int(stop["seq"]),
            "station_code": st_code,
            "station_name": st_name,
            "distance_km": dist_km,
            "scheduled_arrival": sched_arr,
            "predicted_delay_min": p_delay,
            "predicted_eta": p_eta,
            "confidence": p_conf,
            "severe_delay_risk": p_risk,
        })

    final_eta = station_results[-1]["predicted_eta"] if station_results else None
    final_delay = station_results[-1]["predicted_delay_min"] if station_results else None

    return {
        "train_no": train_no,
        "journey_date": journey_date,
        "origin_station": origin,
        "destination_station": destination,
        "total_distance_km": total_dist,
        "stations": station_results,
        "final_predicted_eta": final_eta,
        "final_predicted_delay_min": final_delay,
    }
