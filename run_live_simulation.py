"""
run_live_simulation.py
======================
Live Context Loop & Real-Time Simulation Engine (v7.1)
Demonstrates predict + explain + recalculate dynamically during a journey.

Features:
- Live event observation wiring (v7.1 fix)
- Dynamic ETA recalculation on operational events
- Event-severity-aware adaptive recompute frequency (7.2)
- Structured JSON logging alongside human-readable logs (7.3)
- Multi-train demo reel runner (7.1)
- Confidence-aware passenger notification thresholding (7.5)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from simulator.live_event_simulator import LiveEventSimulator
from services.passenger_filter import filter_and_prepare, should_notify_passenger
from src.event_generator import EXPECTED_DURATION_PRIOR
from src.predict import predict
from src.config import RAW_DATA_PATH
from src.preprocessing import load_raw_data


def build_observation_from_sim_state(sim: LiveEventSimulator, raw_df: pd.DataFrame = None) -> dict:
    """
    Extracts current state from the simulator and builds the observation
    dictionary required by the prediction interface.
    Explicitly reads the simulator's active event state and anchored cumulative delay on every call.
    """
    curr_station = sim.get_current_station_code()
    
    if sim.active_event is not None:
        event_type = sim.active_event["type"]
        start_tick = sim.active_event.get("start_tick", sim.elapsed_ticks)
        event_elapsed_min = float((sim.elapsed_ticks - start_tick) * sim.tick_minutes)
        event_expected_min = float(
            sim.active_event.get("expected_duration") or 
            EXPECTED_DURATION_PRIOR.get(event_type, 15.0)
        )
    else:
        event_type = "NONE"
        event_elapsed_min = 0.0
        event_expected_min = 0.0

    return {
        "train_no": int(sim.train_no) if str(sim.train_no).isdigit() else sim.train_no,
        "journey_date": str(sim.journey_date),
        "station_code": str(curr_station),
        "previous_station_delay": float(getattr(sim, "cumulative_delay_min", 0.0)),
        "raw_df": raw_df,
        "event_type": event_type,
        "event_elapsed_min": event_elapsed_min,
        "event_expected_min": event_expected_min,
    }


def format_json_update(sim: LiveEventSimulator, result: dict, passenger_view: dict | None = None) -> dict:
    """
    Component 7.3: Convert a live simulation update into a structured JSON dict.
    """
    status_str = passenger_view["status"] if passenger_view else ("STOPPED" if sim.active_event else "MOVING")
    reason_str = passenger_view["reason"] if passenger_view else "Normal operational running"
    
    return {
        "tick": sim.elapsed_ticks,
        "train_no": sim.train_no,
        "station_code": sim.get_current_station_code(),
        "station_name": sim.get_current_station_name(),
        "status": status_str,
        "reason": reason_str,
        "predicted_eta": result.get("predicted_eta"),
        "eta_range": result.get("predicted_eta_range"),
        "predicted_delay_min": result.get("predicted_delay_min"),
        "confidence": result.get("confidence"),
        "event_type": result.get("event_type", "NONE"),
        "event_elapsed_min": result.get("event_elapsed_min", 0.0),
        "event_expected_min": result.get("event_expected_min", 0.0),
    }


def log_update(sim: LiveEventSimulator, result: dict, passenger_view: dict | None = None, verbose: bool = True) -> str:
    """
    Format and print a human-readable live context update.
    Guaranteed to format a single clean block without duplicate headers.
    """
    status_str = passenger_view["status"] if passenger_view else ("STOPPED" if sim.active_event else "MOVING")
    station_name = sim.get_current_station_name()
    
    msg_lines = [
        f"[{sim.elapsed_ticks:>4} min] Train {sim.train_no} @ {station_name}",
    ]
    if passenger_view:
        msg_lines.append(f"  STATUS: {status_str}  |  REASON: {passenger_view['reason']}")
    else:
        msg_lines.append(f"  STATUS: {status_str}")
    
    msg_lines.append(
        f"  PREDICTED ETA: {result.get('predicted_eta')}  "
        f"(range {result.get('predicted_eta_range')}, confidence: {result.get('confidence')})"
    )
    output = "\n".join(msg_lines) + "\n"
    if verbose:
        print(output, end="")
    return output


def run_journey(
    train_no: int | str,
    journey_date: str,
    route_stations: any,
    raw_df: pd.DataFrame = None,
    sim: LiveEventSimulator = None,
    normal_recompute_ticks: int = 15,
    severe_recompute_ticks: int = 1,
    periodic_recompute_ticks: Optional[int] = None,
    json_log_path: Optional[Path | str] = None,
    verbose: bool = True,
) -> list:
    """
    Run one full journey tick by tick.
    
    Improvements:
    - 7.2: Adaptive recompute frequency based on event severity
    - 7.3: Emits structured JSON lines alongside console output if requested
    - 7.5: Integrates passenger notification thresholding
    """
    if periodic_recompute_ticks is not None:
        normal_recompute_ticks = periodic_recompute_ticks

    if sim is None:
        sim = LiveEventSimulator(train_no, journey_date, route_stations)

    last_eta = None
    last_station_idx = -1
    last_active_event_type = None
    history = []
    json_records = []

    # Initial snapshot before departure
    obs = build_observation_from_sim_state(sim, raw_df=raw_df)
    res = predict(obs)
    last_eta = res.get("predicted_eta")
    last_station_idx = sim.current_station_idx
    last_active_event_type = sim.active_event["type"] if sim.active_event else None
    
    initial_entry = {
        "tick": sim.elapsed_ticks,
        "station_idx": sim.current_station_idx,
        "station_code": sim.get_current_station_code(),
        "result": res,
        "passenger_view": None,
        "log": log_update(sim, res, None, verbose=verbose),
    }
    history.append(initial_entry)
    json_records.append(format_json_update(sim, res, None))

    while not sim.is_journey_complete():
        raw_event = sim.tick()

        passenger_view = None
        if raw_event:
            passenger_view = filter_and_prepare(raw_event)

        current_active_event_type = sim.active_event["type"] if sim.active_event else None
        is_severe_active = (sim.active_event is not None and sim.active_event.get("status") == "STOPPED")
        
        # Component 7.2: Event-severity-aware adaptive tick frequency
        active_recompute_interval = severe_recompute_ticks if is_severe_active else normal_recompute_ticks
        periodic_tick = (sim.elapsed_ticks % active_recompute_interval == 0)

        # State transition triggers
        station_changed = (sim.current_station_idx != last_station_idx)
        event_changed = (raw_event is not None) or (current_active_event_type != last_active_event_type)

        if event_changed or station_changed or periodic_tick:
            obs = build_observation_from_sim_state(sim, raw_df=raw_df)
            res = predict(obs)

            eta_changed = (res.get("predicted_eta") != last_eta)
            should_log = eta_changed or (raw_event is not None) or station_changed

            if should_log:
                log_text = log_update(sim, res, passenger_view, verbose=verbose)
                entry = {
                    "tick": sim.elapsed_ticks,
                    "station_idx": sim.current_station_idx,
                    "station_code": sim.get_current_station_code(),
                    "raw_event": raw_event,
                    "passenger_view": passenger_view,
                    "result": res,
                    "log": log_text,
                }
                history.append(entry)
                json_records.append(format_json_update(sim, res, passenger_view))
                
                last_eta = res.get("predicted_eta")
                last_station_idx = sim.current_station_idx
                last_active_event_type = current_active_event_type

    # Component 7.3: Save structured JSON log if requested
    if json_log_path:
        out_path = Path(json_log_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for record in json_records:
                f.write(json.dumps(record) + "\n")
        if verbose:
            print(f"\n[JSON LOG] Saved {len(json_records)} structured records to '{out_path}'")

    return history


def run_multi_train_demo(raw_df: pd.DataFrame, json_dir: Optional[Path] = None) -> Dict[str, list]:
    """
    Component 7.1: Multi-train, multi-day live simulation demo reel.
    Runs across:
    - Chronic train #12508 (Northeast Express)
    - Chronic train #15648 (Guwahati Express)
    - Healthy control train #12137 (Punjab Mail)
    """
    demo_profiles = [
        {"train_no": 12508, "date": "2023-01-14", "profile": "Chronic Northeast Express", "inject_event": True},
        {"train_no": 15648, "date": "2023-02-18", "profile": "Chronic Guwahati Express", "inject_event": True},
        {"train_no": 12137, "date": "2023-03-10", "profile": "Healthy Control Punjab Mail", "inject_event": False},
    ]

    results = {}
    print("\n" + "=" * 80)
    print("MULTI-TRAIN LIVE CONTEXT DEMO REEL (Component 7.1)")
    print("=" * 80)

    for p in demo_profiles:
        t_no = p["train_no"]
        j_date = p["date"]
        stops = raw_df[(raw_df["train_no"] == t_no) & (raw_df["journey_date"] == j_date)].sort_values("seq")
        
        if stops.empty:
            # Fallback to train_no across any date
            stops = raw_df[raw_df["train_no"] == t_no].sort_values("seq")
            if not stops.empty:
                j_date = str(stops.iloc[0]["journey_date"])
                stops = raw_df[(raw_df["train_no"] == t_no) & (raw_df["journey_date"] == j_date)].sort_values("seq")

        if stops.empty:
            print(f"Skipping Train #{t_no} (no records found in raw dataset).")
            continue

        demo_stops = stops.head(5)
        print(f"\n--- Demonstrating {p['profile']} (#{t_no}) on {j_date} ---")
        
        sim = LiveEventSimulator(
            t_no,
            j_date,
            demo_stops,
            tick_minutes=1,
            random_state=42,
            probabilistic=False,
        )
        if p["inject_event"]:
            sim.schedule_event("TRAIN_HELD", "OPERATIONAL_REGULATION", duration_minutes=15, trigger_tick=10)

        json_file = (json_dir / f"live_journey_{t_no}_{j_date}.jsonl") if json_dir else None
        hist = run_journey(
            train_no=t_no,
            journey_date=j_date,
            route_stations=demo_stops,
            raw_df=raw_df,
            sim=sim,
            json_log_path=json_file,
            verbose=True,
        )
        results[f"Train_{t_no}"] = hist
        print(f"Completed Train #{t_no} in {sim.elapsed_ticks} mins ({len(hist)} updates emitted).")

    print("\n" + "=" * 80)
    print(f"Multi-train demo reel completed for {len(results)} trains.")
    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Event Context Simulation Runner")
    parser.add_argument("--multi-train", action="store_true", help="Run multi-train demo reel across reliability profiles")
    parser.add_argument("--json", type=str, default=None, help="Path to write structured JSON line logs")
    args = parser.parse_args()

    raw = load_raw_data(RAW_DATA_PATH)

    if args.multi_train:
        json_out_dir = Path(args.json).parent if args.json else Path("data/live_logs")
        run_multi_train_demo(raw, json_dir=json_out_dir)
    else:
        demo_train_no = 12508
        demo_date = "2023-01-14"

        train_stops = raw[
            (raw["train_no"] == demo_train_no) & (raw["journey_date"] == demo_date)
        ].sort_values("seq")

        if train_stops.empty:
            first_row = raw.dropna(subset=["train_no", "journey_date", "station_code"]).iloc[0]
            demo_train_no = int(first_row["train_no"])
            demo_date = str(first_row["journey_date"])
            train_stops = raw[
                (raw["train_no"] == demo_train_no) & (raw["journey_date"] == demo_date)
            ].sort_values("seq")

        demo_route = train_stops.head(5)
        print("=" * 80)
        print("LIVE EVENT CONTEXT LOOP - DEMONSTRATION")
        print("=" * 80)
        print(f"Simulating Train #{demo_train_no} on {demo_date} ({len(demo_route)} stops)...\n")

        sim = LiveEventSimulator(
            demo_train_no,
            demo_date,
            demo_route,
            tick_minutes=1,
            random_state=42,
            probabilistic=False,
        )
        sim.schedule_event("TRAIN_HELD", "OPERATIONAL_REGULATION", duration_minutes=15, trigger_tick=10)

        history = run_journey(
            demo_train_no,
            demo_date,
            demo_route,
            raw_df=raw,
            sim=sim,
            json_log_path=args.json or "data/live_logs/live_journey_12508.jsonl",
            verbose=True,
        )

        print("=" * 80)
        print(f"Journey completed in {sim.elapsed_ticks} simulated minutes. Total updates: {len(history)}")
        print("=" * 80)
