"""
simulator/live_event_simulator.py
=================================
Live Event Simulator (Component 1 of Build Brief v7)
Replays a train's journey tick-by-tick forward in time.

Reuses the FIXED, leakage-free trigger and duration logic from v6 (src/event_generator.py).
Never derives events from the actual delay outcome.
Emits events matching the Phase 2 schema.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.event_generator import (
    EXPECTED_DURATION_PRIOR,
    _compute_trigger_probability,
    _sample_duration_by_type,
    _sample_event_type,
)

EVENT_REASON_MAP = {
    "TRAIN_HELD": "OPERATIONAL_REGULATION",
    "CONGESTION": "OPERATIONAL_REGULATION",
    "SIGNAL_FAILURE": "OPERATIONAL_REGULATION",
    "WEATHER_DISRUPTION": "WEATHER",
    "TRACK_MAINTENANCE": "MAINTENANCE",
    "TRAIN_RESUMED": "RESUMED",
    "NONE": None,
}


class StationStop(dict):
    """
    Representation of a station stop along a train's route.
    Printing it produces the human-friendly station name or code.
    """
    def __getattr__(self, item: str) -> Any:
        return self.get(item)

    def __str__(self) -> str:
        return str(self.get("station_name") or self.get("station_code") or "Station")


class LiveEventSimulator:
    """
    Replays one train's journey tick by tick (e.g. 1 tick = 1 simulated minute),
    probabilistically firing and resolving events using only pre-arrival,
    observable signals.
    """

    def __init__(
        self,
        train_no: int | str,
        journey_date: str,
        route_stations: Any,
        tick_minutes: int = 1,
        random_state: int = 42,
        probabilistic: bool = True,
    ):
        self.train_no = int(train_no) if str(train_no).isdigit() else train_no
        self.journey_date = str(journey_date)
        self.tick_minutes = max(1, int(tick_minutes))
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.probabilistic = probabilistic

        # Normalize route_stations into an ordered list of StationStop objects
        self.route: List[StationStop] = self._normalize_route(route_stations)
        self.current_station_idx: int = 0
        self.elapsed_ticks: int = 0
        self.active_event: Optional[Dict[str, Any]] = None

        # Scheduled/injected events for scenario testing: list of dicts
        self._injected_events: List[Dict[str, Any]] = []

        # Tracking inter-station travel progress
        self._segment_progress_min: float = 0.0
        self._segment_duration_min: float = self._compute_segment_duration(0)
        self._segment_evaluated_for_event: bool = False

        # Fixed anchor point & persistent cumulative delay (v7.3)
        self.scheduled_departure = self.route[0].get("scheduled_departure") if self.route else "08:05:00"
        self.actual_departure_tick = 0
        self.cumulative_delay_min: float = 0.0
        self.force_suppress_events: bool = False

        # Journey start time parsed for timestamp generation
        self._base_datetime = self._init_base_datetime()

    def _realistic_recovery_rate(self) -> float:
        """
        Trains claw back some schedule slack in transit, but not instantly.
        Capped at 0.1 min recovered per 1 simulated minute of running
        (~6 min per hour), which is realistic for Indian Railways.

        0.3 was too aggressive: a 15-min hold fully evaporated in 50 moving
        ticks — precisely the travel time to the next major station — leaving
        zero delay to propagate downstream (Scenario 14 failure).
        """
        MAX_RECOVERY_PER_TICK = 0.1
        return MAX_RECOVERY_PER_TICK * self.tick_minutes

    def _normalize_route(self, route_input: Any) -> List[StationStop]:
        stops: List[StationStop] = []
        if isinstance(route_input, pd.DataFrame):
            df_sorted = route_input.sort_values(by="seq") if "seq" in route_input.columns else route_input
            for _, r in df_sorted.iterrows():
                stops.append(StationStop(r.to_dict()))
        elif isinstance(route_input, list):
            for i, item in enumerate(route_input):
                if isinstance(item, dict):
                    stops.append(StationStop(item))
                elif isinstance(item, StationStop):
                    stops.append(item)
                elif isinstance(item, str):
                    stops.append(
                        StationStop({
                            "station_code": item,
                            "station_name": item,
                            "seq": i + 1,
                            "distance_km": float(i * 30),
                            "scheduled_arrival": f"{8 + (i * 30) // 60:02d}:{(i * 30) % 60:02d}:00",
                            "scheduled_departure": f"{8 + (i * 30) // 60:02d}:{(i * 30) % 60 + 5:02d}:00",
                        })
                    )
                elif hasattr(item, "to_dict"):
                    stops.append(StationStop(item.to_dict()))
        
        if not stops:
            # Fallback minimum route
            stops = [
                StationStop({"station_code": "ORIGIN", "station_name": "Origin", "seq": 1}),
                StationStop({"station_code": "DEST", "station_name": "Destination", "seq": 2}),
            ]
        return stops

    def _init_base_datetime(self) -> datetime:
        first_stop = self.route[0] if self.route else {}
        arr_str = str(first_stop.get("scheduled_arrival") or first_stop.get("scheduled_departure") or "08:00:00")
        try:
            return datetime.strptime(f"{self.journey_date} {arr_str.strip()}", "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.strptime(f"{self.journey_date} 08:00:00", "%Y-%m-%d %H:%M:%S")

    def _compute_segment_duration(self, station_idx: int) -> float:
        if station_idx >= len(self.route) - 1:
            return 0.0
        curr = self.route[station_idx]
        nxt = self.route[station_idx + 1]

        t_curr_str = curr.get("scheduled_departure") or curr.get("scheduled_arrival")
        t_nxt_str = nxt.get("scheduled_arrival") or nxt.get("scheduled_departure")
        if t_curr_str and t_nxt_str:
            try:
                t1 = datetime.strptime(str(t_curr_str).strip(), "%H:%M:%S")
                t2 = datetime.strptime(str(t_nxt_str).strip(), "%H:%M:%S")
                diff = (t2 - t1).total_seconds() / 60.0
                if diff > 0:
                    return diff
            except Exception:
                pass

        # Estimate from distance difference if available
        d1 = curr.get("distance_km")
        d2 = nxt.get("distance_km")
        if d1 is not None and d2 is not None:
            km = abs(float(d2) - float(d1))
            if km > 0:
                return max(5.0, km * 1.0)  # assume 60 km/h = 1 min/km

        return 20.0  # default 20 minutes between stations

    def get_current_station_code(self) -> str:
        if not self.route:
            return "STATION_UNKNOWN"
        idx = min(self.current_station_idx, len(self.route) - 1)
        return str(self.route[idx].get("station_code") or f"STATION_{idx}")

    def get_current_station_name(self) -> str:
        if not self.route:
            return "Station"
        idx = min(self.current_station_idx, len(self.route) - 1)
        return str(self.route[idx])

    def _get_sim_timestamp(self) -> str:
        dt = self._base_datetime + timedelta(minutes=self.elapsed_ticks * self.tick_minutes)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def schedule_event(
        self,
        event_type: str,
        reason_category: Optional[str] = None,
        duration_minutes: float = 15.0,
        trigger_tick: Optional[int] = None,
        trigger_station_idx: Optional[int] = None,
        status: str = "STOPPED",
    ) -> None:
        """
        Inject a planned operational event for deterministic testing.
        """
        self._injected_events.append({
            "event_type": event_type,
            "reason_category": reason_category or EVENT_REASON_MAP.get(event_type, "OPERATIONAL_REGULATION"),
            "duration_minutes": float(duration_minutes),
            "trigger_tick": trigger_tick,
            "trigger_station_idx": trigger_station_idx,
            "status": status,
            "triggered": False,
        })

    def inject_event(
        self,
        event_type: str,
        reason_category: Optional[str] = None,
        duration_minutes: float = 15.0,
        status: str = "STOPPED",
    ) -> Dict[str, Any]:
        """
        Immediately fire an active event on the current tick.
        Returns the raw Phase 2 event dict.
        """
        duration_ticks = max(1, int(np.ceil(duration_minutes / self.tick_minutes)))
        rcat = reason_category if reason_category is not None else EVENT_REASON_MAP.get(event_type, "OPERATIONAL_REGULATION")
        prior = EXPECTED_DURATION_PRIOR.get(event_type, 15.0)

        self.active_event = {
            "type": event_type,
            "reason_category": rcat,
            "status": status,
            "start_tick": self.elapsed_ticks,
            "duration_ticks": duration_ticks,
            "expected_duration": prior,
            "elapsed_ticks": 0,
        }

        return {
            "train_id": str(self.train_no),
            "timestamp": self._get_sim_timestamp(),
            "event_type": event_type,
            "location": self.get_current_station_code(),
            "reason_category": rcat,
            "status": status,
        }

    def tick(self) -> Optional[Dict[str, Any]]:
        """
        Advance one tick. May:
          - start a new event (injected or probabilistic pre-arrival trigger)
          - continue an existing event (increment elapsed time)
          - resolve an existing event (emit TRAIN_RESUMED)
          - advance position toward the next station if no event is blocking
        Returns a raw event dict matching the Phase 2 schema, or None if
        nothing changed this tick.
        """
        if self.is_journey_complete():
            return None

        self.elapsed_ticks += 1
        raw_event: Optional[Dict[str, Any]] = None

        # 1. Check for pre-scheduled injected events (unless force_suppress_events is True)
        if not self.force_suppress_events:
            for inj in self._injected_events:
                if not inj["triggered"]:
                    tick_match = (inj["trigger_tick"] is not None and self.elapsed_ticks >= inj["trigger_tick"])
                    stn_match = (inj["trigger_station_idx"] is not None and self.current_station_idx == inj["trigger_station_idx"])
                    if tick_match or stn_match:
                        inj["triggered"] = True
                        raw_event = self.inject_event(
                            event_type=inj["event_type"],
                            reason_category=inj["reason_category"],
                            duration_minutes=inj["duration_minutes"],
                            status=inj.get("status", "STOPPED"),
                        )
                        break

        # 2. Check active event resolution or continuation
        if raw_event is None and self.active_event is not None:
            self.active_event["elapsed_ticks"] += 1
            if self.active_event["elapsed_ticks"] >= self.active_event["duration_ticks"]:
                # Event resolves
                loc = self.get_current_station_code()
                self.active_event = None
                raw_event = {
                    "train_id": str(self.train_no),
                    "timestamp": self._get_sim_timestamp(),
                    "event_type": "TRAIN_RESUMED",
                    "location": loc,
                    "reason_category": "RESUMED",
                    "status": "MOVING",
                }

        # 3. Probabilistic trigger check (only if enabled, not currently in event, and not suppressed)
        if raw_event is None and self.active_event is None and self.probabilistic and not self.force_suppress_events and not self._segment_evaluated_for_event:
            self._segment_evaluated_for_event = True
            st_data = self.route[self.current_station_idx]
            hist_avg = float(st_data.get("historical_avg_delay_min", 0.0))
            weather_risk = float(st_data.get("weather_risk_score", 0.0))
            season = str(st_data.get("season", "Summer"))
            is_holiday = int(st_data.get("is_holiday", 0))

            trigger_prob = _compute_trigger_probability(
                historical_avg_delay_min=hist_avg,
                weather_risk_score=weather_risk,
                season=season,
                is_holiday=is_holiday,
            )

            if self.rng.random() < trigger_prob:
                etype = _sample_event_type(weather_risk, season, self.rng)
                dur = _sample_duration_by_type(etype, self.rng)
                rcat = EVENT_REASON_MAP.get(etype, "OPERATIONAL_REGULATION")
                status = "STOPPED" if etype in ["TRAIN_HELD", "SIGNAL_FAILURE"] else "DELAYED"
                raw_event = self.inject_event(
                    event_type=etype,
                    reason_category=rcat,
                    duration_minutes=dur,
                    status=status,
                )

        # 4. Update persistent cumulative delay (v7.3 Anchored Delay)
        if self.active_event is not None and self.active_event.get("status") == "STOPPED":
            self.cumulative_delay_min += self.tick_minutes
        else:
            recovery = self._realistic_recovery_rate()
            self.cumulative_delay_min = max(0.0, self.cumulative_delay_min - recovery)

        # 5. Advance position if not blocked
        if self.active_event is None or self.active_event.get("status") != "STOPPED":
            self._segment_progress_min += self.tick_minutes
            if self._segment_progress_min >= self._segment_duration_min:
                if self.current_station_idx < len(self.route) - 1:
                    self.current_station_idx += 1
                    self._segment_progress_min = 0.0
                    self._segment_duration_min = self._compute_segment_duration(self.current_station_idx)
                    self._segment_evaluated_for_event = False

        return raw_event

    def is_journey_complete(self) -> bool:
        return self.current_station_idx >= len(self.route) - 1
