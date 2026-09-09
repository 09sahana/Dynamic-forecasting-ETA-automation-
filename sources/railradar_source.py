"""
sources/railradar_source.py
============================
RailRadar Live API Adapter (Build Brief v8)

Converts RailRadar's live API response into the Common Observation schema
shared by all other input adapters (CSV, manual, GPS).  This is a NEW
fourth adapter — no existing component is modified.

RailRadar API base: https://api.railradar.in/v1
Auth:               Authorization: Bearer <api_key>
Free tier limit:    1,000 requests/month  (see railradar_poller.py)

Key design decisions:
- `resolve_position` uses RailRadar's `segmentProgress` (0–1 between two known
  stations) instead of raw GPS / Haversine — RailRadar has already done
  map-matching.
- Weather / holiday / season are NOT provided by RailRadar; they are filled
  from STATION_DEFAULTS (per-station static fallback values), following the
  same fallback pattern used by the manual-input adapter.
- Any HTTP or API-level failure raises RailRadarError — callers decide
  whether to fall back to the last known observation (see railradar_poller.py).
"""

import requests
from datetime import datetime
from typing import Optional

RAILRADAR_BASE = "https://api.railradar.in/v1"

# ---------------------------------------------------------------------------
# Station-level static defaults for fields RailRadar does not provide.
# Keyed by station_code (upper-case).  A missing station falls back to the
# GLOBAL_DEFAULTS entry.
# ---------------------------------------------------------------------------
GLOBAL_DEFAULTS = {
    "temperature_c":           25.0,
    "precipitation_mm":        0.0,
    "historical_avg_delay_min": 10.0,
    "is_holiday":              0,
    "season":                  "Summer",
    "day_of_week":             None,   # derived at call time from journey_date
}

STATION_DEFAULTS: dict[str, dict] = {
    # Example entries — extend with real per-station values as needed
    "INDB": {"temperature_c": 28.0, "precipitation_mm": 0.0, "historical_avg_delay_min": 8.0, "is_holiday": 0, "season": "Summer"},
    "UJN":  {"temperature_c": 27.0, "precipitation_mm": 0.0, "historical_avg_delay_min": 9.0, "is_holiday": 0, "season": "Summer"},
    "NDLS": {"temperature_c": 22.0, "precipitation_mm": 0.0, "historical_avg_delay_min": 12.0, "is_holiday": 0, "season": "Winter"},
    "CSTM": {"temperature_c": 30.0, "precipitation_mm": 5.0,  "historical_avg_delay_min": 15.0, "is_holiday": 0, "season": "Monsoon"},
}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RailRadarError(Exception):
    """
    Raised on any RailRadar API failure (network, HTTP error, malformed JSON,
    success:false).  Callers must handle this explicitly — never swallow with
    a bare ``except: pass``.
    """
    pass


# ---------------------------------------------------------------------------
# Raw HTTP helpers
# ---------------------------------------------------------------------------

def fetch_live(train_no: str, api_key: str, timeout: int = 5) -> dict:
    """
    Single live-status fetch for one train.

    Returns the ``data`` sub-object from the API response on success.
    Raises ``RailRadarError`` on **any** failure so the caller can decide
    whether to fall back to the last known observation.

    Example response shape::

        {
            "trainNumber": "12952",
            "trainName":   "MUMBAI RAJDHANI",
            "status":      "running",
            "delayMinutes": 12,
            "currentLocation": {
                "segmentProgress": 0.42,
                "speed": 68.5,
                "bearing": 210
            },
            "lastUpdatedAt": "2026-06-22T07:14:00+05:30"
        }
    """
    try:
        resp = requests.get(
            f"{RAILRADAR_BASE}/trains/{train_no}/live",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RailRadarError(f"RailRadar live request failed for train {train_no}: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RailRadarError(
            f"RailRadar returned malformed JSON for train {train_no}: {exc}"
        ) from exc

    if not payload.get("success"):
        raise RailRadarError(
            f"RailRadar returned success=false for train {train_no}: {payload}"
        )

    return payload["data"]


def fetch_route(train_no: str, api_key: str, timeout: int = 5) -> dict:
    """
    Fetch the full schedule / route for a train (halts only).

    Call this rarely — routes change infrequently; cache aggressively
    (see ``RateLimitedPoller.get_route_cached``).

    Returns the ``data`` sub-object from the API response.
    Raises ``RailRadarError`` on any failure.
    """
    try:
        resp = requests.get(
            f"{RAILRADAR_BASE}/trains/{train_no}",
            params={"haltsOnly": "true"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RailRadarError(
            f"RailRadar route request failed for train {train_no}: {exc}"
        ) from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise RailRadarError(
            f"RailRadar returned malformed JSON (route) for train {train_no}: {exc}"
        ) from exc

    return payload["data"]


# ---------------------------------------------------------------------------
# Position resolver
# ---------------------------------------------------------------------------

def resolve_position(
    current_location: dict,
    route: Optional[dict],
    station_master: Optional[dict] = None,
) -> tuple[str, float, int]:
    """
    Derive (nearest_station_code, distance_remaining_km, stops_remaining)
    from RailRadar's ``currentLocation`` block and the cached route.

    RailRadar's ``segmentProgress`` (0.0–1.0) tells us where the train sits
    *between* two consecutive scheduled stations — no Haversine needed because
    RailRadar has already done the map-matching.

    Parameters
    ----------
    current_location : dict
        The ``currentLocation`` sub-object from the live API response.
        Expected keys: ``segmentProgress``, optionally ``speed``.
    route : dict | None
        The cached route from ``fetch_route``.  If None (not yet cached),
        falls back to generic placeholders so the observation is never None.
    station_master : dict | None
        Optional ``{station_code: {distance_from_origin_km: float, ...}}``
        lookup.  Not required — distances are computed from route data if
        available, or estimated from segmentProgress × total_distance.

    Returns
    -------
    nearest_station : str
        Station code of the *next* scheduled halt (ahead of the train).
    distance_remaining : float
        Estimated km to the final destination.
    stops_remaining : int
        Number of scheduled halts ahead of the current position.
    """
    seg_progress = float(current_location.get("segmentProgress", 0.5))

    if route is None:
        # Graceful fallback when route not yet fetched
        return "UNKNOWN", 100.0, 5

    route_stops = route.get("route", [])
    total_halts = len(route_stops)
    train_info = route.get("train", {})
    total_distance_km = float(train_info.get("distance", 0) or 0)

    if total_halts == 0:
        return "UNKNOWN", total_distance_km * (1.0 - seg_progress), 0

    # RailRadar gives segmentProgress across the *entire* route (not per
    # segment).  Treat it as fraction of the journey already completed.
    completed_fraction = seg_progress
    completed_stops = int(completed_fraction * total_halts)
    completed_stops = max(0, min(completed_stops, total_halts - 1))

    # The train is currently between completed_stops and completed_stops+1
    next_stop_idx = completed_stops  # 0-indexed into route_stops
    next_stop = route_stops[next_stop_idx] if next_stop_idx < total_halts else route_stops[-1]

    nearest_station = next_stop.get("station", {}).get("code", "UNKNOWN")
    station_seq_idx = next_stop.get("sequence", next_stop_idx + 1)

    # Distance to destination
    # If station_master provides per-station distances, use them; otherwise
    # interpolate from segmentProgress × total route distance.
    if (
        station_master
        and nearest_station in station_master
        and "distance_from_origin_km" in station_master[nearest_station]
    ):
        dist_at_next = float(station_master[nearest_station]["distance_from_origin_km"])
        distance_remaining = max(0.0, total_distance_km - dist_at_next)
    elif "distance" in next_stop and total_distance_km > 0:
        dist_at_next = float(next_stop["distance"])
        distance_remaining = max(0.0, total_distance_km - dist_at_next)
    else:
        distance_remaining = total_distance_km * (1.0 - completed_fraction)

    stops_remaining = total_halts - completed_stops - 1
    distance_remaining = round(distance_remaining, 2)
    stops_remaining = max(0, stops_remaining)

    print(
        f"[DEBUG] station={nearest_station} seq_idx={station_seq_idx} "
        f"distance_remaining={distance_remaining} stops_remaining={stops_remaining}"
    )

    return nearest_station, distance_remaining, stops_remaining


# ---------------------------------------------------------------------------
# Default-value helper
# ---------------------------------------------------------------------------

def _get_defaults(station_code: str, journey_date: str) -> dict:
    """
    Return static fallback values for fields RailRadar does not supply
    (weather, holidays, season, day_of_week).  Merges GLOBAL_DEFAULTS with
    any per-station overrides in STATION_DEFAULTS.
    """
    defaults = dict(GLOBAL_DEFAULTS)
    station_overrides = STATION_DEFAULTS.get(station_code.upper(), {})
    defaults.update(station_overrides)

    # Derive day_of_week from journey_date if not overridden
    if defaults.get("day_of_week") is None:
        try:
            defaults["day_of_week"] = datetime.strptime(journey_date, "%Y-%m-%d").strftime("%A")
        except ValueError:
            defaults["day_of_week"] = "Monday"

    return defaults


# ---------------------------------------------------------------------------
# Adapter entry point
# ---------------------------------------------------------------------------

def from_railradar(
    train_no: str,
    api_key: str,
    station_master: Optional[dict],
    route_cache: dict,
    journey_date: Optional[str] = None,
) -> dict:
    """
    **Adapter entry point.**

    Calls ``fetch_live``, combines with the cached route, and returns a
    Common Observation dict with the **exact same schema** used by the
    CSV, manual, and GPS adapters — ready to pass directly to ``predict()``.

    Parameters
    ----------
    train_no : str
        Train number as a string (e.g. ``"12952"``).
    api_key : str
        RailRadar API key (Bearer token value).
    station_master : dict | None
        Optional ``{station_code: {...}}`` lookup for distance cross-reference.
    route_cache : dict
        ``{train_no: route_data}`` dict populated by the poller's
        ``get_route_cached``.  Pass ``{}`` if route not yet fetched;
        ``resolve_position`` will fall back gracefully.
    journey_date : str | None
        Optional anchored journey date (``"YYYY-MM-DD"``). If None, defaults
        to today's date at poll time.

    Returns
    -------
    dict
        Common Observation dict.  Fields:
        - train_no, journey_date, station_code
        - previous_station_delay (== delayMinutes from live API)
        - distance_remaining, number_of_stops_remaining
        - raw_speed_kmph, raw_status, source, fetched_at
        - temperature_c, precipitation_mm, historical_avg_delay_min
        - is_holiday, season, day_of_week  (from DEFAULTS)

    Raises
    ------
    RailRadarError
        Propagated from ``fetch_live`` — caller must handle.
    """
    live = fetch_live(train_no, api_key)
    route = route_cache.get(train_no)

    if journey_date is None:
        journey_date = datetime.now().strftime("%Y-%m-%d")

    nearest_station, distance_remaining, stops_remaining = resolve_position(
        live.get("currentLocation", {}),
        route,
        station_master,
    )

    defaults = _get_defaults(nearest_station, journey_date)

    return {
        # ---- Core identifiers ------------------------------------------------
        "train_no":                   int(train_no) if str(train_no).isdigit() else train_no,
        "journey_date":               journey_date,
        "station_code":               nearest_station,

        # ---- Live delay signal -----------------------------------------------
        "previous_station_delay":     float(live.get("delayMinutes", 0)),

        # ---- Position --------------------------------------------------------
        "distance_remaining":         distance_remaining,
        "number_of_stops_remaining":  stops_remaining,

        # ---- Speed / status (raw, for event_inference) ----------------------
        "raw_speed_kmph":             live.get("currentLocation", {}).get("speed"),
        "raw_status":                 live.get("status"),

        # ---- Provenance ------------------------------------------------------
        "source":                     "railradar",
        "fetched_at":                 live.get("lastUpdatedAt"),

        # ---- Filled from DEFAULTS (RailRadar does not provide these) --------
        "temperature_c":              defaults["temperature_c"],
        "precipitation_mm":           defaults["precipitation_mm"],
        "historical_avg_delay_min":   defaults["historical_avg_delay_min"],
        "is_holiday":                 defaults["is_holiday"],
        "season":                     defaults["season"],
        "day_of_week":                defaults["day_of_week"],
    }
