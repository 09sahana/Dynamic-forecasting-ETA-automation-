"""
services/passenger_filter.py
============================
Passenger Filter (Phase 10 of ETA Architecture)
The gate that runs *before* translation, deciding whether an internal event type
is approved to reach passenger-facing channels.

Uses strict whitelisting so unvetted internal/diagnostic event types default to hidden.
"""

from services.translation_layer import translate_event

# Whitelist of approved event types that passengers are permitted to see.
# v8: CONGESTION added to support INFERRED_TRAFFIC entries from RailRadar adapter.
APPROVED_EVENT_TYPES = {
    # -- Authoritative (simulator / authorized feed) --
    "TRAIN_HELD",
    "REGULATION",
    "CROSSING",
    "CONGESTION",
    "DIVERSION",
    "MAINTENANCE",
    "WEATHER",
    "RESUMED",
    "NONE",
    "SIGNAL_FAILURE",
    "WEATHER_DISRUPTION",
    "TRACK_MAINTENANCE",
    "TRAIN_RESUMED",
}



def passenger_can_see(event_type: str | None) -> bool:
    """
    Whitelist check. Anything not explicitly approved is hidden by default
    -- whitelisting, not blacklisting, so a new internal event type added
    later is hidden until explicitly approved, not exposed by accident.
    """
    if not event_type:
        return False
    return event_type in APPROVED_EVENT_TYPES


def filter_and_prepare(raw_event: dict | None) -> dict | None:
    """
    Returns a passenger-safe event dict, or None if this event should not
    be shown at all (e.g. an internal-only diagnostic event type).
    """
    if not raw_event:
        return None

    event_type = raw_event.get("event_type")
    if not passenger_can_see(event_type):
        return None

    return {
        "status": raw_event.get("status", "UNKNOWN"),
        "reason": translate_event(event_type, raw_event.get("reason_category")),
        "location": raw_event.get("location"),
    }


def should_notify_passenger(
    last_update: dict | None,
    curr_update: dict,
    min_delay_delta_min: float = 3.0,
) -> bool:
    """
    Component 7.5: Confidence-Aware Passenger Notification Threshold.
    Decides whether a live journey update warrants notifying the passenger.
    
    Filters out minor noise/fluctuations (< 3 minutes) unless:
    - Status transitions (e.g. MOVING <-> STOPPED)
    - Confidence level changes tier (e.g. High -> Medium or Medium -> Low)
    - A visible disruption or resolution event has fired
    - Delay change exceeds min_delay_delta_min (default 3.0 minutes)
    """
    if last_update is None:
        return True

    # 1. New passenger-facing event / disruption
    if curr_update.get("passenger_view") is not None:
        return True

    last_res = last_update.get("result", {})
    curr_res = curr_update.get("result", {})

    # 2. Confidence tier boundary crossed
    last_conf = last_res.get("confidence")
    curr_conf = curr_res.get("confidence")
    if last_conf != curr_conf:
        return True

    # 3. Station stop transition
    if last_update.get("station_code") != curr_update.get("station_code"):
        return True

    # 4. Significant delay delta (> min_delay_delta_min)
    last_delay = last_res.get("predicted_delay_min")
    curr_delay = curr_res.get("predicted_delay_min")
    if last_delay is not None and curr_delay is not None:
        if abs(curr_delay - last_delay) >= min_delay_delta_min:
            return True

    # 5. Cancellation or severe risk alert
    if curr_res.get("is_cancelled") != last_res.get("is_cancelled"):
        return True

    return False

