"""
services/translation_layer.py
==============================
Translation Layer (Phase 9 of ETA Architecture)
Translates internal raw event and reason codes into clear, passenger-friendly messages.

Non-negotiable rule: Fails safe. Unmapped combinations return a generic safe message;
raw event and reason codes are NEVER exposed directly to passengers.
"""

MAP_VERSION = "1.1.0"  # v8: added inferred-event entries for RailRadar adapter

# Versioned, auditable mapping table -- NOT inline code, so it can be
# reviewed and extended without touching logic
EVENT_TRANSLATION_MAP = {
    # ----------------------------------------------------------------
    # Authoritative events (sourced from simulator / authorized feed)
    # ----------------------------------------------------------------
    ("TRAIN_HELD", "OPERATIONAL_REGULATION"): "Train temporarily stopped — operational regulation",
    ("TRAIN_HELD", "CROSSING"):                "Waiting due to another train's movement",
    ("CONGESTION", "OPERATIONAL_REGULATION"):  "Delayed due to traffic congestion",
    ("SIGNAL_FAILURE", "OPERATIONAL_REGULATION"): "Operational regulation in effect",
    ("WEATHER_DISRUPTION", "WEATHER"):         "Delayed due to weather conditions",
    ("TRACK_MAINTENANCE", "MAINTENANCE"):      "Delayed due to scheduled maintenance",
    ("TRAIN_RESUMED", "RESUMED"):              "Train has resumed movement",
    ("NONE", None):                             "On schedule",
    ("NONE", "NONE"):                           "On schedule",

    # ----------------------------------------------------------------
    # Inferred events (v8 — derived from RailRadar delay/speed telemetry)
    # Deliberately hedged: RailRadar has no authorized event taxonomy,
    # so these messages never claim more certainty than the data provides.
    # ----------------------------------------------------------------
    ("TRAIN_HELD", "INFERRED_REGULATION"): "Train appears to be stopped \u2014 reason not confirmed",
    ("CONGESTION",  "INFERRED_TRAFFIC"):   "Train appears to be delayed \u2014 reason not confirmed",
}

DEFAULT_SAFE_MESSAGE = "Operational update — details unavailable"


def translate_event(event_type: str | None, reason_category: str | None) -> str:
    """
    Raw internal event -> plain-language passenger message.
    Fails SAFE: any (event_type, reason_category) combination without an
    approved mapping returns a generic message, never the raw code itself.
    """
    if event_type is None and reason_category is None:
        return "On schedule"
    
    key = (event_type, reason_category)
    if key in EVENT_TRANSLATION_MAP:
        return EVENT_TRANSLATION_MAP[key]
    
    # Check if (event_type, None) is mapped
    if (event_type, None) in EVENT_TRANSLATION_MAP:
        return EVENT_TRANSLATION_MAP[(event_type, None)]

    return DEFAULT_SAFE_MESSAGE
