"""
src/explainability.py
======================
Translates model inputs and prediction state into plain-language
contributing factors for the "Why this prediction?" dashboard panel.
"""

from typing import Any, Dict, List


def explain_prediction(obs: Dict[str, Any], pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate the top 3 human-interpretable contributing factors for this prediction.

    Returns a list of dicts:
    [
        {"factor": str, "impact": str, "description": str, "direction": "increase"|"decrease"|"neutral"},
        ...
    ]
    """
    explanations = []

    # 1. Active incident / operational event impact
    event_type = obs.get("event_type", "NONE")
    event_elapsed = float(obs.get("event_elapsed_min", 0.0) or 0.0)
    event_expected = float(obs.get("event_expected_min", 0.0) or 0.0)
    if event_type not in ("NONE", None):
        impact_min = max(event_elapsed, event_expected)
        explanations.append({
            "factor": f"Active Event: {event_type}",
            "impact": f"+{impact_min:.0f} min",
            "description": f"Incident underway for {event_elapsed:.0f}m (dispatcher duration prior ~{event_expected:.0f}m)",
            "direction": "increase",
        })

    # 2. Accumulated upstream delay
    prev_delay = float(obs.get("previous_station_delay", 0.0) or 0.0)
    if prev_delay > 10.0:
        explanations.append({
            "factor": "Upstream Departure Lag",
            "impact": f"+{prev_delay:.0f} min",
            "description": f"Carried forward running delay accumulated since journey origin",
            "direction": "increase",
        })
    elif prev_delay <= 3.0:
        explanations.append({
            "factor": "On-Time Section Running",
            "impact": "Minimal",
            "description": "Train departed previous halt on or near schedule",
            "direction": "neutral",
        })

    # 3. Chronic train tier
    tier = pred.get("chronic_delay_tier", "T3")
    is_fallback = pred.get("chronic_tier_is_fallback", 0)
    if tier in ("T4", "T5_worst", "T5"):
        explanations.append({
            "factor": f"Chronic Delay History ({tier})",
            "impact": "+15–30 min",
            "description": "Historical punctuality data shows recurring corridor congestion for this train",
            "direction": "increase",
        })
    elif is_fallback:
        explanations.append({
            "factor": "Cold-Start Service (T3)",
            "impact": "Baseline",
            "description": "Unseen train service in production — applying neutral T3 corridor prior",
            "direction": "neutral",
        })

    # 4. Weather / visibility impact
    precip = float(obs.get("precipitation_mm", 0.0) or 0.0)
    temp = float(obs.get("temperature_c", 25.0) or 25.0)
    if precip > 25.0:
        explanations.append({
            "factor": "Heavy Precipitation",
            "impact": "+10–20 min",
            "description": f"Monsoon rainfall ({precip:.1f} mm) enforcing precautionary speed restrictions",
            "direction": "increase",
        })
    elif temp > 42.0:
        explanations.append({
            "factor": "High Ambient Heat",
            "impact": "+5–10 min",
            "description": f"Track heat warnings ({temp:.1f}°C) requiring cautionary headway",
            "direction": "increase",
        })

    # 5. Route section historical density
    hist_delay = float(obs.get("historical_avg_delay_min", 0.0) or 0.0)
    if len(explanations) < 3 and hist_delay > 15.0:
        explanations.append({
            "factor": "Corridor Section Density",
            "impact": f"+{hist_delay:.0f} min",
            "description": f"Approaching high-traffic junction with elevated historical section delay",
            "direction": "increase",
        })

    # Fallback to keep at least 3 factors
    if len(explanations) < 3:
        dist_rem = float(obs.get("distance_remaining", 0.0) or 0.0)
        stops_rem = int(obs.get("number_of_stops_remaining", 0) or 0)
        explanations.append({
            "factor": "Journey Progression",
            "impact": "Nominal",
            "description": f"{dist_rem:.0f} km and {stops_rem} scheduled halts remaining to destination",
            "direction": "neutral",
        })

    return explanations[:3]
