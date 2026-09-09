"""
services/event_inference.py
============================
Event Inference Engine (Build Brief v8)

RailRadar provides ``delayMinutes`` and ``status`` but has **no internal
event taxonomy** — no TRAIN_HELD, no reason_category.  This module derives
a plausible ``event_type`` from the *shape* of the delay / speed signal
over consecutive polls.

**Important:** every output from this module carries a ``confidence_note``
field that explicitly labels it as *inferred*, not authoritative.  Downstream
code (translation_layer.py) uses hedged phrasing for inferred events so that
uncertainty is honestly communicated to passengers — following the same
passenger-trust principle as Phase 10.

One ``DelayTrendTracker`` instance per tracked train; do not share instances
across trains.
"""

from __future__ import annotations

from typing import NamedTuple


# ---------------------------------------------------------------------------
# Inference thresholds (module-level constants so they are easy to tune)
# ---------------------------------------------------------------------------

_DELAY_RISE_THRESHOLD_MIN: float = 3.0   # minutes of increase to count as "rising"
_DELAY_DROP_THRESHOLD_MIN: float = 3.0   # minutes of decrease to count as "recovering"
_STATIONARY_SPEED_KMH: float     = 5.0   # below this → train considered near-stationary


class _HistoryPoint(NamedTuple):
    timestamp: str | None
    delay_min: float
    speed_kmph: float | None


class DelayTrendTracker:
    """
    Keeps a short rolling history of ``(timestamp, delayMinutes, speed_kmph)``
    per train and infers a plausible ``event_type`` from the trend shape.

    Usage::

        tracker = DelayTrendTracker(window=5)
        tracker.update(ts, delay_min, speed_kmph)
        inferred = tracker.infer_event()
        # inferred is a dict with keys:
        #   event_type, reason_category (optional), confidence_note

    Parameters
    ----------
    window : int
        Number of consecutive polls to keep.  Default 5 — enough to see a
        clear trend while staying memory-light.
    """

    def __init__(self, window: int = 5) -> None:
        if window < 2:
            raise ValueError("window must be >= 2 to compare first and last readings")
        self.window = window
        self.history: list[_HistoryPoint] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, timestamp: str | None, delay_min: float, speed_kmph: float | None) -> None:
        """
        Append a new observation and trim to the configured window.

        Parameters
        ----------
        timestamp : str | None
            ISO-8601 string from ``lastUpdatedAt`` (used for diagnostics only).
        delay_min : float
            Current ``delayMinutes`` from the live API.
        speed_kmph : float | None
            Current speed in km/h.  Pass ``None`` when not available.
        """
        self.history.append(_HistoryPoint(timestamp, float(delay_min), speed_kmph))
        self.history = self.history[-self.window:]

    def infer_event(self) -> dict:
        """
        Derive a plausible event classification from the rolling history.

        Returns
        -------
        dict with keys:
            - ``event_type``     : str  — one of TRAIN_HELD, CONGESTION,
                                          TRAIN_RESUMED, NONE
            - ``reason_category``: str  — present when event_type != NONE
            - ``confidence_note``: str  — always present; explicitly marks
                                          output as *inferred*, never authoritative
        """
        if len(self.history) < 2:
            return {
                "event_type":      "NONE",
                "confidence_note": "insufficient_history",
            }

        oldest = self.history[0]
        latest = self.history[-1]
        prev   = self.history[-2]

        delay_rising = (latest.delay_min - oldest.delay_min) > _DELAY_RISE_THRESHOLD_MIN
        delay_recovering = (prev.delay_min - latest.delay_min) > _DELAY_DROP_THRESHOLD_MIN

        # Speed: treat None as "unknown" — only assert stationary if we have a value
        near_stationary = (
            latest.speed_kmph is not None
            and latest.speed_kmph < _STATIONARY_SPEED_KMH
        )

        # Priority order: stationary + rising > moving + rising > recovering > no change
        if near_stationary and delay_rising:
            return {
                "event_type":      "TRAIN_HELD",
                "reason_category": "INFERRED_REGULATION",
                "confidence_note": "inferred_from_speed_and_delay_trend",
            }

        if delay_rising and not near_stationary:
            return {
                "event_type":      "CONGESTION",
                "reason_category": "INFERRED_TRAFFIC",
                "confidence_note": "inferred_from_delay_trend_only",
            }

        if delay_recovering:
            return {
                "event_type":      "TRAIN_RESUMED",
                "reason_category": "RESUMED",
                "confidence_note": "inferred_from_delay_recovering",
            }

        return {
            "event_type":      "NONE",
            "confidence_note": "no_significant_change",
        }

    # ------------------------------------------------------------------
    # Introspection helpers (useful in tests and logging)
    # ------------------------------------------------------------------

    def latest_delay(self) -> float | None:
        """Return the most recent delay reading, or None if history is empty."""
        return self.history[-1].delay_min if self.history else None

    def latest_speed(self) -> float | None:
        """Return the most recent speed reading, or None if unavailable."""
        return self.history[-1].speed_kmph if self.history else None

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return (
            f"DelayTrendTracker(window={self.window}, "
            f"readings={len(self.history)}, "
            f"latest_delay={self.latest_delay()}, "
            f"latest_speed={self.latest_speed()})"
        )
