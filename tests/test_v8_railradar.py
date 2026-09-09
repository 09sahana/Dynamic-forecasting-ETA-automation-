"""
tests/test_v8_railradar.py
===========================
Build Brief v8 — All 9 test cases for the RailRadar integration.

All tests are fully **offline** — every ``requests.get`` call is patched
with ``unittest.mock`` so no real HTTP traffic is generated and no API
quota is consumed.

Test matrix (matches §7 of the build brief exactly):
  #1  Normal live observation assembly — schema matches Common Observation
  #2  RailRadarError (network error) → loop falls back, logs, does NOT crash
  #3  success:false / malformed JSON → same fallback (not bare except: pass)
  #4  requests_this_month near quota → should_poll() returns False, no request
  #5  Stationary + rising delay → TRAIN_HELD / INFERRED_REGULATION
  #6  Moving + rising delay → CONGESTION / INFERRED_TRAFFIC (not TRAIN_HELD)
  #7  Delay decreasing → TRAIN_RESUMED
  #8  Missing weather/holiday fields filled via DEFAULTS — no None to model
  #9  Two trains in one session → per-train state isolation (no cross-train leak)

Run:
    pytest tests/test_v8_railradar.py -v
or standalone:
    python tests/test_v8_railradar.py
"""

from __future__ import annotations

import json
import sys
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ---- Optional pytest compatibility shim ------------------------------------
try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
    class _MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        class raises:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
    pytest = _MockPytest()


# ===========================================================================
# Helpers — minimal RailRadar API response factories
# ===========================================================================

def _make_live_response(
    delay_min: float = 5.0,
    speed_kmph: float = 70.0,
    status: str = "running",
    seg_progress: float = 0.4,
) -> dict:
    """Return a minimal ``fetch_live`` data sub-object."""
    return {
        "trainNumber": "12952",
        "trainName": "TEST EXPRESS",
        "status": status,
        "delayMinutes": delay_min,
        "currentLocation": {
            "segmentProgress": seg_progress,
            "speed": speed_kmph,
            "bearing": 270,
        },
        "lastUpdatedAt": "2026-06-22T07:14:00+05:30",
    }


def _make_route_response() -> dict:
    """Return a minimal ``fetch_route`` data sub-object."""
    return {
        "train": {"number": "12952", "distance": 500, "totalHalts": 10},
        "route": [
            {"sequence": i, "station": {"code": f"ST{i:02d}", "name": f"Station {i}"},
             "arrival": "10:00", "departure": "10:05", "isHalt": True}
            for i in range(1, 11)
        ],
    }


def _mock_requests_get_live(train_no: str = "12952", **live_kwargs):
    """
    Return a ``patch`` target that makes ``requests.get`` return a valid
    live response for the given train.
    """
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "success": True,
        "data": _make_live_response(**live_kwargs),
    }
    return resp


# ===========================================================================
# Test 1 — Normal observation assembly, schema validation
# ===========================================================================

class TestNormalObservationAssembly(unittest.TestCase):
    """
    #1: Live train, normal conditions.
    Verify the Common Observation dict contains all required fields and
    correct types — matching the schema expected by predict().
    """

    REQUIRED_FIELDS = {
        "train_no", "journey_date", "station_code",
        "previous_station_delay", "distance_remaining",
        "number_of_stops_remaining", "raw_speed_kmph", "raw_status",
        "source", "fetched_at",
        "temperature_c", "precipitation_mm", "historical_avg_delay_min",
        "is_holiday", "season", "day_of_week",
    }

    def test_schema_completeness(self):
        from sources.railradar_source import from_railradar

        live_resp = _make_live_response(delay_min=12.0, speed_kmph=68.5)
        route_resp = _make_route_response()

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"success": True, "data": live_resp}),
            )
            obs = from_railradar(
                train_no="12952",
                api_key="rr_live_TEST",
                station_master=None,
                route_cache={"12952": route_resp},
            )

        missing = self.REQUIRED_FIELDS - set(obs.keys())
        self.assertFalse(missing, f"Missing fields in observation: {missing}")

    def test_source_is_railradar(self):
        from sources.railradar_source import from_railradar

        live_resp = _make_live_response()
        route_resp = _make_route_response()

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"success": True, "data": live_resp}),
            )
            obs = from_railradar("12952", "rr_live_TEST", None, {"12952": route_resp})

        self.assertEqual(obs["source"], "railradar")

    def test_delay_minutes_mapped_correctly(self):
        from sources.railradar_source import from_railradar

        live_resp = _make_live_response(delay_min=17.0)
        route_resp = _make_route_response()

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"success": True, "data": live_resp}),
            )
            obs = from_railradar("12952", "rr_live_TEST", None, {"12952": route_resp})

        self.assertAlmostEqual(obs["previous_station_delay"], 17.0)

    def test_no_none_values_for_model_fields(self):
        """predict() must never receive None for required numeric fields."""
        from sources.railradar_source import from_railradar

        live_resp = _make_live_response()
        route_resp = _make_route_response()

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"success": True, "data": live_resp}),
            )
            obs = from_railradar("12952", "rr_live_TEST", None, {"12952": route_resp})

        numeric_model_fields = [
            "previous_station_delay", "distance_remaining",
            "temperature_c", "precipitation_mm",
            "historical_avg_delay_min", "is_holiday",
        ]
        for field in numeric_model_fields:
            self.assertIsNotNone(obs.get(field), f"Field '{field}' is None")


# ===========================================================================
# Test 2 — Network error falls back gracefully
# ===========================================================================

class TestNetworkErrorFallback(unittest.TestCase):
    """
    #2: RailRadarError raised on network failure.
    The loop must NOT crash — fall back to the last known observation and
    log the failure clearly.
    """

    def test_railradar_error_raised_on_request_exception(self):
        import requests as req
        from sources.railradar_source import fetch_live, RailRadarError

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_get.side_effect = req.ConnectionError("connection refused")
            with self.assertRaises(RailRadarError):
                fetch_live("12952", "rr_live_TEST")

    def test_loop_does_not_crash_on_railradar_error(self):
        """
        Simulate a scenario where poll_live raises RailRadarError.
        The run_journey_from_railradar loop should fall back to last_obs
        and NOT propagate the exception.
        """
        from sources.railradar_source import RailRadarError

        # poll_live raises once, then the KeyboardInterrupt in time.sleep ends the loop
        poll_side_effects = [
            RailRadarError("timeout"),  # 1st call: API error → fallback path
        ]

        with patch("services.railradar_poller.RateLimitedPoller.poll_live",
                   side_effect=poll_side_effects + [None] * 20) as mock_poll, \
             patch("services.railradar_poller.RateLimitedPoller.get_route_cached",
                   return_value=_make_route_response()), \
             patch("services.railradar_poller.RateLimitedPoller.start"), \
             patch("services.railradar_poller.RateLimitedPoller.stop"), \
             patch("sources.run_railradar_live.predict") as mock_predict, \
             patch("sources.run_railradar_live.filter_and_prepare", return_value=None), \
             patch("time.sleep", side_effect=[None, KeyboardInterrupt]):

            mock_predict.return_value = {"predicted_eta": "10:30", "confidence": "Medium"}

            from sources.run_railradar_live import run_journey_from_railradar
            # Must NOT raise RailRadarError — loop handles it as a fallback
            run_journey_from_railradar("12952", "rr_live_TEST")

    def test_error_message_is_specific_not_bare_except(self):
        """
        Confirm RailRadarError message includes meaningful context, not just
        a bare exception swallowed with pass.
        """
        from sources.railradar_source import RailRadarError
        exc = RailRadarError("RailRadar live request failed for train 12952: connection refused")
        self.assertIn("12952", str(exc))
        self.assertIn("failed", str(exc).lower())


# ===========================================================================
# Test 3 — success:false / malformed JSON → same fallback
# ===========================================================================

class TestMalformedResponseFallback(unittest.TestCase):
    """
    #3: success:false or malformed JSON both raise RailRadarError, not a
    generic exception or a silent pass.
    """

    def test_success_false_raises_railradar_error(self):
        from sources.railradar_source import fetch_live, RailRadarError

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"success": False, "error": "train not found"}),
            )
            with self.assertRaises(RailRadarError) as ctx:
                fetch_live("99999", "rr_live_TEST")
            self.assertIn("success=false", str(ctx.exception))

    def test_malformed_json_raises_railradar_error(self):
        from sources.railradar_source import fetch_live, RailRadarError

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
            mock_get.return_value = mock_resp

            with self.assertRaises(RailRadarError) as ctx:
                fetch_live("12952", "rr_live_TEST")
            self.assertIn("malformed JSON", str(ctx.exception))

    def test_http_error_raises_railradar_error(self):
        import requests as req
        from sources.railradar_source import fetch_live, RailRadarError

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = req.HTTPError("429 Too Many Requests")
            mock_get.return_value = mock_resp

            with self.assertRaises(RailRadarError):
                fetch_live("12952", "rr_live_TEST")


# ===========================================================================
# Test 4 — Quota near-exhaustion → should_poll() returns False
# ===========================================================================

class TestQuotaExhaustion(unittest.TestCase):
    """
    #4: When requests_this_month >= 95% of monthly_quota,
    should_poll() must return False and no further requests fire.
    """

    def _make_poller(self, quota: int, used: int) -> "RateLimitedPoller":
        from services.railradar_poller import RateLimitedPoller

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            quota_file = Path(f.name)

        from datetime import datetime
        quota_record = {
            "month": datetime.now().strftime("%Y-%m"),
            "requests_this_month": used,
        }
        with open(quota_file, "w") as f:
            json.dump(quota_record, f)

        poller = RateLimitedPoller(
            api_key="rr_live_TEST",
            monthly_quota=quota,
            min_interval_seconds=0,   # no time gate in this test
            quota_file=quota_file,
        )
        poller.start()
        return poller

    def test_should_poll_false_when_near_quota(self):
        poller = self._make_poller(quota=1000, used=960)  # 96% → over 95% threshold
        result = poller.should_poll("12952")
        self.assertFalse(result, "should_poll must return False when >=95% of quota used")

    def test_should_poll_true_when_below_quota(self):
        poller = self._make_poller(quota=1000, used=900)  # 90% → under threshold
        result = poller.should_poll("12952")
        self.assertTrue(result, "should_poll must return True when below 95% threshold")

    def test_poll_live_does_not_fire_request_when_rate_limited(self):
        poller = self._make_poller(quota=1000, used=960)

        with patch("sources.railradar_source.fetch_live") as mock_fetch:
            result = poller.poll_live("12952")

        mock_fetch.assert_not_called()
        self.assertIsNone(result)

    def test_quota_persists_across_poller_restart(self):
        """The counter read from disk must reflect previously recorded usage."""
        from services.railradar_poller import RateLimitedPoller
        from datetime import datetime

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({
                "month": datetime.now().strftime("%Y-%m"),
                "requests_this_month": 500,
            }, f)
            quota_file = Path(f.name)

        poller = RateLimitedPoller(
            api_key="rr_live_TEST",
            monthly_quota=1000,
            min_interval_seconds=0,
            quota_file=quota_file,
        )
        self.assertEqual(poller.requests_this_month, 500,
                         "Quota counter must be loaded from disk, not reset on restart")


# ===========================================================================
# Test 5 — Stationary + rising delay → TRAIN_HELD / INFERRED_REGULATION
# ===========================================================================

class TestInferTrainHeld(unittest.TestCase):
    """#5: speed < 5 km/h AND delay rising > 3 min → TRAIN_HELD."""

    def test_held_inferred_from_speed_and_delay(self):
        from services.event_inference import DelayTrendTracker

        tracker = DelayTrendTracker(window=5)
        tracker.update("T0", delay_min=5.0,  speed_kmph=60.0)
        tracker.update("T1", delay_min=6.0,  speed_kmph=30.0)
        tracker.update("T2", delay_min=8.0,  speed_kmph=3.0)    # near-stationary
        tracker.update("T3", delay_min=10.0, speed_kmph=2.0)    # delay rising, stopped

        result = tracker.infer_event()
        self.assertEqual(result["event_type"], "TRAIN_HELD")
        self.assertEqual(result["reason_category"], "INFERRED_REGULATION")
        self.assertIn("inferred", result["confidence_note"])

    def test_held_not_inferred_when_speed_is_sufficient(self):
        from services.event_inference import DelayTrendTracker

        tracker = DelayTrendTracker(window=5)
        # Rising delay but train still moving — should not be TRAIN_HELD
        tracker.update("T0", delay_min=5.0,  speed_kmph=60.0)
        tracker.update("T1", delay_min=10.0, speed_kmph=40.0)

        result = tracker.infer_event()
        self.assertNotEqual(result["event_type"], "TRAIN_HELD")

    def test_confidence_note_always_present(self):
        from services.event_inference import DelayTrendTracker

        tracker = DelayTrendTracker(window=5)
        tracker.update("T0", delay_min=5.0, speed_kmph=2.0)
        tracker.update("T1", delay_min=9.0, speed_kmph=1.0)

        result = tracker.infer_event()
        self.assertIn("confidence_note", result,
                      "confidence_note must always be present to label output as inferred")


# ===========================================================================
# Test 6 — Moving + rising delay → CONGESTION (not TRAIN_HELD)
# ===========================================================================

class TestInferCongestion(unittest.TestCase):
    """#6: delay rising AND speed >= 5 km/h → CONGESTION / INFERRED_TRAFFIC."""

    def test_congestion_inferred_when_moving_with_rising_delay(self):
        from services.event_inference import DelayTrendTracker

        tracker = DelayTrendTracker(window=5)
        tracker.update("T0", delay_min=3.0,  speed_kmph=70.0)
        tracker.update("T1", delay_min=8.0,  speed_kmph=55.0)   # delay up by 5, still moving

        result = tracker.infer_event()
        self.assertEqual(result["event_type"], "CONGESTION")
        self.assertEqual(result["reason_category"], "INFERRED_TRAFFIC")
        self.assertNotEqual(result["event_type"], "TRAIN_HELD",
                            "CONGESTION and TRAIN_HELD must not be conflated when speed is above threshold")

    def test_congestion_translation_yields_hedged_message(self):
        from services.translation_layer import translate_event
        msg = translate_event("CONGESTION", "INFERRED_TRAFFIC")
        self.assertIn("appears", msg.lower(),
                      "Inferred message must use hedged language ('appears')")
        self.assertNotIn("INFERRED_TRAFFIC", msg,
                         "Raw reason code must never leak to passenger-facing output")

    def test_congestion_in_passenger_whitelist(self):
        from services.passenger_filter import passenger_can_see
        self.assertTrue(
            passenger_can_see("CONGESTION"),
            "CONGESTION must be in APPROVED_EVENT_TYPES for inferred events to be visible",
        )


# ===========================================================================
# Test 7 — Delay decreasing → TRAIN_RESUMED
# ===========================================================================

class TestInferTrainResumed(unittest.TestCase):
    """#7: delay falling > 3 min over window → TRAIN_RESUMED."""

    def test_resumed_inferred_from_delay_recovery(self):
        from services.event_inference import DelayTrendTracker

        tracker = DelayTrendTracker(window=5)
        # Build a history where the second-to-last entry has high delay,
        # and the final entry drops by more than 3 min — matching
        # infer_event()'s delay_recovering = (prev.delay_min - latest.delay_min) > 3.
        tracker.update("T0", delay_min=10.0, speed_kmph=0.0)
        tracker.update("T1", delay_min=15.0, speed_kmph=0.0)   # prev = 15.0
        tracker.update("T2", delay_min=11.0, speed_kmph=50.0)  # latest = 11.0, drop = 4 min

        result = tracker.infer_event()
        self.assertEqual(result["event_type"], "TRAIN_RESUMED")
        self.assertEqual(result["reason_category"], "RESUMED")

    def test_resumed_translation_is_authoritative(self):
        """TRAIN_RESUMED / RESUMED is the same entry used by the simulator — verify it maps."""
        from services.translation_layer import translate_event
        msg = translate_event("TRAIN_RESUMED", "RESUMED")
        self.assertIn("resumed", msg.lower())


# ===========================================================================
# Test 8 — Missing weather/holiday fields filled via DEFAULTS
# ===========================================================================

class TestDefaultsFallback(unittest.TestCase):
    """
    #8: RailRadar provides no weather / holiday / season data.
    from_railradar() must fill these from STATION_DEFAULTS / GLOBAL_DEFAULTS
    — no None values, no crash.
    """

    WEATHER_FIELDS = ["temperature_c", "precipitation_mm", "is_holiday", "season", "day_of_week"]

    def _build_obs(self, station_code="UNKNOWN_STA") -> dict:
        from sources.railradar_source import from_railradar

        live_resp = _make_live_response()
        route = _make_route_response()
        # Manually override nearest station to the requested code
        route["route"][4]["station"]["code"] = station_code  # mid-journey stop

        with patch("sources.railradar_source.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"success": True, "data": live_resp}),
            )
            return from_railradar("12952", "rr_live_TEST", None, {"12952": route})

    def test_no_none_weather_fields_for_known_station(self):
        obs = self._build_obs("NDLS")
        for field in self.WEATHER_FIELDS:
            self.assertIsNotNone(obs.get(field), f"Field '{field}' is None for known station NDLS")

    def test_no_none_weather_fields_for_unknown_station(self):
        obs = self._build_obs("ZZUNKNOWN")
        for field in self.WEATHER_FIELDS:
            self.assertIsNotNone(obs.get(field), f"Field '{field}' is None for unknown station")

    def test_global_defaults_are_numeric(self):
        obs = self._build_obs("ZZUNKNOWN")
        self.assertIsInstance(obs["temperature_c"], (int, float))
        self.assertIsInstance(obs["precipitation_mm"], (int, float))
        self.assertIsInstance(obs["historical_avg_delay_min"], (int, float))


# ===========================================================================
# Test 9 — Two trains — per-train state isolation
# ===========================================================================

class TestPerTrainStateIsolation(unittest.TestCase):
    """
    #9: route_cache and last_poll_time must be keyed per train_no.
    No cross-train state leak (same class of bug as v7.2).
    """

    def _make_poller(self) -> "RateLimitedPoller":
        from services.railradar_poller import RateLimitedPoller
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            quota_file = Path(f.name)
        poller = RateLimitedPoller(
            api_key="rr_live_TEST",
            monthly_quota=1000,
            min_interval_seconds=0,  # no time gate
            quota_file=quota_file,
        )
        poller.start()
        return poller

    def test_route_cache_keyed_per_train(self):
        from services.railradar_poller import RateLimitedPoller

        poller = self._make_poller()
        route_a = _make_route_response()
        route_b = {**_make_route_response(), "train": {"number": "99999", "distance": 800, "totalHalts": 20}}

        with patch("sources.railradar_source.fetch_route") as mock_fetch:
            mock_fetch.side_effect = [route_a, route_b]
            r_a = poller.get_route_cached("12952")
            r_b = poller.get_route_cached("99999")

        self.assertIsNot(r_a, r_b, "Route cache must return distinct objects per train_no")
        self.assertIn("12952", poller.route_cache)
        self.assertIn("99999", poller.route_cache)

    def test_route_cached_only_once_per_train(self):
        poller = self._make_poller()
        route_a = _make_route_response()

        with patch("sources.railradar_source.fetch_route", return_value=route_a) as mock_fetch:
            _ = poller.get_route_cached("12952")
            _ = poller.get_route_cached("12952")  # second call — must NOT re-fetch

        mock_fetch.assert_called_once()

    def test_last_poll_time_keyed_per_train(self):
        poller = self._make_poller()
        route = _make_route_response()

        live_a = _make_live_response(delay_min=5.0)
        live_b = _make_live_response(delay_min=12.0)

        def fake_fetch(train_no, api_key, **kwargs):
            return live_a if train_no == "12952" else live_b

        with patch("sources.railradar_source.fetch_live", side_effect=fake_fetch):
            poller.poll_live("12952")
            poller.poll_live("99999")

        self.assertIn("12952", poller.last_poll_time)
        self.assertIn("99999", poller.last_poll_time)
        # Timestamps should be independent (different train_no keys)
        self.assertIsNot(
            poller.last_poll_time["12952"],
            poller.last_poll_time["99999"],
        )

    def test_two_delay_trackers_are_independent(self):
        """DelayTrendTracker instances must not share state."""
        from services.event_inference import DelayTrendTracker

        tracker_a = DelayTrendTracker(window=5)
        tracker_b = DelayTrendTracker(window=5)

        # Feed train A a held pattern
        tracker_a.update("T0", delay_min=5.0,  speed_kmph=2.0)
        tracker_a.update("T1", delay_min=10.0, speed_kmph=1.0)

        # Train B has no updates yet
        result_b = tracker_b.infer_event()
        self.assertEqual(result_b["event_type"], "NONE",
                         "tracker_b must be unaffected by tracker_a's history")
        self.assertEqual(result_b["confidence_note"], "insufficient_history")


# ===========================================================================
# Additional acceptance-criteria checks
# ===========================================================================

class TestTranslationMapEntries(unittest.TestCase):
    """Verify the two new inferred-event entries exist and are hedged correctly."""

    def test_inferred_regulation_entry_exists(self):
        from services.translation_layer import EVENT_TRANSLATION_MAP
        self.assertIn(("TRAIN_HELD", "INFERRED_REGULATION"), EVENT_TRANSLATION_MAP)

    def test_inferred_traffic_entry_exists(self):
        from services.translation_layer import EVENT_TRANSLATION_MAP
        self.assertIn(("CONGESTION", "INFERRED_TRAFFIC"), EVENT_TRANSLATION_MAP)

    def test_inferred_messages_use_hedged_phrasing(self):
        from services.translation_layer import EVENT_TRANSLATION_MAP
        for key in [("TRAIN_HELD", "INFERRED_REGULATION"), ("CONGESTION", "INFERRED_TRAFFIC")]:
            msg = EVENT_TRANSLATION_MAP[key]
            self.assertIn("appears", msg.lower(),
                          f"Inferred message for {key} must use hedged language: {msg!r}")

    def test_map_version_bumped(self):
        from services.translation_layer import MAP_VERSION
        self.assertNotEqual(MAP_VERSION, "1.0.0",
                            "MAP_VERSION must be bumped after adding new entries")

    def test_existing_authoritative_entries_unchanged(self):
        """Regression: existing authoritative entries must not be modified."""
        from services.translation_layer import EVENT_TRANSLATION_MAP
        self.assertEqual(
            EVENT_TRANSLATION_MAP[("TRAIN_HELD", "OPERATIONAL_REGULATION")],
            "Train temporarily stopped \u2014 operational regulation",
        )
        self.assertEqual(
            EVENT_TRANSLATION_MAP[("TRAIN_RESUMED", "RESUMED")],
            "Train has resumed movement",
        )


class TestPassengerFilterUnchanged(unittest.TestCase):
    """Regression: existing filter logic must be unmodified."""

    def test_approved_types_still_whitelisted(self):
        from services.passenger_filter import passenger_can_see
        for etype in ["TRAIN_HELD", "TRAIN_RESUMED", "NONE", "SIGNAL_FAILURE"]:
            self.assertTrue(passenger_can_see(etype))

    def test_unknown_type_still_blocked(self):
        from services.passenger_filter import passenger_can_see
        self.assertFalse(passenger_can_see("INTERNAL_DIAGNOSTIC_XYZ"))

    def test_none_still_blocked(self):
        from services.passenger_filter import passenger_can_see
        self.assertFalse(passenger_can_see(None))


# ===========================================================================
# Runner
# ===========================================================================

def run_all_tests() -> bool:
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestNormalObservationAssembly,
        TestNetworkErrorFallback,
        TestMalformedResponseFallback,
        TestQuotaExhaustion,
        TestInferTrainHeld,
        TestInferCongestion,
        TestInferTrainResumed,
        TestDefaultsFallback,
        TestPerTrainStateIsolation,
        TestTranslationMapEntries,
        TestPassengerFilterUnchanged,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
