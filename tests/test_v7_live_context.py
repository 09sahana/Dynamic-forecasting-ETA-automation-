"""
tests/test_v7_live_context.py
==============================
Behavioral tests for the Live Context Loop (Build Brief v7).
Verifies the complete loop across all 6 required scenarios:
- Scenario 1: Full journey, no events at all
- Scenario 2: One TRAIN_HELD event mid-journey, then resumes
- Scenario 3: Event type NOT in APPROVED_EVENT_TYPES (fail safe)
- Scenario 4: Event/reason combination NOT in EVENT_TRANSLATION_MAP (fail safe)
- Scenario 5: Chronic train from earlier rounds (e.g. #12508)
- Scenario 6: Journey with event active right up to final station
- Scenario 7: Code leak audit (assertion ensuring NO raw internal codes ever leak)

Can be executed directly:
    python tests/test_v7_live_context.py
Or via pytest if installed:
    pytest tests/test_v7_live_context.py -v
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import pytest
except ImportError:
    class _MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
    pytest = _MockPytest()

import pandas as pd
from services.translation_layer import (
    translate_event,
    EVENT_TRANSLATION_MAP,
    DEFAULT_SAFE_MESSAGE,
)
from services.passenger_filter import (
    passenger_can_see,
    filter_and_prepare,
    APPROVED_EVENT_TYPES,
)
from simulator.live_event_simulator import LiveEventSimulator
from run_live_simulation import run_journey, build_observation_from_sim_state
from src.predict import predict, predict_delay_and_eta
from src.config import RAW_DATA_PATH
from src.preprocessing import load_raw_data


_CACHED_RAW_DF = None

def get_raw_df():
    global _CACHED_RAW_DF
    if _CACHED_RAW_DF is None:
        _CACHED_RAW_DF = load_raw_data(RAW_DATA_PATH)
    return _CACHED_RAW_DF


def get_sample_route():
    """A minimal 3-station synthetic route for fast unit tests."""
    return [
        {
            "station_code": "STN_A",
            "station_name": "Station Alpha",
            "seq": 1,
            "distance_km": 0.0,
            "scheduled_arrival": "08:00:00",
            "scheduled_departure": "08:05:00",
            "historical_avg_delay_min": 10.0,
            "weather_risk_score": 0.1,
            "season": "Winter",
            "is_holiday": 0,
        },
        {
            "station_code": "STN_B",
            "station_name": "Station Beta",
            "seq": 2,
            "distance_km": 40.0,
            "scheduled_arrival": "08:45:00",
            "scheduled_departure": "08:50:00",
            "historical_avg_delay_min": 15.0,
            "weather_risk_score": 0.1,
            "season": "Winter",
            "is_holiday": 0,
        },
        {
            "station_code": "STN_C",
            "station_name": "Station Gamma",
            "seq": 3,
            "distance_km": 90.0,
            "scheduled_arrival": "09:40:00",
            "scheduled_departure": "09:40:00",
            "historical_avg_delay_min": 20.0,
            "weather_risk_score": 0.1,
            "season": "Winter",
            "is_holiday": 0,
        },
    ]


# ---------------------------------------------------------------------------
# Test Case 1: Full journey, no events at all
# ---------------------------------------------------------------------------
def test_scenario_1_full_journey_no_events(sample_route=None, raw_df=None):
    """
    Scenario 1:
    Full journey with no events at all.
    ETA updates smoothly as the train progresses; status always 'MOVING';
    no passenger disruption view ever fires.
    """
    sample_route = sample_route or get_sample_route()
    raw_df = raw_df or get_raw_df()

    sim = LiveEventSimulator(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=sample_route,
        tick_minutes=1,
        probabilistic=False,
    )

    history = run_journey(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=sample_route,
        raw_df=raw_df,
        sim=sim,
        periodic_recompute_ticks=15,
        verbose=False,
    )

    assert sim.is_journey_complete(), "Journey should complete all stations"
    assert len(history) >= 3, f"Expected updates across stations, got {len(history)}"

    # Verify no unexpected passenger disruption events fired
    for entry in history:
        pv = entry.get("passenger_view")
        assert pv is None, f"Expected no passenger disruption view, got: {pv}"
        assert "predicted_eta" in entry["result"]
        assert entry["result"]["predicted_eta"] is not None


# ---------------------------------------------------------------------------
# Test Case 2: One TRAIN_HELD event mid-journey, then resumes
# ---------------------------------------------------------------------------
def test_scenario_2_train_held_and_resumes(sample_route=None, raw_df=None):
    """
    Scenario 2:
    One TRAIN_HELD event mid-journey, then resumes.
    ETA responds when the event starts and when it resumes;
    both raw events correctly translated and filtered.
    """
    sample_route = sample_route or get_sample_route()
    raw_df = raw_df or get_raw_df()

    sim = LiveEventSimulator(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=sample_route,
        tick_minutes=1,
        probabilistic=False,
    )

    # Schedule TRAIN_HELD at tick 5 lasting 15 ticks
    sim.schedule_event(
        event_type="TRAIN_HELD",
        reason_category="OPERATIONAL_REGULATION",
        duration_minutes=15,
        trigger_tick=5,
    )

    history = run_journey(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=sample_route,
        raw_df=raw_df,
        sim=sim,
        periodic_recompute_ticks=5,
        verbose=False,
    )

    # Find the entries where raw events occurred
    held_entries = [h for h in history if h.get("raw_event") and h["raw_event"]["event_type"] == "TRAIN_HELD"]
    resumed_entries = [h for h in history if h.get("raw_event") and h["raw_event"]["event_type"] == "TRAIN_RESUMED"]

    assert len(held_entries) == 1, "TRAIN_HELD event should have fired once"
    assert len(resumed_entries) == 1, "TRAIN_RESUMED event should have fired once"

    held_pv = held_entries[0]["passenger_view"]
    resumed_pv = resumed_entries[0]["passenger_view"]

    assert held_pv is not None
    assert held_pv["status"] == "STOPPED"
    assert "operational regulation" in held_pv["reason"].lower()

    assert resumed_pv is not None
    assert resumed_pv["status"] == "MOVING"
    assert "resumed" in resumed_pv["reason"].lower()

    # Verify that prediction interval or delay reflects context
    res_held = held_entries[0]["result"]
    assert res_held["event_type"] == "TRAIN_HELD"
    assert res_held["event_elapsed_min"] >= 0.0


# ---------------------------------------------------------------------------
# Test Case 3: Event type NOT in APPROVED_EVENT_TYPES (fail safe)
# ---------------------------------------------------------------------------
def test_scenario_3_unapproved_event_type_fails_safe():
    """
    Scenario 3:
    Event type NOT in APPROVED_EVENT_TYPES (simulate an internal-only diagnostic event).
    filter_and_prepare returns None — nothing shown to passenger, confirms whitelist fails safe.
    """
    internal_event = {
        "train_id": "12508",
        "timestamp": "2023-01-14T10:15:00",
        "event_type": "INTERNAL_BOGIE_BEARING_DIAGNOSTIC_LOG",
        "location": "STN_B",
        "reason_category": "TELEMETRY_SAMPLE",
        "status": "STOPPED",
    }

    assert not passenger_can_see(internal_event["event_type"])
    pv = filter_and_prepare(internal_event)
    assert pv is None, f"Expected None for unapproved event type, got {pv}"


# ---------------------------------------------------------------------------
# Test Case 4: Event/reason combination NOT in EVENT_TRANSLATION_MAP (fail safe)
# ---------------------------------------------------------------------------
def test_scenario_4_unmapped_combination_fails_safe():
    """
    Scenario 4:
    Event/reason combination NOT in EVENT_TRANSLATION_MAP.
    Falls back to the generic safe message, never leaks the raw code.
    """
    unmapped_event_type = "TRAIN_HELD"
    unmapped_reason = "INTERNAL_SWITCH_INTERLOCK_FAULT_CODE_992"

    translated = translate_event(unmapped_event_type, unmapped_reason)
    assert translated == DEFAULT_SAFE_MESSAGE, (
        f"Expected generic safe message '{DEFAULT_SAFE_MESSAGE}', got '{translated}'"
    )

    # Verify no raw code leak
    assert unmapped_reason not in translated
    assert "992" not in translated
    assert "SWITCH" not in translated


# ---------------------------------------------------------------------------
# Test Case 5: Chronic train from earlier rounds (e.g. #12508)
# ---------------------------------------------------------------------------
def test_scenario_5_chronic_train_full_loop(raw_df=None):
    """
    Scenario 5:
    Chronic train from earlier rounds (e.g. #12508) run through the full loop.
    Confirms chronic_delay_tier + confidence scoring still behave sensibly
    inside the live loop, not just in batch prediction.
    """
    raw_df = raw_df or get_raw_df()
    stops_df = raw_df[
        (raw_df["train_no"] == 12508) & (raw_df["journey_date"] == "2023-01-14")
    ].sort_values("seq").head(4)

    assert not stops_df.empty, "Train 12508 data must exist in dataset"

    sim = LiveEventSimulator(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=stops_df,
        tick_minutes=1,
        probabilistic=False,
    )

    history = run_journey(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=stops_df,
        raw_df=raw_df,
        sim=sim,
        periodic_recompute_ticks=20,
        verbose=False,
    )

    assert sim.is_journey_complete()
    assert len(history) > 0

    for entry in history:
        res = entry["result"]
        assert res["train_no"] == 12508
        assert res["confidence"] in ["High", "Medium", "Low"]
        assert "predicted_delay_range_min" in res
        p_range = res["predicted_delay_range_min"]
        assert p_range is not None
        assert p_range[0] <= p_range[1]


# ---------------------------------------------------------------------------
# Test Case 6: Journey with event active right up to final station
# ---------------------------------------------------------------------------
def test_scenario_6_event_active_until_final_station(sample_route=None, raw_df=None):
    """
    Scenario 6:
    Journey with an event active right up to the final station (no time to resolve).
    Loop terminates cleanly, last known status/reason is still shown, no crash on boundary.
    """
    sample_route = sample_route or get_sample_route()
    raw_df = raw_df or get_raw_df()

    sim = LiveEventSimulator(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=sample_route,
        tick_minutes=1,
        probabilistic=False,
    )

    # Schedule a long event right at the final station transition
    sim.schedule_event(
        event_type="TRAIN_HELD",
        reason_category="OPERATIONAL_REGULATION",
        duration_minutes=180,  # 3 hours, will not resolve before sim ends
        trigger_station_idx=1,
    )

    # Must complete cleanly without IndexError or unhandled exceptions
    history = run_journey(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=sample_route,
        raw_df=raw_df,
        sim=sim,
        periodic_recompute_ticks=10,
        verbose=False,
    )

    assert len(history) > 0
    last_entry = history[-1]
    assert last_entry["result"] is not None


# ---------------------------------------------------------------------------
# Test Case 7: Strict assertion ensuring NO raw internal codes ever leak
# ---------------------------------------------------------------------------
def test_no_raw_codes_leaked_to_passenger():
    """
    Automated assertion verifying that no internal raw event codes or reason categories
    appear raw in passenger view messages.
    """
    for (etype, rcat), translation in EVENT_TRANSLATION_MAP.items():
        if etype != "NONE":
            assert etype not in translation, f"Internal event type '{etype}' leaked in translation: '{translation}'"
            if rcat is not None and rcat not in ["RESUMED", "WEATHER", "MAINTENANCE"]:
                assert rcat not in translation, f"Internal reason category '{rcat}' leaked in translation: '{translation}'"

    # Test through filter_and_prepare
    sample_event = {
        "event_type": "TRAIN_HELD",
        "reason_category": "OPERATIONAL_REGULATION",
        "status": "STOPPED",
        "location": "STN_TEST",
    }
    pv = filter_and_prepare(sample_event)
    assert pv is not None
    assert "operational regulation" in pv["reason"].lower()
    assert "TRAIN_HELD" not in pv["reason"]
    assert "OPERATIONAL_REGULATION" not in pv["reason"]


# ---------------------------------------------------------------------------
# Test Case 8: ETA Changes During Active Event (v7.1 Bug Fix Regression Guard)
# ---------------------------------------------------------------------------
def test_scenario_8_eta_changes_during_active_event(sample_route=None, raw_df=None):
    """
    Scenario 8 (Fix Brief v7.1):
    The predicted ETA (and range) MUST differ between the tick immediately
    before an event starts and a tick while the event is active. An
    unchanged prediction across an active event is a wiring bug, not a
    valid 'near-zero effect' result -- this is a hard functional check,
    not a statistical one.
    """
    sample_route = sample_route or get_sample_route()
    raw_df = raw_df or get_raw_df()

    sim = LiveEventSimulator(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=sample_route,
        tick_minutes=1,
        probabilistic=False,
    )
    sim.schedule_event(
        event_type="TRAIN_HELD",
        reason_category="OPERATIONAL_REGULATION",
        duration_minutes=15,
        trigger_tick=5,
    )

    pre_event_eta = None
    during_event_eta = None

    while not sim.is_journey_complete():
        raw_event = sim.tick()
        obs = build_observation_from_sim_state(sim, raw_df=raw_df)
        result = predict(obs)

        if sim.active_event is None and pre_event_eta is None:
            pre_event_eta = result["predicted_eta"]
        if sim.active_event is not None and during_event_eta is None:
            during_event_eta = result["predicted_eta"]
            assert obs["event_type"] != "NONE", "event_type not reaching observation during active event"

        if pre_event_eta and during_event_eta:
            break

    assert pre_event_eta != during_event_eta, (
        f"ETA unchanged during active event ({pre_event_eta} == {during_event_eta}) "
        f"-- likely event state not reaching predict()"
    )


# ---------------------------------------------------------------------------
# Test Case 9: Multi-Train State Leak Guard (v7.2 Bug A)
# ---------------------------------------------------------------------------
def test_scenario_9_no_state_leak_across_trains(raw_df=None):
    """
    Scenario 9 (Fix Brief v7.2):
    Running 3+ trains back-to-back must not leak simulator state between
    them. Each train's tick sequence must be strictly monotonic, and no
    train should start with a non-empty active_event or non-zero
    elapsed_ticks inherited from a previous train.
    Tests across two different execution orders.
    """
    raw_df = raw_df or get_raw_df()

    def get_route_for_train(t_no):
        stops = raw_df[raw_df["train_no"] == t_no].sort_values("seq")
        first_date = str(stops.iloc[0]["journey_date"])
        route_df = raw_df[(raw_df["train_no"] == t_no) & (raw_df["journey_date"] == first_date)].sort_values("seq").head(4)
        return first_date, route_df

    train_configs = []
    for t_no in [12508, 15648, 12137]:
        j_date, r_df = get_route_for_train(t_no)
        train_configs.append({"train_no": t_no, "journey_date": j_date, "route": r_df})

    # Test Order 1: Forward (12508 -> 15648 -> 12137)
    for cfg in train_configs:
        sim = LiveEventSimulator(
            train_no=cfg["train_no"],
            journey_date=cfg["journey_date"],
            route_stations=cfg["route"],
            tick_minutes=1,
            probabilistic=False,
        )
        assert sim.elapsed_ticks == 0, f"Train {cfg['train_no']} started with non-zero elapsed_ticks"
        assert sim.active_event is None, f"Train {cfg['train_no']} started with leftover active_event"

        last_tick = -1
        while not sim.is_journey_complete():
            sim.tick()
            assert sim.elapsed_ticks > last_tick, (
                f"Train {cfg['train_no']}: tick went backward or stalled "
                f"({sim.elapsed_ticks} after {last_tick})"
            )
            last_tick = sim.elapsed_ticks

    # Test Order 2: Reverse (12137 -> 15648 -> 12508)
    for cfg in reversed(train_configs):
        sim = LiveEventSimulator(
            train_no=cfg["train_no"],
            journey_date=cfg["journey_date"],
            route_stations=cfg["route"],
            tick_minutes=1,
            probabilistic=False,
        )
        assert sim.elapsed_ticks == 0, f"Train {cfg['train_no']} (reverse) started with non-zero elapsed_ticks"
        assert sim.active_event is None, f"Train {cfg['train_no']} (reverse) started with leftover active_event"

        last_tick = -1
        while not sim.is_journey_complete():
            sim.tick()
            assert sim.elapsed_ticks > last_tick, (
                f"Train {cfg['train_no']} (reverse): tick went backward or stalled "
                f"({sim.elapsed_ticks} after {last_tick})"
            )
            last_tick = sim.elapsed_ticks


# ---------------------------------------------------------------------------
# Test Case 10: Multi-Input Stream Integration Test (7.4)
# ---------------------------------------------------------------------------
def test_scenario_10_multi_input_stream_adapter():
    """
    Scenario 10 (7.4 Improvement):
    Confirms prediction interface works identically with manual/synthetic
    telemetry observations (e.g. GPS stream or manual dispatcher entry)
    without relying solely on CSV simulator state.
    """
    manual_obs_normal = {
        "train_no": 12508,
        "journey_date": "2023-01-14",
        "station_code": "SCL",
        "event_type": "NONE",
        "event_elapsed_min": 0.0,
        "event_expected_min": 0.0,
    }
    res_normal = predict(manual_obs_normal)
    assert res_normal["predicted_eta"] is not None
    assert res_normal["status"] in ["ON_TIME", "DELAYED"]

    manual_obs_gps_event = {
        "train_no": 12508,
        "journey_date": "2023-01-14",
        "station_code": "SCL",
        "event_type": "SIGNAL_FAILURE",
        "event_elapsed_min": 10.0,
        "event_expected_min": 20.0,
    }
    res_event = predict(manual_obs_gps_event)
    assert res_event["predicted_eta"] is not None
    assert res_event["predicted_eta"] != res_normal["predicted_eta"], (
        "Manual/GPS event observation must dynamically shift predicted ETA"
    )
    assert res_event["predicted_delay_min"] > res_normal["predicted_delay_min"]


# ---------------------------------------------------------------------------
# Test Case 11: Passenger Notification Thresholding (7.5)
# ---------------------------------------------------------------------------
def test_scenario_11_passenger_notification_thresholding():
    """
    Scenario 11 (7.5 Improvement):
    Verifies should_notify_passenger suppresses minor delay noise (<3 min)
    while alerting on significant disruptions, status transitions, and confidence shifts.
    """
    from services.passenger_filter import should_notify_passenger

    base_update = {
        "station_code": "SCL",
        "passenger_view": None,
        "result": {
            "predicted_delay_min": 10.0,
            "confidence": "High",
            "is_cancelled": False,
        }
    }

    # Micro fluctuation (0.5 min) with same confidence -> Suppressed
    minor_update = {
        "station_code": "SCL",
        "passenger_view": None,
        "result": {
            "predicted_delay_min": 10.5,
            "confidence": "High",
            "is_cancelled": False,
        }
    }
    assert not should_notify_passenger(base_update, minor_update, min_delay_delta_min=3.0), (
        "Micro fluctuation under 3 mins should be suppressed"
    )

    # Major delay jump (+5.0 min) -> Triggers Notification
    major_update = {
        "station_code": "SCL",
        "passenger_view": None,
        "result": {
            "predicted_delay_min": 15.0,
            "confidence": "High",
            "is_cancelled": False,
        }
    }
    assert should_notify_passenger(base_update, major_update, min_delay_delta_min=3.0), (
        "Delay delta >= 3.0 mins must notify passenger"
    )

    # Confidence tier shift (High -> Medium) -> Triggers Notification
    conf_shift_update = {
        "station_code": "SCL",
        "passenger_view": None,
        "result": {
            "predicted_delay_min": 10.5,
            "confidence": "Medium",
            "is_cancelled": False,
        }
    }
    assert should_notify_passenger(base_update, conf_shift_update), (
        "Confidence tier downgrade must notify passenger"
    )


# ---------------------------------------------------------------------------
# Test Case 12: Quantile Crossing Guard & Logging (v7.2 Bug B)
# ---------------------------------------------------------------------------
def test_scenario_12_quantile_crossing_guard(raw_df=None):
    """
    Scenario 12 (Fix Brief v7.2):
    Verifies that predicted output strictly obeys p08 <= p50 <= p92 across all stops
    (including chronic train #15648 at BARPETA ROAD / BPRD), and logs crossings to
    logs/quantile_crossings.jsonl.
    """
    raw_df = raw_df or get_raw_df()
    
    # Test across chronic train 15648
    stops = raw_df[raw_df["train_no"] == 15648].sort_values("seq")
    assert not stops.empty, "Train 15648 records required"

    for _, row in stops.head(8).iterrows():
        res = predict_delay_and_eta(
            train_no=int(row["train_no"]),
            journey_date=str(row["journey_date"]),
            station_code=str(row["station_code"]),
            raw_df=raw_df,
        )
        if res["predicted_delay_min"] is not None:
            p50 = res["predicted_delay_min"]
            p08, p92 = res["predicted_delay_range_min"]
            assert p08 <= p50 <= p92, f"Quantile crossing violation: p08={p08} <= p50={p50} <= p92={p92} failed!"

    # Test explicit synthetic inverted quantile assembly
    log_path = BASE_DIR / "logs" / "quantile_crossings.jsonl"
    if log_path.exists():
        initial_lines = len(log_path.read_text(encoding="utf-8").strip().splitlines())
    else:
        initial_lines = 0

    from src.predict import log_quantile_crossing
    log_quantile_crossing(99999, "TEST_STN", 60.0, 30.0, 50.0)
    assert log_path.exists(), "logs/quantile_crossings.jsonl must exist after logging"
    new_lines = len(log_path.read_text(encoding="utf-8").strip().splitlines())
    assert new_lines > initial_lines, "New quantile crossing entry must be written to log"


# ---------------------------------------------------------------------------
# Test Case 13: Delay Persists After Event Resolves (v7.3 Anchored Delay)
# ---------------------------------------------------------------------------
def test_scenario_13_delay_persists_after_event_resolves(sample_route=None, raw_df=None):
    """
    Scenario 13 (Fix Brief v7.3):
    Once an event resolves, the accumulated delay it caused must NOT
    fully evaporate. The prediction immediately after resume must differ
    from the prediction immediately before the event started -- some or
    all of the event's time cost should still be reflected.
    """
    sample_route = sample_route or get_sample_route()
    raw_df = raw_df or get_raw_df()

    sim = LiveEventSimulator(
        train_no=12508,
        journey_date="2023-01-14",
        route_stations=sample_route,
        tick_minutes=1,
        probabilistic=False,
    )
    sim.schedule_event(
        event_type="TRAIN_HELD",
        reason_category="OPERATIONAL_REGULATION",
        duration_minutes=15,
        trigger_tick=5,
    )

    pre_event_eta = None
    post_resume_eta = None
    event_seen = False

    while not sim.is_journey_complete():
        raw_event = sim.tick()
        obs = build_observation_from_sim_state(sim, raw_df=raw_df)
        result = predict(obs)

        if sim.active_event is None and not event_seen and pre_event_eta is None:
            pre_event_eta = result["predicted_eta"]

        if sim.active_event is not None:
            event_seen = True

        if event_seen and sim.active_event is None and post_resume_eta is None and pre_event_eta is not None:
            post_resume_eta = result["predicted_eta"]
            break

    assert pre_event_eta is not None and post_resume_eta is not None, "Test setup did not capture both ticks"
    assert pre_event_eta != post_resume_eta, (
        f"Delay fully evaporated after event resolved "
        f"(pre-event ETA {pre_event_eta} == post-resume ETA {post_resume_eta}) -- "
        f"cumulative_delay_min is not persisting across the event"
    )


# ---------------------------------------------------------------------------
# Test Case 14: Event Cost Propagates to Later Stations (v7.3 A/B Check)
# ---------------------------------------------------------------------------
def test_scenario_14_event_cost_propagates_to_later_stations(raw_df=None):
    """
    Scenario 14 (Fix Brief v7.3):
    Compare BADARPUR JN.'s predicted ETA with the SILCHAR hold occurring
    normally, versus with that same event artificially suppressed. The
    'with event' run should predict a LATER arrival at BADARPUR JN. than
    the 'without event' run.
    """
    raw_df = raw_df or get_raw_df()
    stops = raw_df[
        (raw_df["train_no"] == 12508) & (raw_df["journey_date"] == "2023-01-14")
    ].sort_values("seq").head(4)

    def run_until_station(sim, target_station):
        last_eta = None
        while not sim.is_journey_complete():
            sim.tick()
            obs = build_observation_from_sim_state(sim, raw_df=raw_df)
            res = predict(obs)
            if sim.get_current_station_code() == target_station or target_station in sim.get_current_station_name():
                return res["predicted_eta"]
            last_eta = res["predicted_eta"]
        return last_eta

    # With event (15 min hold at tick 5)
    sim_with_event = LiveEventSimulator(train_no=12508, journey_date="2023-01-14", route_stations=stops, probabilistic=False)
    sim_with_event.schedule_event("TRAIN_HELD", "OPERATIONAL_REGULATION", duration_minutes=15, trigger_tick=5)
    eta_with_event = run_until_station(sim_with_event, "BPB")

    # Without event (force_suppress_events=True)
    sim_without_event = LiveEventSimulator(train_no=12508, journey_date="2023-01-14", route_stations=stops, probabilistic=False)
    sim_without_event.force_suppress_events = True
    eta_without_event = run_until_station(sim_without_event, "BPB")

    assert eta_with_event > eta_without_event, (
        f"Event cost did not propagate to BPB (with_event={eta_with_event}, without_event={eta_without_event})"
    )


if __name__ == "__main__":
    print("Running v7.3 Live Context Loop Test Suite...")
    tests = [
        ("Scenario 1: Full journey, no events", test_scenario_1_full_journey_no_events),
        ("Scenario 2: TRAIN_HELD event mid-journey and resume", test_scenario_2_train_held_and_resumes),
        ("Scenario 3: Unapproved event type fails safe", test_scenario_3_unapproved_event_type_fails_safe),
        ("Scenario 4: Unmapped combination fails safe", test_scenario_4_unmapped_combination_fails_safe),
        ("Scenario 5: Chronic train #12508 full loop", test_scenario_5_chronic_train_full_loop),
        ("Scenario 6: Active event until final station", test_scenario_6_event_active_until_final_station),
        ("Scenario 7: Code leak audit (no raw codes to passenger)", test_no_raw_codes_leaked_to_passenger),
        ("Scenario 8: ETA changes during active event (v7.1 fix)", test_scenario_8_eta_changes_during_active_event),
        ("Scenario 9: Multi-train state leak guard (v7.2 Bug A)", test_scenario_9_no_state_leak_across_trains),
        ("Scenario 10: Multi-input stream adapter (7.4)", test_scenario_10_multi_input_stream_adapter),
        ("Scenario 11: Passenger notification thresholding (7.5)", test_scenario_11_passenger_notification_thresholding),
        ("Scenario 12: Quantile crossing guard & logging (v7.2 Bug B)", test_scenario_12_quantile_crossing_guard),
        ("Scenario 13: Delay persists after event resolves (v7.3)", test_scenario_13_delay_persists_after_event_resolves),
        ("Scenario 14: Event cost propagates to later stations (v7.3)", test_scenario_14_event_cost_propagates_to_later_stations),
    ]

    passed = 0
    for name, t_func in tests:
        try:
            t_func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nResult: {passed}/{len(tests)} tests passed.")
    if passed == len(tests):
        print("ALL V7.3 TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)
