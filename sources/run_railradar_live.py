"""
sources/run_railradar_live.py
==============================
RailRadar-Backed Live Journey Runner (Build Brief v8)

Wires together:
  RateLimitedPoller  →  from_railradar  →  DelayTrendTracker
  →  predict()  →  filter_and_prepare()  →  log_update

Key behaviours
--------------
- Rate-limited: polls at most every ``min_interval_seconds`` per train
- Quota-safe: ``should_poll`` returns False near the monthly ceiling
- Graceful degradation: ``RailRadarError`` (network, malformed, success=false)
  falls back to the last known observation — never crashes the loop
- Stops cleanly when ``status == "terminated"`` or the user presses Ctrl-C
- All existing components (predict, filter_and_prepare) are UNCHANGED

Usage
-----
  python sources/run_railradar_live.py --train 12952 --api-key rr_live_XXXX

Environment variable ``RAILRADAR_API_KEY`` is also accepted as a fallback
so the key doesn't have to appear in shell history.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---- Project path setup -----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.event_inference import DelayTrendTracker
from services.railradar_poller import RateLimitedPoller
from services.passenger_filter import filter_and_prepare
from sources.railradar_source import RailRadarError, from_railradar
from src.predict import predict

# ---- Logging ----------------------------------------------------------------
logger = logging.getLogger("railradar_live")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Structured log helper
# ---------------------------------------------------------------------------

def log_update(
    train_no: str,
    obs: dict,
    result: dict,
    inferred: dict,
    passenger_view: Optional[dict],
    verbose: bool = True,
) -> dict:
    """
    Emit a human-readable console line and return a structured dict
    (same pattern as the simulator's run_live_simulation.py).
    """
    status_str = passenger_view["status"] if passenger_view else obs.get("raw_status", "UNKNOWN")
    reason_str = passenger_view["reason"] if passenger_view else "Normal operational running"

    entry = {
        "ts":                  datetime.now().isoformat(),
        "train_no":            train_no,
        "station_code":        obs.get("station_code"),
        "source":              "railradar",
        "raw_status":          obs.get("raw_status"),
        "delay_min":           obs.get("previous_station_delay"),
        "speed_kmph":          obs.get("raw_speed_kmph"),
        "inferred_event":      inferred.get("event_type"),
        "confidence_note":     inferred.get("confidence_note"),
        "status":              status_str,
        "reason":              reason_str,
        "predicted_eta":       result.get("predicted_eta"),
        "eta_range":           result.get("predicted_eta_range"),
        "predicted_delay_min": result.get("predicted_delay_min"),
        "confidence":          result.get("confidence"),
        "fetched_at":          obs.get("fetched_at"),
    }

    if verbose:
        print(
            f"[{entry['ts']}] Train {train_no} @ {obs.get('station_code')} | "
            f"delay={obs.get('previous_station_delay')} min | "
            f"inferred={inferred.get('event_type')} | "
            f"ETA={result.get('predicted_eta')} | "
            f"conf={result.get('confidence')} | "
            f"{reason_str}"
        )

    return entry


# ---------------------------------------------------------------------------
# Core live journey loop
# ---------------------------------------------------------------------------

def run_journey_from_railradar(
    train_no: str,
    api_key: str,
    station_master: Optional[dict] = None,
    min_interval_seconds: int = 120,
    monthly_quota: int = 1_000,
    json_log_path: Optional[Path] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Run a RailRadar-backed live journey session until the train terminates
    or the caller interrupts (Ctrl-C).

    Parameters
    ----------
    train_no : str
        Train number (e.g. ``"12952"``).
    api_key : str
        RailRadar Bearer token.
    station_master : dict | None
        Optional ``{station_code: {...}}`` lookup for distance enrichment.
    min_interval_seconds : int
        Minimum seconds between successive polls per train.
    monthly_quota : int
        RailRadar monthly request ceiling (default 1,000).
    json_log_path : Path | None
        If given, write structured JSON-lines log to this path.
    verbose : bool
        Whether to print human-readable updates to stdout.

    Returns
    -------
    list[dict]
        Chronological list of structured update dicts emitted during the session.
    """
    poller  = RateLimitedPoller(
        api_key=api_key,
        monthly_quota=monthly_quota,
        min_interval_seconds=min_interval_seconds,
    )
    tracker = DelayTrendTracker(window=5)
    history: list[dict] = []
    last_obs: Optional[dict] = None  # fallback when poll is rate-limited / fails

    poller.start()
    session_journey_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(
        "Starting RailRadar live session for train %s (anchored journey_date=%s).",
        train_no,
        session_journey_date,
    )

    # Pre-fetch route (long-lived cache, counts once against quota)
    try:
        poller.get_route_cached(train_no)
    except RailRadarError as exc:
        logger.error("Could not fetch initial route for train %s: %s", train_no, exc)
        # Non-fatal — resolve_position will fall back gracefully

    try:
        while True:
            # ---- Poll -------------------------------------------------------
            live_data: Optional[dict] = None
            try:
                live_data = poller.poll_live(train_no)
            except RailRadarError as exc:
                # Network / API / malformed JSON — log, fall back, never crash
                logger.error(
                    "RailRadar poll failed for train %s: %s. "
                    "Falling back to last known observation.",
                    train_no,
                    exc,
                )

            if live_data is None:
                # Rate-limited (should_poll returned False) or API error
                if last_obs is None:
                    # No observation at all yet — wait and retry
                    time.sleep(10)
                    continue
                # Reuse last observation; skip tracker update (no new data)
                obs = last_obs
                inferred = tracker.infer_event()
            else:
                # ---- Build observation --------------------------------------
                try:
                    obs = from_railradar(
                        train_no=train_no,
                        api_key=api_key,
                        station_master=station_master,
                        route_cache=poller.route_cache,
                        journey_date=session_journey_date,
                    )
                except RailRadarError as exc:
                    logger.error(
                        "from_railradar failed for train %s: %s. Using last obs.",
                        train_no,
                        exc,
                    )
                    obs = last_obs if last_obs is not None else {}

                obs["journey_date"] = session_journey_date
                last_obs = obs

                # ---- Infer event from delay / speed trend -------------------
                tracker.update(
                    timestamp  = live_data.get("lastUpdatedAt"),
                    delay_min  = float(live_data.get("delayMinutes", 0)),
                    speed_kmph = live_data.get("currentLocation", {}).get("speed"),
                )
                inferred = tracker.infer_event()
                obs.update(inferred)   # merge event_type / reason_category into obs

            # ---- Guard: skip predict if obs is incomplete (e.g. first-poll failure) --
            _required = {"train_no", "journey_date", "station_code"}
            if not _required.issubset(obs.keys()):
                logger.warning(
                    "Incomplete observation for train %s — skipping predict until valid data arrives.",
                    train_no,
                )
                time.sleep(10)
                continue

            # ---- Predict (existing v6 model, completely unchanged) ----------
            result = predict(obs)

            # ---- Passenger filter (existing v7, unchanged) ------------------
            passenger_view = (
                filter_and_prepare(obs)
                if inferred.get("event_type") != "NONE"
                else None
            )

            # ---- Emit update ------------------------------------------------
            entry = log_update(train_no, obs, result, inferred, passenger_view, verbose)
            history.append(entry)

            # ---- Termination check ------------------------------------------
            if live_data and live_data.get("status") == "terminated":
                logger.info("Train %s status is 'terminated'. Ending session.", train_no)
                break

            # ---- Wait -------------------------------------------------------
            time.sleep(poller.min_interval_seconds)

    except KeyboardInterrupt:
        logger.info("Session interrupted by user (Ctrl-C). Stopping poller.")
    finally:
        poller.stop()

    # ---- Save JSON log ------------------------------------------------------
    if json_log_path and history:
        out = Path(json_log_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rec in history:
                f.write(json.dumps(rec) + "\n")
        logger.info("Saved %d records to '%s'.", len(history), out)

    logger.info(
        "Session complete — %d updates emitted. Quota summary: %s",
        len(history),
        poller.quota_summary(),
    )
    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RailRadar Live Journey Runner (Build Brief v8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train", required=True,
        help="Train number to track (e.g. 12952)",
    )
    p.add_argument(
        "--api-key", default=None,
        help="RailRadar API key. Defaults to RAILRADAR_API_KEY env var.",
    )
    p.add_argument(
        "--interval", type=int, default=120,
        help="Minimum seconds between successive polls per train.",
    )
    p.add_argument(
        "--quota", type=int, default=1_000,
        help="Monthly request quota ceiling.",
    )
    p.add_argument(
        "--json", default=None,
        help="Path to write structured JSON-lines log.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress console output (log file still written).",
    )
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("RAILRADAR_API_KEY")
    if not api_key:
        parser.error(
            "Provide --api-key or set the RAILRADAR_API_KEY environment variable."
        )

    run_journey_from_railradar(
        train_no=args.train,
        api_key=api_key,
        min_interval_seconds=args.interval,
        monthly_quota=args.quota,
        json_log_path=Path(args.json) if args.json else None,
        verbose=not args.quiet,
    )
