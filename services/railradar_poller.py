"""
services/railradar_poller.py
=============================
Rate-Limit-Aware Polling & Caching Layer (Build Brief v8)

The RailRadar free sandbox tier allows 1,000 requests/month.  Naive
continuous polling of even a single train would exhaust the quota in days.
This module enforces:

  - Per-train minimum poll interval (default 120 s)
  - Monthly quota ceiling with 5% safety margin
  - **Persistent** quota counter across process restarts (JSON file)
  - Route caching with long TTL (routes don't change minute-to-minute)
  - Explicit start / stop so polling NEVER runs silently in the background

Design notes
------------
- ``requests_this_month`` is read from / written to
  ``data/railradar_quota.json`` on every poll attempt — a process restart
  does NOT silently reset the counter.
- ``route_cache`` has an in-process TTL; for multi-day sessions a persistent
  route cache could be added, but for demo use the in-memory cache is
  sufficient (routes are fetched once per session per train).
- ``poll_live`` returns ``None`` (not an exception) when rate-limited or
  quota-exhausted — callers fall back to the last known observation.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_QUOTA_FILE = BASE_DIR / "data" / "railradar_quota.json"
_LOG_FILE   = BASE_DIR / "logs" / "railradar_poller.log"

# Module logger — writes to both console (WARNING+) and file (INFO+)
logger = logging.getLogger("railradar_poller")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.WARNING)
    _ch.setFormatter(logging.Formatter("%(levelname)s railradar_poller: %(message)s"))
    logger.addHandler(_fh)
    logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Quota persistence helpers
# ---------------------------------------------------------------------------

def _load_quota(quota_file: Path) -> dict:
    """
    Load the persisted quota record from disk.  Returns a default structure
    if the file does not exist or is malformed.
    """
    if quota_file.exists():
        try:
            with open(quota_file, encoding="utf-8") as f:
                record = json.load(f)
            # Reset counter if we are in a new calendar month
            stored_month = record.get("month", "")
            current_month = datetime.now().strftime("%Y-%m")
            if stored_month != current_month:
                logger.info(
                    "New month (%s vs stored %s) — resetting request counter.",
                    current_month,
                    stored_month,
                )
                return {"month": current_month, "requests_this_month": 0}
            return record
        except (json.JSONDecodeError, KeyError):
            logger.warning("Quota file malformed; starting fresh counter.")

    return {"month": datetime.now().strftime("%Y-%m"), "requests_this_month": 0}


def _save_quota(quota_file: Path, record: dict) -> None:
    """Persist the quota record to disk atomically (write-then-replace)."""
    quota_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = quota_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    tmp.replace(quota_file)


def log_quota_warning(requests_used: int, quota: int) -> None:
    """
    Emit a structured quota-exhaustion warning to both the log file and
    the console.  Called automatically by ``RateLimitedPoller.should_poll``
    when the safety margin is breached.
    """
    msg = (
        f"QUOTA WARNING: {requests_used}/{quota} RailRadar requests used this month "
        f"(>= 95% threshold). Polling suspended until next calendar month."
    )
    logger.warning(msg)


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------

class RateLimitedPoller:
    """
    Rate-limit-aware wrapper around the RailRadar live API.

    Parameters
    ----------
    api_key : str
        RailRadar Bearer token.
    monthly_quota : int
        Maximum API calls allowed per calendar month (default 1,000).
    min_interval_seconds : int
        Minimum seconds between successive live polls for the **same** train
        (default 120 — at most 30 polls/hour per train).
    quota_file : Path | None
        Override path for the persisted quota JSON.  Defaults to
        ``data/railradar_quota.json`` relative to the project root.
    """

    def __init__(
        self,
        api_key: str,
        monthly_quota: int = 1_000,
        min_interval_seconds: int = 120,
        quota_file: Path | None = None,
    ) -> None:
        self.api_key               = api_key
        self.monthly_quota         = monthly_quota
        self.min_interval_seconds  = min_interval_seconds
        self._quota_file           = quota_file or _QUOTA_FILE

        # In-session state
        self.route_cache: dict[str, dict]   = {}   # {train_no: route_data}
        self.last_poll_time: dict[str, float] = {}  # {train_no: unix_timestamp}
        self._active: bool                  = False

        # Load persistent quota (survives restarts)
        self._quota_record = _load_quota(self._quota_file)
        logger.info(
            "Poller initialised — %d/%d requests used this month.",
            self._quota_record["requests_this_month"],
            self.monthly_quota,
        )

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the polling session as active.  No background thread is started;
        the caller controls the loop (see ``run_railradar_live.py``)."""
        self._active = True
        logger.info("RateLimitedPoller session started.")

    def stop(self) -> None:
        """Mark the session as inactive.  ``poll_live`` will return None immediately
        if called after ``stop()``."""
        self._active = False
        logger.info("RateLimitedPoller session stopped.")

    # ------------------------------------------------------------------
    # Core rate-limit logic
    # ------------------------------------------------------------------

    @property
    def requests_this_month(self) -> int:
        return self._quota_record["requests_this_month"]

    def _increment_quota(self) -> None:
        self._quota_record["requests_this_month"] += 1
        _save_quota(self._quota_file, self._quota_record)

    def should_poll(self, train_no: str) -> bool:
        """
        Return ``True`` only when:
        - The session is active
        - Enough time has elapsed since the last poll for this train
        - We are below 95% of the monthly quota

        Returns ``False`` and logs a warning if the quota ceiling is breached.
        """
        if not self._active:
            return False

        # Per-train interval gate
        last = self.last_poll_time.get(train_no, 0.0)
        if time.time() - last < self.min_interval_seconds:
            return False

        # Monthly quota gate (5% safety margin)
        if self.requests_this_month >= self.monthly_quota * 0.95:
            log_quota_warning(self.requests_this_month, self.monthly_quota)
            return False

        return True

    # ------------------------------------------------------------------
    # Route cache
    # ------------------------------------------------------------------

    def get_route_cached(self, train_no: str) -> dict:
        """
        Return the cached route for ``train_no``, fetching from the API once
        if not already in cache.  Route data rarely changes; fetch once,
        reuse for the entire session.

        Counts against the monthly quota on first fetch only.
        """
        from sources.railradar_source import fetch_route  # local import avoids circular dependency at import time

        if train_no not in self.route_cache:
            logger.info("Fetching route for train %s (quota before: %d).", train_no, self.requests_this_month)
            self.route_cache[train_no] = fetch_route(train_no, self.api_key)
            self._increment_quota()
            logger.info("Route cached for train %s (quota after: %d).", train_no, self.requests_this_month)

        return self.route_cache[train_no]

    # ------------------------------------------------------------------
    # Live poll
    # ------------------------------------------------------------------

    def poll_live(self, train_no: str) -> dict | None:
        """
        Poll the live API for ``train_no`` if the rate-limit gate allows it.

        Returns
        -------
        dict
            The ``data`` sub-object from the live API response (same as
            ``fetch_live`` return value).
        None
            Returned (not raised) when rate-limited or quota-exhausted.
            Callers should fall back to the last known observation.

        Raises
        ------
        RailRadarError
            Propagated from ``fetch_live`` — caller must handle explicitly.
        """
        from sources.railradar_source import fetch_live  # local import

        if not self.should_poll(train_no):
            return None

        logger.info(
            "Polling live status for train %s (quota before: %d).",
            train_no,
            self.requests_this_month,
        )
        data = fetch_live(train_no, self.api_key)
        self._increment_quota()
        self.last_poll_time[train_no] = time.time()
        logger.info(
            "Live poll complete for train %s (quota after: %d).",
            train_no,
            self.requests_this_month,
        )
        return data

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def quota_summary(self) -> dict:
        """Return a dict summarising current quota usage — useful for logging."""
        return {
            "requests_this_month": self.requests_this_month,
            "monthly_quota":       self.monthly_quota,
            "pct_used":            round(self.requests_this_month / self.monthly_quota * 100, 1),
            "month":               self._quota_record["month"],
            "active":              self._active,
        }

    def __repr__(self) -> str:
        return (
            f"RateLimitedPoller(trains_tracked={len(self.last_poll_time)}, "
            f"quota={self.requests_this_month}/{self.monthly_quota}, "
            f"active={self._active})"
        )
