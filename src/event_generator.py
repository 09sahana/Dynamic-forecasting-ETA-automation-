import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

EVENT_TYPES = [
    'NONE',
    'CONGESTION',
    'TRAIN_HELD',
    'SIGNAL_FAILURE',
    'WEATHER_DISRUPTION',
    'TRACK_MAINTENANCE'
]

# ---------------------------------------------------------------------------
# Independent duration distributions — keyed by event type, NOT derived from
# the delay target. This breaks the target leakage confirmed in the v5 audit.
# ---------------------------------------------------------------------------
DURATION_DISTRIBUTIONS = {
    'TRAIN_HELD':        {'mean_log': np.log(15), 'sigma': 0.6},
    'CONGESTION':        {'mean_log': np.log(10), 'sigma': 0.5},
    'SIGNAL_FAILURE':    {'mean_log': np.log(20), 'sigma': 0.65},
    'WEATHER_DISRUPTION':{'mean_log': np.log(25), 'sigma': 0.7},
    'TRACK_MAINTENANCE': {'mean_log': np.log(30), 'sigma': 0.5},
}

# Dispatcher prior — typical durations known at incident-start, NOT retroactive.
# This is a legitimate, non-leaking feature (a published lookup table).
EXPECTED_DURATION_PRIOR = {
    'TRAIN_HELD': 15,
    'CONGESTION': 10,
    'SIGNAL_FAILURE': 20,
    'WEATHER_DISRUPTION': 25,
    'TRACK_MAINTENANCE': 30,
    'NONE': 0,
}


def _sample_duration_by_type(event_type: str, rng: np.random.RandomState) -> float:
    """Sample total event duration from an independent lognormal distribution.

    Duration is a property of the *event type*, not the eventual delay value.
    The only natural correlation with simulated_delay_min is indirect:
    severe incident types tend to cause severe delays — not a direct derivation.
    """
    params = DURATION_DISTRIBUTIONS.get(event_type, DURATION_DISTRIBUTIONS['TRAIN_HELD'])
    return float(rng.lognormal(mean=params['mean_log'], sigma=params['sigma']))


def _compute_trigger_probability(
    historical_avg_delay_min: float,
    weather_risk_score: float,
    season: str,
    is_holiday: int,
) -> float:
    """
    Compute the probability of an event occurring using ONLY pre-arrival,
    observable signals — no target (simulated_delay_min) involved.
    """
    # Base probability from historical chronic delay tendency
    hist_factor = min(0.40, historical_avg_delay_min / 100.0)

    # Weather contribution
    weather_factor = min(0.30, weather_risk_score * 0.20)

    # Seasonal uplift
    season_uplift = 0.05 if season == 'Monsoon' else 0.0

    # Holiday/weekend congestion
    holiday_uplift = 0.03 if is_holiday else 0.0

    prob = hist_factor + weather_factor + season_uplift + holiday_uplift
    return float(min(prob, 0.80))  # cap at 80% max trigger rate


def _sample_event_type(
    weather_risk_score: float,
    season: str,
    rng: np.random.RandomState,
) -> str:
    """
    Sample event type from seasonal / weather-conditional probabilities.
    No dependence on the delay target.
    """
    is_monsoon = (season == 'Monsoon')

    if is_monsoon or weather_risk_score > 0.6:
        # High-weather scenario — weather disruption more likely
        weights = [0.20, 0.30, 0.15, 0.25, 0.10]  # CONGESTION,TRAIN_HELD,SIGNAL_FAILURE,WEATHER_DISRUPTION,TRACK_MAINTENANCE
    elif weather_risk_score > 0.3:
        weights = [0.35, 0.25, 0.20, 0.10, 0.10]
    else:
        weights = [0.40, 0.25, 0.20, 0.05, 0.10]

    types = ['CONGESTION', 'TRAIN_HELD', 'SIGNAL_FAILURE', 'WEATHER_DISRUPTION', 'TRACK_MAINTENANCE']
    chosen = rng.choice(types, p=weights)
    return str(chosen)


def generate_events_for_dataframe(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic operational events using ONLY pre-arrival, observable signals.

    v6 fix — NO dependence on simulated_delay_min or any post-hoc outcome.
    Exposes two leakage-free event features to the model:
      - event_elapsed_min : how long the event has been running *so far*
                            (simulated as a random fraction of total duration,
                            representing the observation point mid-event)
      - event_expected_min: dispatcher's published prior for this incident type

    The internal field `event_duration_truth_only` is retained only for simulator
    realism and ground-truth consistency checks — it is NEVER added to feature cols.
    """
    df = df.copy()
    rng = np.random.RandomState(random_state)
    n = len(df)

    event_types        = np.array(['NONE'] * n, dtype=object)
    event_elapsed_min  = np.zeros(n, dtype=np.float32)
    event_expected_min = np.zeros(n, dtype=np.float32)
    # Internal truth — never exposed to model
    event_duration_truth_only = np.zeros(n, dtype=np.float32)

    # Pull pre-arrival observable columns (with safe defaults)
    hist_avg    = df['historical_avg_delay_min'].values if 'historical_avg_delay_min' in df.columns else np.zeros(n)
    weather     = df['weather_risk_score'].values       if 'weather_risk_score' in df.columns       else np.zeros(n)
    seasons     = df['season'].values                   if 'season' in df.columns                   else np.full(n, 'Summer')
    is_holiday  = df['is_holiday'].values               if 'is_holiday' in df.columns               else np.zeros(n, dtype=int)

    for i in range(n):
        trigger_prob = _compute_trigger_probability(
            historical_avg_delay_min=float(hist_avg[i]),
            weather_risk_score=float(weather[i]),
            season=str(seasons[i]),
            is_holiday=int(is_holiday[i]),
        )

        if rng.random() > trigger_prob:
            # No event — all fields stay at zero / NONE
            event_types[i] = 'NONE'
            continue

        etype = _sample_event_type(
            weather_risk_score=float(weather[i]),
            season=str(seasons[i]),
            rng=rng,
        )
        event_types[i] = etype

        # Sample true duration independently of the target
        total_dur = _sample_duration_by_type(etype, rng)
        event_duration_truth_only[i] = round(float(total_dur), 1)

        # Simulate observation point: somewhere between 5% and 95% into the event.
        # This teaches the model the relationship between partial elapsed time and
        # eventual delay — the real prediction task.
        obs_fraction = rng.uniform(0.05, 0.95)
        event_elapsed_min[i] = round(float(total_dur * obs_fraction), 1)

        # Dispatcher prior — a lookup, not derived from this row's outcome
        event_expected_min[i] = float(EXPECTED_DURATION_PRIOR.get(etype, 15))

    df['event_type']               = event_types
    df['event_elapsed_min']        = event_elapsed_min
    df['event_expected_min']       = event_expected_min
    # Truth-only — simulator realism, never in FINAL_FEATURE_COLS
    df['event_duration_truth_only'] = event_duration_truth_only
    return df


def inject_event(
    row: pd.DataFrame,
    event_type: str,
    event_elapsed_min: float,
    event_expected_min: float = None,
) -> pd.DataFrame:
    """
    Inject a specific operational event into a single row for scenario testing.

    Parameters
    ----------
    event_elapsed_min  : how long the event has been running so far (known at observation time)
    event_expected_min : dispatcher's duration prior; if None, auto-filled from lookup table
    """
    row = row.copy()
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type: {event_type}. Allowed: {EVENT_TYPES}")
    row['event_type'] = event_type
    row['event_elapsed_min'] = float(event_elapsed_min)
    row['event_expected_min'] = float(
        event_expected_min if event_expected_min is not None
        else EXPECTED_DURATION_PRIOR.get(event_type, 15)
    )
    return row


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    from src.preprocessing import load_raw_data, preprocess_data

    raw = load_raw_data(RAW_DATA_PATH)
    clean = preprocess_data(raw)
    events_df = generate_events_for_dataframe(clean.head(1000))
    print("Generated event value counts:")
    print(events_df['event_type'].value_counts())
    print("\nSample with active events:")
    active = events_df[events_df['event_type'] != 'NONE']
    print(active[['train_no', 'station_code', 'simulated_delay_min',
                   'event_type', 'event_elapsed_min', 'event_expected_min',
                   'event_duration_truth_only']].head(5))
    print("\nCorrelation of new fields with target (should be well below 0.75):")
    for col in ['event_elapsed_min', 'event_expected_min']:
        corr = events_df[[col, 'simulated_delay_min']].corr().iloc[0, 1]
        print(f"  {col:25s}: {corr:.4f}")
