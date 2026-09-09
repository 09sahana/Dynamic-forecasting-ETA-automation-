import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from src.event_generator import EVENT_TYPES

FINAL_FEATURE_COLS = [
    'historical_avg_delay_min', 'previous_station_delay', 'delay_trend',
    'delay_deviation_ratio', 'chronic_delay_tier', 'chronic_tier_is_fallback',
    'rolling_avg_delay_last_3', 'distance_remaining', 'distance_km',
    'number_of_stops_remaining', 'stops_density_remaining', 'pct_journey_complete',
    'delay_per_km', 'temperature_c', 'precipitation_mm', 'weather_risk_score',
    'is_holiday', 'is_weekend', 'season_encoded', 'day_of_week_encoded',
    'station_code_encoded', 'train_no_encoded', 'seq',
    # v6 leakage-free event features (event_duration removed — was target-derived)
    'event_type_encoded', 'event_elapsed_min', 'event_expected_min', 'event_active_flag'
]

# Base static feature subset without event features (for comparison/fallback benchmarking)
STATIC_FEATURE_COLS = [
    c for c in FINAL_FEATURE_COLS
    if c not in ['event_type_encoded', 'event_elapsed_min', 'event_expected_min', 'event_active_flag']
]

def get_chronic_tier(train_no: int, tier_lookup: dict, default: int = 3) -> int:
    """
    Production-ready cold-start lookup for chronic delay tier.
    Returns median tier default (3) for unseen trains.
    """
    return tier_lookup.get(train_no, default)

class FeaturePipeline:
    def __init__(self):
        self.season_mapping = {}
        self.day_of_week_mapping = {}
        self.event_type_mapping = {t: i for i, t in enumerate(EVENT_TYPES)}
        self.station_code_target_map = {}
        self.train_no_target_map = {}
        self.chronic_delay_tier_map = {}
        self.default_chronic_tier = 3
        self.global_target_mean = 0.0

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = self.create_base_features(df)
        
        # Fit target encodings on train split
        self.global_target_mean = float(df[target_col].mean())
        self.station_code_target_map = df.groupby('station_code')[target_col].mean().to_dict()
        self.train_no_target_map = df.groupby('train_no')[target_col].mean().to_dict()
        
        # Fit chronic delay tiers on train split only (no data leakage)
        train_stats = df.groupby('train_no')['previous_station_delay'].mean()
        tiers = pd.qcut(train_stats, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
        self.chronic_delay_tier_map = tiers.to_dict()
        self.default_chronic_tier = int(tiers.median()) if not tiers.empty else 3

        # Categorical encodings
        seasons = sorted(df['season'].unique().tolist())
        self.season_mapping = {s: i for i, s in enumerate(seasons)}
        
        days = sorted(df['day_of_week'].unique().tolist())
        self.day_of_week_mapping = {d: i for i, d in enumerate(days)}
        
        df = self.apply_encodings(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_base_features(df)
        df = self.apply_encodings(df)
        return df

    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Ensure sorting by sequence if train_no/journey_date/seq are present
        sort_cols = [c for c in ['train_no', 'journey_date', 'seq'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols).reset_index(drop=True)
        
        # A. Schedule context
        if 'is_holiday' in df.columns:
            df['is_holiday'] = df['is_holiday'].astype(int)
            
        weekend_days = ['Saturday', 'Sunday', 'Sat', 'Sun', 5, 6]
        if 'day_of_week' in df.columns:
            df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d in weekend_days else 0)
        else:
            df['is_weekend'] = 0
        
        # B. Live delay signal
        hist_avg = df['historical_avg_delay_min'] if 'historical_avg_delay_min' in df.columns else 0.0
        prev_delay = df['previous_station_delay'] if 'previous_station_delay' in df.columns else 0.0
        
        df['delay_trend'] = prev_delay - hist_avg
        df['delay_deviation_ratio'] = (prev_delay - hist_avg) / (hist_avg.clip(lower=0) + 1.0)
        
        # Rolling avg delay last 3 stops per train run
        if 'train_no' in df.columns and 'journey_date' in df.columns and 'previous_station_delay' in df.columns:
            df['rolling_avg_delay_last_3'] = df.groupby(['train_no', 'journey_date'])['previous_station_delay'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
        else:
            df['rolling_avg_delay_last_3'] = prev_delay
        
        # C. Route position
        if 'distance_remaining' in df.columns:
            if 'train_no' in df.columns and 'journey_date' in df.columns:
                max_dist = df.groupby(['train_no', 'journey_date'])['distance_remaining'].transform('max')
            else:
                max_dist = df['distance_remaining']
            df['pct_journey_complete'] = 1.0 - (df['distance_remaining'] / (max_dist.replace(0, np.nan)))
            df['pct_journey_complete'] = df['pct_journey_complete'].fillna(0.0).clip(0.0, 1.0)
            
            num_stops = df['number_of_stops_remaining'] if 'number_of_stops_remaining' in df.columns else 0.0
            df['stops_density_remaining'] = num_stops / (df['distance_remaining'] + 1.0)
        else:
            df['pct_journey_complete'] = 0.5
            df['stops_density_remaining'] = 0.0

        dist_km = df['distance_km'] if 'distance_km' in df.columns else 100.0
        df['delay_per_km'] = prev_delay / (dist_km + 1.0)
        
        # D. Environment
        precip = df['precipitation_mm'] if 'precipitation_mm' in df.columns else 0.0
        max_precip = precip.max() if hasattr(precip, 'max') else 1.0
        max_precip = max_precip if pd.notnull(max_precip) and max_precip > 0 else 1.0
        
        season_s = df['season'] if 'season' in df.columns else 'Summer'
        df['is_monsoon'] = (season_s == 'Monsoon').astype(int)
        df['weather_risk_score'] = (precip / max_precip) * (1.5 * df['is_monsoon'] + 1.0)
        
        # E. Operational Events (v6 — leakage-free fields)
        if 'event_type' not in df.columns:
            df['event_type'] = 'NONE'
        # event_elapsed_min: how long the event has been running so far (partial observability)
        if 'event_elapsed_min' not in df.columns:
            df['event_elapsed_min'] = 0.0
        # event_expected_min: dispatcher prior duration for this incident type (lookup table)
        if 'event_expected_min' not in df.columns:
            df['event_expected_min'] = 0.0

        df['event_elapsed_min']  = df['event_elapsed_min'].fillna(0.0).astype(float)
        df['event_expected_min'] = df['event_expected_min'].fillna(0.0).astype(float)
        df['event_active_flag']  = (df['event_type'] != 'NONE').astype(int)

        return df

    def apply_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'season' in df.columns:
            df['season_encoded'] = df['season'].map(self.season_mapping).fillna(-1).astype(int)
        else:
            df['season_encoded'] = -1

        if 'day_of_week' in df.columns:
            df['day_of_week_encoded'] = df['day_of_week'].map(self.day_of_week_mapping).fillna(-1).astype(int)
        else:
            df['day_of_week_encoded'] = -1
        
        if 'station_code' in df.columns:
            df['station_code_encoded'] = df['station_code'].map(self.station_code_target_map).fillna(self.global_target_mean)
        else:
            df['station_code_encoded'] = self.global_target_mean

        if 'train_no' in df.columns:
            df['train_no_encoded'] = df['train_no'].map(self.train_no_target_map).fillna(self.global_target_mean)
            # FIX 3: Cold-start fallback detection & tier assignment
            df['chronic_tier_is_fallback'] = df['train_no'].apply(
                lambda x: 0 if x in self.chronic_delay_tier_map else 1
            ).astype(int)
            df['chronic_delay_tier'] = df['train_no'].apply(
                lambda x: get_chronic_tier(x, self.chronic_delay_tier_map, default=self.default_chronic_tier)
            ).astype(int)
        else:
            df['train_no_encoded'] = self.global_target_mean
            df['chronic_tier_is_fallback'] = 1
            df['chronic_delay_tier'] = self.default_chronic_tier

        # Event type encoding
        df['event_type_encoded'] = df['event_type'].map(self.event_type_mapping).fillna(0).astype(int)
        
        return df

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH, TARGET_COL
    from src.preprocessing import load_raw_data, preprocess_data
    
    raw_df = load_raw_data(RAW_DATA_PATH)
    clean_df = preprocess_data(raw_df)
    
    fe = FeaturePipeline()
    feat_df = fe.fit_transform(clean_df, TARGET_COL)
    print("Engineered features shape:", feat_df[FINAL_FEATURE_COLS].shape)
    print("Features sample:\n", feat_df[FINAL_FEATURE_COLS].head(2))
    
    # Test cold-start tier lookup
    unseen_tier = get_chronic_tier(99999, fe.chronic_delay_tier_map, default=3)
    print(f"Cold-start unseen train 99999 tier: {unseen_tier} (expected 3)")
