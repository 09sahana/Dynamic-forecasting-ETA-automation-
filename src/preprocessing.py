import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from src.config import TARGET_COL, CANCELLED_COL

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Loads raw dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame, filter_cancelled: bool = True) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Drops unusable columns (visibility_m)
    - Filters out cancelled trips if requested
    - Deduplicates rows
    - Parses dates & timestamps
    - Imputes missing coordinates by station_code mode
    - Imputes missing weather variables by seasonal median
    """
    df = df.copy()
    
    # 0. Add delay_is_capped flag before any transformation if target is present
    if TARGET_COL in df.columns:
        df['delay_is_capped'] = (df[TARGET_COL] >= 360).astype(int)

    # 1. Drop visibility_m (100% null)
    if 'visibility_m' in df.columns:
        df.drop(columns=['visibility_m'], inplace=True)

    # 2. Filter out cancelled trips for regression modeling
    if filter_cancelled and CANCELLED_COL in df.columns:
        row_count_before = len(df)
        df = df[df[CANCELLED_COL] != True].copy()
        df = df[df[CANCELLED_COL] != 1].copy()
        row_count_after = len(df)
        print(f"[PREPROCESSING] Cancelled rows filter: {row_count_before} -> {row_count_after} (removed {row_count_before - row_count_after})")
        assert (df[CANCELLED_COL] == False).all() or (df[CANCELLED_COL] == 0).all() or CANCELLED_COL not in df.columns, \
            "Cancelled rows leaked into training data"
        
    # 3. Deduplicate
    dedup_cols = ['train_no', 'journey_date', 'station_code']
    df.drop_duplicates(subset=dedup_cols, inplace=True)
    
    # 4. Datetime handling
    df['journey_date'] = pd.to_datetime(df['journey_date'])
    
    # 5. Impute lat/lon by station_code mode
    if 'lat' in df.columns and 'station_code' in df.columns:
        df['lat'] = df.groupby('station_code')['lat'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.median()))
    if 'lon' in df.columns and 'station_code' in df.columns:
        df['lon'] = df.groupby('station_code')['lon'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.median()))

    # 6. Impute weather by season median
    if 'temperature_c' in df.columns and 'season' in df.columns:
        df['temperature_c'] = df.groupby('season')['temperature_c'].transform(lambda x: x.fillna(x.median()))
    if 'precipitation_mm' in df.columns and 'season' in df.columns:
        df['precipitation_mm'] = df.groupby('season')['precipitation_mm'].transform(lambda x: x.fillna(x.median()))
        
    # Fill any remaining edge-case nulls with overall medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    return df

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    raw_df = load_raw_data(RAW_DATA_PATH)
    clean_df = preprocess_data(raw_df)
    print(f"Preprocessed dataset shape: {clean_df.shape}")
