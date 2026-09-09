# ML Pipeline — Dynamic ETA Delay Prediction

Dynamic ETA delay prediction system for train station stops built using Python, Scikit-Learn, LightGBM, and Pandas.

## Project Structure

```
ml-pipeline/
├── data/
│   ├── raw/
│   │   └── synthetic_training_dataset_FINAL_clean.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_exploration.ipynb        # EDA, feature walkthrough, model comparison
│   └── generate_notebook.py        # Generator for Jupyter notebook
│
├── src/
│   ├── config.py                   # Paths, constants, split date, target column
│   ├── preprocessing.py            # Null handling, cancellation filter, deduplication
│   ├── features.py                 # Feature engineering pipeline
│   ├── train.py                    # Train & compare 6 models, auto-save best
│   ├── evaluate.py                 # Evaluation metrics & diagnostic charts
│   └── predict.py                  # Single & batch prediction interface
│
├── models/
│   ├── best_model.joblib           # Saved model bundle (LightGBM)
│   ├── metrics_summary.json        # Benchmark metrics summary
│   └── evaluation_plots.png        # Residuals, scatter, & feature importance charts
│
├── requirements.txt
└── README.md
```

## Dataset & Target

- **Raw Dataset**: `synthetic_training_dataset_FINAL_clean.csv` (108,800 rows, 23 columns, Jan 1 – Dec 31 2023).
- **Target Variable**: `simulated_delay_min` (Delay in minutes at each station stop).
- **Target Transformation**: $\log(1 + y)$ target compression, with inverse $\exp(\hat{y}) - 1$ at inference to resolve large-delay underprediction.
- **Data Ceiling Flag**: `delay_is_capped` isolates the 4.4% 360-minute synthetic generator cap for honest segmented evaluation.
- **ETA Formula**: $\text{Predicted ETA} = \text{scheduled\_arrival} + \text{Predicted Delay}$.
- **Cancellation Filter & Assertion**: Cancelled runs (`simulated_cancelled_flag == True`, 1,908 rows) are strictly asserted and excluded prior to regression training, and explicitly flagged at inference.

## Features Engineered

1. **Schedule Context**: `is_holiday`, `is_weekend`, `season_encoded`, `day_of_week_encoded`.
2. **Live Delay Signals**: `previous_station_delay`, `delay_trend`, `delay_deviation_ratio` (relative station deviation), `rolling_avg_delay_last_3` (3-stop rolling mean).
3. **Stacked Severe Risk Feature**: `severe_delay_risk` (out-of-fold probability from 0.98 AUC classifier).
4. **Route Position**: `pct_journey_complete`, `stops_density_remaining`, `delay_per_km`.
5. **Environment**: `weather_risk_score`, `is_monsoon`.
6. **High-Cardinality Target Encodings**: `station_code_encoded`, `train_no_encoded`.

## Time-Based Split & Model Benchmark

- **Train**: Jan 1 - Oct 31, 2023 (87,579 rows)
- **Test**: Nov 1 - Dec 31, 2023 (17,603 unseen rows)

### Benchmark Results V3 (Unseen Test Set — Stacked Regressors)

| Model | MAE (min) | RMSE (min) | $R^2$ Score | Within $\pm 5$ min | Within $\pm 10$ min |
|---|---|---|---|---|---|
| **LightGBM (Winner)** | **24.9100** | **39.4000** | **0.6800** | **23.18%** | **41.94%** |
| HistGradient Boosting | 25.0600 | 39.4800 | 0.6790 | 23.36% | 41.62% |
| Gradient Boosting | 25.0400 | 39.4600 | 0.6790 | 22.84% | 41.20% |
| Random Forest | 25.2800 | 39.9500 | 0.6710 | 22.88% | 41.41% |
| Decision Tree | 27.6100 | 44.1200 | 0.5990 | 22.80% | 41.49% |
| Linear Regression | 32.7100 | 63.3200 | 0.1740 | 18.30% | 35.60% |
| **Quantile p50 Regressor** | **25.8684** | **43.2711** | **0.6144** | **27.11%** | **46.01%** |

### Key Improvements in Pipeline v3:
- **Quantile Prediction Intervals ($[p_{10}, p_{90}]$)**: Achieved **79.06% coverage** of actual delays on the non-capped unseen test set (practically exact match for the 80% theoretical calibration target).
- **Within $\pm 10$ min**: Median ($p_{50}$) model delivers **46.01%** accuracy within $\pm 10$ min and **27.11%** within $\pm 5$ min.
- **Stacked Risk Signal**: `severe_delay_risk` integrated into the top 10 features (6.12% importance share).
- **Walk-Forward Validation**: 3 rolling temporal splits demonstrate rock-solid stability ($25.14 \pm 0.11$ min MAE, $41.59\% \pm 0.06\%$ within $\pm 10$ min).
- **Calibrated High Confidence**: Delivers **10.24 min MAE** and **77.9% accuracy within $\pm 10$ min**.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train & Benchmark Models
```bash
python src/train.py
```

### 3. Evaluate Best Model & Plot Diagnostic Charts
```bash
python src/evaluate.py
```

### 4. Run Sample Inference Prediction
```bash
python src/predict.py
```
