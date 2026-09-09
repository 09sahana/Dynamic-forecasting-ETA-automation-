import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Dynamic ETA Delay Prediction — Exploratory & Model Comparison Notebook

This notebook walks through the full end-to-end Machine Learning Pipeline developed according to `ml_pipeline_design.md`:
1. **Raw Data Inspection & Audit**: Null counts, dataset stats, and distributions.
2. **Preprocessing**: Filtering cancelled journeys, handling nulls, and deduplicating.
3. **Feature Engineering**: Deriving live delay signals, schedule context, route position, and weather risk scores.
4. **Time-Based Train/Test Split**: Jan-Oct 2023 (Train) vs Nov-Dec 2023 (Test).
5. **Model Benchmarking**: Comparing Linear Regression, Decision Tree, Random Forest, Gradient Boosting, HistGradient Boosting, and LightGBM.
6. **Best Model Diagnostics**: Feature importances, predicted vs actual scatter, and residual distributions.
7. **Sample ETA Predictions**: Interactive prediction queries.
"""))

# Cell 1: Setup & Imports
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
BASE_DIR = Path.cwd().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.config import RAW_DATA_PATH, PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_DIR, TARGET_COL, SPLIT_DATE
from src.preprocessing import load_raw_data, preprocess_data
from src.features import FeaturePipeline, FINAL_FEATURE_COLS
from src.predict import predict_delay_and_eta

sns.set_theme(style="whitegrid")
print("Environment initialized successfully.")
"""))

# Cell 2: Raw Data Inspection
cells.append(nbf.v4.new_markdown_cell("## 1. Raw Dataset Inspection & Audit"))
cells.append(nbf.v4.new_code_cell("""raw_df = load_raw_data(RAW_DATA_PATH)
print(f"Dataset Shape: {raw_df.shape}")
print("\\nMissing Value Audit:")
print(raw_df.isnull().sum()[raw_df.isnull().sum() > 0])
raw_df.head(3)
"""))

# Cell 3: Delay Distribution Analysis
cells.append(nbf.v4.new_markdown_cell("## 2. Delay Distribution Analysis"))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(raw_df[TARGET_COL], bins=50, kde=True, color="indigo")
plt.title("Overall Simulated Delay Distribution (minutes)")
plt.xlabel("Delay (min)")

plt.subplot(1, 2, 2)
sns.boxplot(data=raw_df, x="season", y=TARGET_COL, palette="Set2")
plt.title("Delay Distribution by Season")
plt.xlabel("Season")
plt.ylabel("Delay (min)")

plt.tight_layout()
plt.show()
"""))

# Cell 4: Feature Engineering Walkthrough
cells.append(nbf.v4.new_markdown_cell("## 3. Preprocessing & Feature Engineering Walkthrough"))
cells.append(nbf.v4.new_code_cell("""clean_df = preprocess_data(raw_df, filter_cancelled=True)
pipeline = FeaturePipeline()
feat_df = pipeline.fit_transform(clean_df, TARGET_COL)

print(f"Clean non-cancelled dataset shape: {clean_df.shape}")
print(f"Engineered feature set shape: {feat_df[FINAL_FEATURE_COLS].shape}")

feat_df[['previous_station_delay', 'delay_trend', 'rolling_avg_delay_last_3', 'pct_journey_complete', 'weather_risk_score']].head()
"""))

# Cell 5: Time-Based Split Visual
cells.append(nbf.v4.new_markdown_cell("## 4. Time-Based Train/Test Split Visualization"))
cells.append(nbf.v4.new_code_cell("""feat_df['journey_date'] = pd.to_datetime(feat_df['journey_date'])
daily_avg = feat_df.groupby('journey_date')[TARGET_COL].mean().reset_index()

plt.figure(figsize=(14, 4))
plt.plot(daily_avg['journey_date'], daily_avg[TARGET_COL], label="Daily Mean Delay (min)", color="teal")
plt.axvline(pd.to_datetime(SPLIT_DATE), color="red", linestyle="--", linewidth=2, label=f"Train/Test Cutoff ({SPLIT_DATE})")
plt.title("Time-Series Split: Jan-Oct 2023 (Train) vs Nov-Dec 2023 (Test)")
plt.xlabel("Date")
plt.ylabel("Average Delay (min)")
plt.legend()
plt.tight_layout()
plt.show()
"""))

# Cell 6: Model Benchmark Results
# Cell 6: Model Benchmark Results
cells.append(nbf.v4.new_markdown_cell("## 5. Model Benchmark & Comparison"))
cells.append(nbf.v4.new_code_cell("""model_bundle = joblib.load(MODEL_DIR / "best_model.joblib")
all_results = (
    model_bundle.get("all_results_v5", {}).get("non_capped")
    or model_bundle.get("all_results_v4", {}).get("non_capped")
    or model_bundle.get("all_results_v2", {}).get("non_capped")
    or model_bundle.get("all_results", {})
)
results_df = pd.DataFrame(all_results).T

print("=" * 65)
print("BENCHMARK COMPARISON TABLE (Non-Capped Test Set Evaluation):")
print("=" * 65)
display(results_df)

plt.figure(figsize=(12, 4))
results_df['MAE'].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Model MAE Comparison on Non-Capped Segment (Lower is better)")
plt.ylabel("Mean Absolute Error (minutes)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
"""))

# Cell 7: Best Model Diagnostics
cells.append(nbf.v4.new_markdown_cell("## 6. Best Model Diagnostics"))
cells.append(nbf.v4.new_code_cell("""best_model_name = model_bundle["model_name"]
best_model = model_bundle["model"]
target_transform = model_bundle.get("target_transform", "log1p")
print(f"Loaded Best Model: {best_model_name}")

test_df = pd.read_csv(PROCESSED_TEST_PATH)
feature_cols = model_bundle["feature_cols"]
base_cols = model_bundle.get("base_feature_cols", FINAL_FEATURE_COLS)
if 'severe_delay_risk' not in test_df.columns and model_bundle.get("severe_classifier") is not None:
    test_df['severe_delay_risk'] = model_bundle["severe_classifier"].predict_proba(test_df[base_cols])[:, 1]

X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL].values

raw_pred = best_model.predict(X_test)
if target_transform == "log1p":
    y_pred = np.maximum(0, np.expm1(raw_pred))
else:
    y_pred = np.maximum(0, raw_pred)

plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.15, color="darkcyan", s=10)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.title(f"Predicted vs Actual Delays ({best_model_name})")
plt.xlabel("Actual Delay (min)")
plt.ylabel("Predicted Delay (min)")

plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred, bins=40, kde=True, color="darkmagenta")
plt.title("Residual Error Distribution")
plt.xlabel("Residual (Actual - Predicted, min)")
plt.tight_layout()
plt.show()
"""))

# Cell 8: Live Sample Predictions
cells.append(nbf.v4.new_markdown_cell("## 7. Sample Predictions & ETA Calculation"))
cells.append(nbf.v4.new_code_cell("""sample_row = clean_df.iloc[100]
train_no = int(sample_row['train_no'])
journey_date = str(sample_row['journey_date'])[:10]
station_code = str(sample_row['station_code'])

pred_res = predict_delay_and_eta(train_no, journey_date, station_code, raw_df=raw_df)

print("Sample Single-Stop Dynamic ETA Prediction:")
print("-" * 45)
for k, v in pred_res.items():
    print(f"  {k:20s}: {v}")
"""))

nb.cells = cells

with open(Path("e:/Sample ETA/notebooks/01_exploration.ipynb"), "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Generated notebooks/01_exploration.ipynb successfully.")
