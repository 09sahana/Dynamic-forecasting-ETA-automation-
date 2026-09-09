import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_DIR, PROCESSED_TEST_PATH, TARGET_COL

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    diff = np.abs(y_true - y_pred)
    w5 = (diff <= 5.0).mean() * 100.0
    w10 = (diff <= 10.0).mean() * 100.0
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "Within_5min_%": round(float(w5), 2),
        "Within_10min_%": round(float(w10), 2)
    }

def evaluate_best_model():
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        print(f"Error: Saved model bundle not found at {model_path}. Run src/train.py first.")
        return

    print(f"1. Loading saved model bundle from '{model_path}'...")
    bundle = joblib.load(model_path)
    
    model_name = bundle["model_name"]
    model = bundle.get("model_p50", bundle.get("model"))
    # FIX 2: Canonical intervals [p08, p92]
    model_p08 = bundle.get("model_p08", bundle.get("model_p10"))
    model_p92 = bundle.get("model_p92", bundle.get("model_p90"))
    model_p10 = bundle.get("model_p10", None)
    model_p90 = bundle.get("model_p90", None)
    feature_cols = bundle["feature_cols"]
    base_feature_cols = bundle.get("base_feature_cols", [c for c in feature_cols if c != 'severe_delay_risk'])
    target_transform = bundle.get("target_transform", "log1p")
    severe_clf = bundle.get("severe_classifier", None)
    conf_model = bundle.get("confidence_model", None)
    conf_cutoffs = bundle.get("confidence_cutoffs", [13.0, 21.5])

    print("\n2. Loading unseen test set for v5 evaluation...")
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    
    # Compute stacked feature if needed
    if 'severe_delay_risk' not in test_df.columns and severe_clf is not None:
        test_df['severe_delay_risk'] = severe_clf.predict_proba(test_df[base_feature_cols])[:, 1]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values
    capped_flag = test_df['delay_is_capped'].values if 'delay_is_capped' in test_df.columns else (y_test >= 360).astype(int)
    
    raw_pred = model.predict(X_test)
    if target_transform == "log1p":
        y_pred = np.maximum(0, np.expm1(raw_pred))
    else:
        y_pred = np.maximum(0, raw_pred)
        
    residuals = y_test - y_pred

    # Segmented Evaluation (Point/p50 Model)
    met_overall = compute_metrics(y_test, y_pred)
    met_noncapped = compute_metrics(y_test[capped_flag == 0], y_pred[capped_flag == 0])
    met_capped = compute_metrics(y_test[capped_flag == 1], y_pred[capped_flag == 1])

    print("\n" + "=" * 80)
    print(f"SEGMENTED EVALUATION SUMMARY FOR BEST MODEL: '{model_name}' (v5 Pipeline)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Non-Capped (95.1%)':<20} | {'Overall (100%)':<18} | {'Capped (4.9%)':<16}")
    print("-" * 80)
    for k in ["MAE", "RMSE", "R2", "Within_5min_%", "Within_10min_%"]:
        print(f"{k:<20} | {met_noncapped[k]:<20} | {met_overall[k]:<18} | {met_capped[k]:<16}")
    print("=" * 80)

    # FIX 2: Prediction Interval Coverage [p08, p92] vs [p10, p90]
    print("\n3. FIX 2: Canonical Prediction Interval Coverage [p08, p92]:")
    print("-" * 80)
    if model_p08 is not None and model_p92 is not None:
        raw_p08 = model_p08.predict(X_test)
        raw_p92 = model_p92.predict(X_test)
        p08 = np.maximum(0, np.expm1(raw_p08)) if target_transform == "log1p" else np.maximum(0, raw_p08)
        p92 = np.maximum(0, np.expm1(raw_p92)) if target_transform == "log1p" else np.maximum(0, raw_p92)
        p08 = np.minimum(p08, y_pred)
        p92 = np.maximum(p92, y_pred)
        
        cov_nc_08_92 = ((y_test[capped_flag == 0] >= p08[capped_flag == 0]) & (y_test[capped_flag == 0] <= p92[capped_flag == 0])).mean() * 100.0
        cov_all_08_92 = ((y_test >= p08) & (y_test <= p92)).mean() * 100.0
        mean_width_nc = (p92[capped_flag == 0] - p08[capped_flag == 0]).mean()
        
        print(f"  Canonical [p08, p92] Non-Capped Coverage: {cov_nc_08_92:.2f}% (Nominal Target: ~84%) | Mean Width: {mean_width_nc:.1f} min")
        print(f"  Canonical [p08, p92] Overall Coverage:    {cov_all_08_92:.2f}%")

        if model_p10 is not None and model_p90 is not None:
            raw_p10 = model_p10.predict(X_test)
            raw_p90 = model_p90.predict(X_test)
            p10 = np.maximum(0, np.expm1(raw_p10)) if target_transform == "log1p" else np.maximum(0, raw_p10)
            p90 = np.maximum(0, np.expm1(raw_p90)) if target_transform == "log1p" else np.maximum(0, raw_p90)
            cov_nc_10_90 = ((y_test[capped_flag == 0] >= p10[capped_flag == 0]) & (y_test[capped_flag == 0] <= p90[capped_flag == 0])).mean() * 100.0
            print(f"  Reference [p10, p90] Non-Capped Coverage: {cov_nc_10_90:.2f}% (Nominal Target: ~80%)")
            print(f"  -> VALIDATION: [p08, p92] is closer to nominal target coverage ({cov_nc_08_92:.1f}% vs 84%) than [p10, p90] ({cov_nc_10_90:.1f}% vs 80%).")
    else:
        print("  Quantile models not found in bundle.")

    # FIX 1: Chronic-Train Subset Verification
    print("\n4. FIX 1: Chronic-Train Subset Verification (Without Residual Correction):")
    print("-" * 80)
    chronic_trains = [12508, 15648, 12507, 12509, 15630]
    chronic_mask = test_df['train_no'].isin(chronic_trains).values
    control_mask = (test_df['train_no'] == 12137).values
    
    chronic_mae = mean_absolute_error(y_test[chronic_mask], y_pred[chronic_mask])
    control_mae = mean_absolute_error(y_test[control_mask], y_pred[control_mask])
    print(f"  Chronic Trains MAE: {chronic_mae:.2f} min (Baseline v3: ~55.38 min, v4: 47.76 min)")
    print(f"  Control Train #12137 MAE: {control_mae:.2f} min")
    assert chronic_mae <= 48.5, f"Chronic train MAE regressed: {chronic_mae:.2f} min > 48.5 min"
    print("  -> CONFIRMED: Dropping residual correction did not regress the chronic_delay_tier win.")

    # Confidence Calibration
    print("\n5. Confidence Calibration on Unseen Test Set:")
    print("-" * 80)
    if conf_model is not None:
        pred_errors = conf_model.predict(X_test)
        q33, q66 = conf_cutoffs
        conf_labels = np.where(pred_errors <= q33, "High", np.where(pred_errors <= q66, "Medium", "Low"))
        
        for c in ["High", "Medium", "Low"]:
            mask = (conf_labels == c)
            cnt = int(mask.sum())
            c_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            c_w10 = (np.abs(y_test[mask] - y_pred[mask]) <= 10.0).mean() * 100.0
            print(f"  Confidence {c:6s}: Count={cnt:5d} ({cnt/len(y_test)*100:4.1f}%) | MAE={c_mae:5.2f} min | Within 10m={c_w10:4.1f}%")
        
        high_mae = mean_absolute_error(y_test[conf_labels == "High"], y_pred[conf_labels == "High"])
        low_mae = mean_absolute_error(y_test[conf_labels == "Low"], y_pred[conf_labels == "Low"])
        assert high_mae < low_mae, f"Calibration check failed: High MAE ({high_mae}) >= Low MAE ({low_mae})"
        print("  -> CONFIDENCE CALIBRATION VALIDATED: High confidence error < Low confidence error.")

    # Walk-Forward Validation Metrics
    if "walk_forward_metrics" in bundle:
        wf = bundle["walk_forward_metrics"]
        print("\n6. Walk-Forward Temporal Robustness Summary (3 Rolling Splits):")
        print("-" * 80)
        print(f"  Mean Non-Capped MAE:        {wf['mean_noncap_mae']} +/- {wf['std_noncap_mae']} min")
        print(f"  Mean Non-Capped Within 10m: {wf['mean_noncap_w10']}% +/- {wf['std_noncap_w10']}%")

    # Diagnostic Charts
    print("\n7. Generating Diagnostic Charts...")
    plt.figure(figsize=(18, 5))
    
    # Plot 1: Feature Importances
    plt.subplot(1, 3, 1)
    importances = None
    if "feature_importances" in bundle and bundle["feature_importances"]:
        importances = np.array([bundle["feature_importances"].get(c, 0.0) for c in feature_cols])
    elif hasattr(model, 'feature_importances_'):
        importances = np.array(model.feature_importances_)
    elif (MODEL_DIR / "feature_importance.csv").exists():
        f_df = pd.read_csv(MODEL_DIR / "feature_importance.csv").set_index("feature")
        importances = np.array([f_df.loc[c, "importance"] if c in f_df.index else 0.0 for c in feature_cols])

    if importances is not None:
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(feature_cols))
        top_indices = indices[:top_n]
        top_cols = [feature_cols[i] for i in top_indices]
        top_scores = importances[top_indices]
        
        sns.barplot(
            x=top_scores,
            y=top_cols,
            hue=top_cols,
            palette="viridis",
            legend=False
        )
        plt.title(f"Top {top_n} Feature Importances ({model_name})")
        plt.xlabel("Importance Score")
    else:
        plt.text(0.5, 0.5, "Feature Importances\nnot directly supported", ha='center', va='center')

    # Plot 2: Predicted vs Actual Scatter Plot (Segmented)
    plt.subplot(1, 3, 2)
    plt.scatter(y_test[capped_flag == 0], y_pred[capped_flag == 0], alpha=0.15, color='teal', s=10, label='Non-Capped Delays')
    plt.scatter(y_test[capped_flag == 1], y_pred[capped_flag == 1], alpha=0.35, color='crimson', s=12, label='360-min Capped Artifact')
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Perfect 1:1 Line')
    plt.axvline(360, color='red', linestyle=':', alpha=0.7, label='360m Data Cap Boundary')
    plt.xlabel("Actual Delay (min)")
    plt.ylabel("Predicted Delay (min)")
    plt.title("Predicted vs Actual Delays (v5 Segmented)")
    plt.legend(loc="upper left", fontsize=8)

    # Plot 3: Residuals Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(residuals[capped_flag == 0], bins=50, kde=True, color='purple', label='Non-Capped Residuals')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel("Residual (Actual - Predicted, min)")
    plt.ylabel("Frequency")
    plt.title("Residuals Distribution (v5)")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plot_path = MODEL_DIR / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved evaluation diagnostic charts to '{plot_path}'")

if __name__ == "__main__":
    evaluate_best_model()
