import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from validate_model import calculate_errors

def main():
    print("Loading datasets and model for evaluation...")
    
    # Load model
    model = xgb.XGBRegressor()
    model.load_model("tuned_model.ubj")
    
    # Load 15-min and Hourly datasets
    df_15 = pd.read_csv("dataset_15min.csv")
    df_15['date'] = pd.to_datetime(df_15['date'])
    
    df_1h = pd.read_csv("dataset_1h.csv")
    df_1h['date'] = pd.to_datetime(df_1h['date'])
    
    # Filter for testing set (odd days)
    df_15['day_of_year'] = df_15['date'].dt.dayofyear
    test_15 = df_15[df_15['day_of_year'] % 2 != 0].copy()
    
    df_1h['day_of_year'] = df_1h['date'].dt.dayofyear
    test_1h = df_1h[df_1h['day_of_year'] % 2 != 0].copy()
    
    print(f"Evaluating on {len(test_15['day_of_year'].unique())} unseen test days...")
    
    feature_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                    'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                    'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                    'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
    
    # --- SCENARIO A: 15-Minute Native -> Average to Hourly ---
    # Predict at 15-min
    preds_15 = model.predict(test_15[feature_cols]) * 1000.0 # Convert MW to kW
    
    # Set to 0 at night or negative
    preds_15 = np.where(test_15['is_day'] == 0, 0, preds_15)
    preds_15 = np.maximum(0, preds_15)
    
    test_15['pred_15'] = preds_15
    
    # Calculate Scenario A
    # Group by hourly to get the average
    test_15_hourly = test_15.set_index('date').resample('1h').mean()
    # We only want to keep the odd days because resampling creates NaNs for missing even days
    test_15_hourly = test_15_hourly.dropna(subset=['actual_power_kw'])
    
    error_A = calculate_errors(test_15_hourly['actual_power_kw'], test_15_hourly['pred_15'])
    
    
    # --- SCENARIO B: Hourly Native ---
    preds_1h = model.predict(test_1h[feature_cols]) * 1000.0
    preds_1h = np.where(test_1h['is_day'] == 0, 0, preds_1h)
    preds_1h = np.maximum(0, preds_1h)
    test_1h['pred_1h'] = preds_1h
    
    error_B = calculate_errors(test_1h['actual_power_kw'], test_1h['pred_1h'])
    
    
    # --- SCENARIO C: Hourly Extrapolated to 15-Minute ---
    # We need to take the hourly predictions and resample to 15-min.
    # We create a dataframe of just dates and predictions.
    df_hourly_preds = test_1h[['date', 'pred_1h']].set_index('date')
    
    # Because we skipped even days, resampling directly across the whole year will interpolate across missing days.
    # We shouldn't interpolate across 24 hour gaps.
    # Instead, let's just group by day, and resample/interpolate within each day.
    df_hourly_preds['day'] = df_hourly_preds.index.dayofyear
    df_extrapolated = df_hourly_preds[['pred_1h']].groupby(df_hourly_preds['day']).apply(lambda x: x.resample('15min').interpolate(method='linear')).reset_index(level=0, drop=True)
    
    # Merge back to test_15 to get actual 15-min actuals
    df_C = test_15[['date', 'actual_power_kw']].set_index('date')
    df_C = df_C.join(df_extrapolated, how='inner')
    
    error_C = calculate_errors(df_C['actual_power_kw'], df_C['pred_1h'])
    
    
    print("\n" + "="*50)
    print("      GRANULARITY EVALUATION RESULTS (~180 DAYS)      ")
    print("="*50)
    print("\n[Hourly Benchmark]")
    print(f"Scenario A (15-min Native -> Avg to 1H) | MAE: {error_A['MAE']:7.2f} kW | RMSE: {error_A['RMSE']:7.2f} kW")
    print(f"Scenario B (1-Hour Native Prediction)   | MAE: {error_B['MAE']:7.2f} kW | RMSE: {error_B['RMSE']:7.2f} kW")
    
    print("\n[15-Minute Benchmark]")
    # We calculate native 15-min error here just for comparison
    error_15_native = calculate_errors(test_15['actual_power_kw'], test_15['pred_15'])
    print(f"Native 15-Min Prediction              | MAE: {error_15_native['MAE']:7.2f} kW | RMSE: {error_15_native['RMSE']:7.2f} kW")
    print(f"Scenario C (1H Native -> Extrapolate) | MAE: {error_C['MAE']:7.2f} kW | RMSE: {error_C['RMSE']:7.2f} kW")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
