import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CAPACITY_KWP = 1000.0

def get_metrics(actual, forecast, capacity=CAPACITY_KWP):
    # Only calculate errors during sunny hours (actual or forecast > 0.5% capacity)
    daytime_mask = (actual > (capacity * 0.005)) | (forecast > (capacity * 0.005))
    if not daytime_mask.any():
        return {'MAE': 0.0, 'MAPE': 0.0, 'MAE_%_Cap': 0.0}
    
    a_day = actual[daytime_mask]
    f_day = forecast[daytime_mask]
    
    mae = np.mean(np.abs(a_day - f_day))
    
    # MAE% of interval (MAPE)
    mape_mask = a_day > (capacity * 0.05)
    if mape_mask.any():
        mape = np.mean(np.abs((a_day[mape_mask] - f_day[mape_mask]) / a_day[mape_mask])) * 100
    else:
        mape = 0.0
        
    mae_pct_cap = (mae / capacity) * 100
    
    return {
        'MAE': mae,
        'MAPE': mape,
        'MAE_%_Cap': mae_pct_cap
    }

def main():
    print("Loading datasets and model for 14200 comprehensive evaluation...")
    
    model = xgb.XGBRegressor()
    model.load_model("tuned_model_14200.ubj")
    
    df_15 = pd.read_csv("dataset_14200_15min.csv")
    df_15['date'] = pd.to_datetime(df_15['date'])
    
    df_1h = pd.read_csv("dataset_14200_1h.csv")
    df_1h['date'] = pd.to_datetime(df_1h['date'])
    
    df_15['day_of_year'] = df_15['date'].dt.dayofyear
    test_15 = df_15[df_15['day_of_year'] % 2 != 0].copy()
    
    df_1h['day_of_year'] = df_1h['date'].dt.dayofyear
    test_1h = df_1h[df_1h['day_of_year'] % 2 != 0].copy()
    
    # Filter out outage days
    day_max = test_15.groupby('day_of_year')['actual_power_kw'].max()
    outage_days = day_max[day_max < 50].index
    print(f"Found {len(outage_days)} suspicious outage days. Removing them from evaluation.")
    
    test_15 = test_15[~test_15['day_of_year'].isin(outage_days)].copy()
    test_1h = test_1h[~test_1h['day_of_year'].isin(outage_days)].copy()
    
    # Base model
    base_model_path = os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast", "quartz_solar_forecast", "models", "model_10_202405.ubj")
    base_model = xgb.XGBRegressor()
    base_model.load_model(base_model_path)
    
    # Residual model
    residual_model = xgb.XGBRegressor()
    residual_model.load_model('residual_model_14200.ubj')
    
    feature_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                    'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                    'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                    'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
                    
    # --- SCENARIO 1: 15min Native ---
    preds_15_base = base_model.predict(test_15[feature_cols])
    preds_15_res = residual_model.predict(test_15[feature_cols])
    preds_15 = (preds_15_base + preds_15_res) * 1000.0
    
    # Set to 0 at night or negative
    preds_15 = np.where(test_15['is_day'] == 0, 0, preds_15)
    preds_15 = np.maximum(0, preds_15)
    
    test_15['pred_native_15'] = preds_15
    
    # --- Scenario 2: 15-Min Native -> Avg to Hourly ---
    test_15_hourly = test_15.set_index('date').resample('1h').mean()
    test_15_hourly = test_15_hourly.dropna(subset=['actual_power_kw']).reset_index()
    
    # --- SCENARIO 3: Hourly Native ---
    preds_1h_base = base_model.predict(test_1h[feature_cols])
    preds_1h_res = residual_model.predict(test_1h[feature_cols])
    preds_1h = (preds_1h_base + preds_1h_res) * 1000.0
    
    preds_1h = np.where(test_1h['is_day'] == 0, 0, preds_1h)
    preds_1h = np.maximum(0, preds_1h)
    test_1h['pred_native_1h'] = preds_1h
    
    # --- Scenario 4: Hourly Extrapolated to 15-Minute ---
    df_hourly_preds = test_1h[['date', 'pred_native_1h']].set_index('date')
    df_hourly_preds['day'] = df_hourly_preds.index.dayofyear
    df_extrapolated = df_hourly_preds[['pred_native_1h']].groupby(df_hourly_preds['day']).apply(lambda x: x.resample('15min').interpolate(method='linear'))
    df_extrapolated = df_extrapolated.reset_index(level=0, drop=True)
    
    df_C = test_15[['date', 'actual_power_kw', 'day_of_year']].set_index('date')
    df_C = df_C.join(df_extrapolated, how='inner').reset_index()
    df_C = df_C.rename(columns={'pred_native_1h': 'pred_extrapolated_15'})
    
    # --- Calculate Overall Metrics ---
    metrics_15_native = get_metrics(test_15['actual_power_kw'], test_15['pred_native_15'], capacity=CAPACITY_KWP)
    metrics_A = get_metrics(test_15_hourly['actual_power_kw'], test_15_hourly['pred_native_15'], capacity=CAPACITY_KWP)
    metrics_B = get_metrics(test_1h['actual_power_kw'], test_1h['pred_native_1h'], capacity=CAPACITY_KWP)
    metrics_C = get_metrics(df_C['actual_power_kw'], df_C['pred_extrapolated_15'], capacity=CAPACITY_KWP)
    
    # --- Generate Plots for 5 Random Days ---
    # Find 5 odd days that have no missing data and high solar potential
    valid_days = test_15.groupby('day_of_year')['actual_power_kw'].max()
    valid_days = valid_days[valid_days > 300].index.tolist()
    import random
    random.seed(42)
    sample_days = random.sample(valid_days, 5)
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.5)
    
    for i, day in enumerate(sample_days):
        day_15 = test_15[test_15['day_of_year'] == day]
        day_1h = test_1h[test_1h['day_of_year'] == day]
        day_C = df_C[df_C['day_of_year'] == day]
        
        ax = axes[i]
        
        # Plot Actuals (15min)
        ax.plot(day_15['date'], day_15['actual_power_kw'], label='Actual (15m)', color='black', linewidth=2)
        
        # Plot 1: Native 15m
        ax.plot(day_15['date'], day_15['pred_native_15'], label='Native 15m (Scen C)', color='blue', linestyle='--')
        
        # Plot 2: Extrapolated 15m
        ax.plot(day_C['date'], day_C['pred_extrapolated_15'], label='Extrapolated 15m (Scen D)', color='green', linestyle=':')
        
        # Plot 3: Native Hourly
        # Step plot for hourly data to show it remains constant over the hour
        ax.step(day_1h['date'], day_1h['pred_native_1h'], label='Native 1h (Scen B)', color='red', where='post')
        
        ax.set_title(f"Forecast comparison for Day {day}")
        ax.set_ylabel("Power (kW)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        if i == 0:
            ax.legend(loc='upper right')
            
    plt.tight_layout()
    plt.savefig('validation_plots_14200.png')
    print("Saved 5-day validation plots to validation_plots_14200.png")
    
    print("\n" + "="*60)
    print("      SAS_SF1 (1MW) DAY-AHEAD FORECAST EVALUATION      ")
    print("="*60)
    
    print("\n[ 15-Minute Market Approaches ]")
    print(f"1. Native 15-min Predict:    MAE = {metrics_15_native['MAE']:6.1f} kW | MAE% = {metrics_15_native['MAE_%_Cap']:4.1f}% | MAE% of interval = {metrics_15_native['MAPE']:5.1f}%")
    print(f"2. Scenario C (Extrapolate): MAE = {metrics_C['MAE']:6.1f} kW | MAE% = {metrics_C['MAE_%_Cap']:4.1f}% | MAE% of interval = {metrics_C['MAPE']:5.1f}%")
    
    print("\n[ Hourly Market Approaches ]")
    print(f"3. Scenario B (Native 1H):   MAE = {metrics_B['MAE']:6.1f} kW | MAE% = {metrics_B['MAE_%_Cap']:4.1f}% | MAE% of interval = {metrics_B['MAPE']:5.1f}%")
    print(f"4. Scenario A (15m -> Avg):  MAE = {metrics_A['MAE']:6.1f} kW | MAE% = {metrics_A['MAE_%_Cap']:4.1f}% | MAE% of interval = {metrics_A['MAPE']:5.1f}%")
    
    print("\nReport generation complete!")

if __name__ == "__main__":
    main()
