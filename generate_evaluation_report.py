import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def safe_mape(actual, forecast, capacity=1340.0):
    # If actual is 0, MAPE blows up. We use a threshold, or capacity normalized MAPE.
    # The user asked for "MAE in %". Usually, for PV, this means MAE / Capacity.
    # We will compute both: standard MAPE (on non-zero) and Capacity-Normalized MAE.
    mask = actual > (capacity * 0.05)
    if not mask.any():
        return 0.0, 0.0
    a = actual[mask]
    f = forecast[mask]
    mape = np.mean(np.abs((a - f) / a)) * 100
    mae_pct = (np.mean(np.abs(a - f)) / capacity) * 100
    return mape, mae_pct

def get_metrics(actual, forecast, capacity=1340.0):
    # Only calculate errors during the day (where either actual or forecast is > 0.5% of capacity)
    # This removes the 0 sun periods from artificially diluting the MAE
    daytime_mask = (actual > (capacity * 0.005)) | (forecast > (capacity * 0.005))
    if not daytime_mask.any():
        return {'MAE': 0.0, 'MAPE': 0.0, 'MAE_%_Cap': 0.0, 'Max_Err': 0.0, 'Min_Err': 0.0}
    
    a_day = actual[daytime_mask]
    f_day = forecast[daytime_mask]
    
    mae = np.mean(np.abs(a_day - f_day))
    
    # For MAPE we use a slightly higher threshold to avoid dividing by tiny numbers
    mape_mask = a_day > (capacity * 0.05)
    if mape_mask.any():
        mape = np.mean(np.abs((a_day[mape_mask] - f_day[mape_mask]) / a_day[mape_mask])) * 100
    else:
        mape = 0.0
        
    mae_pct_cap = (mae / capacity) * 100
    
    errors = np.abs(a_day - f_day)
    max_err = errors.max()
    min_err = errors.min()
    
    return {
        'MAE': mae,
        'MAPE': mape,
        'MAE_%_Cap': mae_pct_cap,
        'Max_Err': max_err,
        'Min_Err': min_err
    }

def main():
    print("Loading datasets and model for comprehensive evaluation...")
    
    model = xgb.XGBRegressor()
    model.load_model("tuned_model.ubj")
    
    df_15 = pd.read_csv("dataset_15min.csv")
    df_15['date'] = pd.to_datetime(df_15['date'])
    
    df_1h = pd.read_csv("dataset_1h.csv")
    df_1h['date'] = pd.to_datetime(df_1h['date'])
    
    # Filter for testing set (odd days)
    df_15['day_of_year'] = df_15['date'].dt.dayofyear
    test_15 = df_15[df_15['day_of_year'] % 2 != 0].copy()
    
    df_1h['day_of_year'] = df_1h['date'].dt.dayofyear
    test_1h = df_1h[df_1h['day_of_year'] % 2 != 0].copy()
    
    # Filter out suspicious outage days (e.g. max generation < 50 kW for the entire day)
    day_max = test_15.groupby('day_of_year')['actual_power_kw'].max()
    outage_days = day_max[day_max < 50].index
    print(f"Found {len(outage_days)} suspicious outage days. Removing them from evaluation.")
    
    test_15 = test_15[~test_15['day_of_year'].isin(outage_days)].copy()
    test_1h = test_1h[~test_1h['day_of_year'].isin(outage_days)].copy()
    
    feature_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                    'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                    'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                    'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
    
    # --- Native 15-Minute ---
    preds_15 = model.predict(test_15[feature_cols]) * 1000.0 
    preds_15 = np.where(test_15['is_day'] == 0, 0, preds_15)
    preds_15 = np.maximum(0, preds_15)
    test_15['pred_native_15'] = preds_15
    
    # --- Scenario A: 15-Min Native -> Avg to Hourly ---
    test_15_hourly = test_15.set_index('date').resample('1h').mean()
    test_15_hourly = test_15_hourly.dropna(subset=['actual_power_kw']).reset_index()
    
    # --- Scenario B: Hourly Native ---
    preds_1h = model.predict(test_1h[feature_cols]) * 1000.0
    preds_1h = np.where(test_1h['is_day'] == 0, 0, preds_1h)
    preds_1h = np.maximum(0, preds_1h)
    test_1h['pred_native_1h'] = preds_1h
    
    # --- Scenario C: Hourly Extrapolated to 15-Minute ---
    df_hourly_preds = test_1h[['date', 'pred_native_1h']].set_index('date')
    df_hourly_preds['day'] = df_hourly_preds.index.dayofyear
    df_extrapolated = df_hourly_preds[['pred_native_1h']].groupby(df_hourly_preds['day']).apply(lambda x: x.resample('15min').interpolate(method='linear')).reset_index(level=0, drop=True)
    
    df_C = test_15[['date', 'actual_power_kw', 'day_of_year']].set_index('date')
    df_C = df_C.join(df_extrapolated, how='inner').reset_index()
    df_C = df_C.rename(columns={'pred_native_1h': 'pred_extrapolated_15'})
    
    # --- Calculate Overall Metrics (~180 days) ---
    print("\n" + "="*60)
    print("      OVERALL AVERAGE ERROR ON ENTIRE TEST PERIOD      ")
    print("="*60)
    
    metrics_15_native = get_metrics(test_15['actual_power_kw'], test_15['pred_native_15'])
    metrics_A = get_metrics(test_15_hourly['actual_power_kw'], test_15_hourly['pred_native_15'])
    metrics_B = get_metrics(test_1h['actual_power_kw'], test_1h['pred_native_1h'])
    metrics_C = get_metrics(df_C['actual_power_kw'], df_C['pred_extrapolated_15'])
    
    print("\n[ 15-Minute Market Approaches ]")
    print(f"1. Native 15-min Predict:    MAE = {metrics_15_native['MAE']:6.1f} kW | MAE% (Cap) = {metrics_15_native['MAE_%_Cap']:4.1f}% | MAPE = {metrics_15_native['MAPE']:5.1f}% | MaxErr = {metrics_15_native['Max_Err']:6.1f} kW")
    print(f"2. Scenario C (Extrapolate): MAE = {metrics_C['MAE']:6.1f} kW | MAE% (Cap) = {metrics_C['MAE_%_Cap']:4.1f}% | MAPE = {metrics_C['MAPE']:5.1f}% | MaxErr = {metrics_C['Max_Err']:6.1f} kW")
    
    print("\n[ Hourly Market Approaches ]")
    print(f"3. Scenario B (Native 1H):   MAE = {metrics_B['MAE']:6.1f} kW | MAE% (Cap) = {metrics_B['MAE_%_Cap']:4.1f}% | MAPE = {metrics_B['MAPE']:5.1f}% | MaxErr = {metrics_B['Max_Err']:6.1f} kW")
    print(f"4. Scenario A (15m -> Avg):  MAE = {metrics_A['MAE']:6.1f} kW | MAE% (Cap) = {metrics_A['MAE_%_Cap']:4.1f}% | MAPE = {metrics_A['MAPE']:5.1f}% | MaxErr = {metrics_A['Max_Err']:6.1f} kW")
    
    # --- Generate Graphics for 5 Different Days ---
    # Pick 5 distinct days from the test set (e.g. spring, summer, fall, winter, varying cloud cover)
    test_days = test_15['day_of_year'].unique()
    # Let's pick 5 days evenly spaced from the valid days
    np.random.seed(42)
    selected_days = [test_days[10], test_days[len(test_days)//4], test_days[len(test_days)//2], test_days[(len(test_days)*3)//4], test_days[-10]]
    
    # We will create 4 images.
    # Image 1: Native 15-Min Prediction (across the 5 days)
    # Image 2: Scenario C (1H Extrapolated) (across the 5 days)
    # Image 3: Scenario B (Native 1H) (across the 5 days)
    # Image 4: Scenario A (15m Avg to 1H) (across the 5 days)
    
    approaches = [
        {'name': 'Native 15-Min Prediction', 'df': test_15, 'pred_col': 'pred_native_15', 'filename': 'plot_native_15.png'},
        {'name': 'Scenario C: 1H Extrapolated to 15m', 'df': df_C, 'pred_col': 'pred_extrapolated_15', 'filename': 'plot_scenario_c.png'},
        {'name': 'Scenario B: Native 1H Prediction', 'df': test_1h, 'pred_col': 'pred_native_1h', 'filename': 'plot_scenario_b.png'},
        {'name': 'Scenario A: 15m Averaged to 1H', 'df': test_15_hourly, 'pred_col': 'pred_native_15', 'filename': 'plot_scenario_a.png'},
    ]
    
    for approach in approaches:
        fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharey=True)
        fig.suptitle(f"{approach['name']} - 5 Test Days Performance", fontsize=18, y=0.98)
        df_app = approach['df']
        
        for i, day in enumerate(selected_days):
            ax = axes[i]
            day_data = df_app[df_app['day_of_year'] == day]
            if len(day_data) == 0:
                continue
                
            ax.plot(day_data['date'], day_data['actual_power_kw'], label='Actual (kW)', color='red', linewidth=2, alpha=0.8)
            ax.plot(day_data['date'], day_data[approach['pred_col']], label='Predicted (kW)', color='blue', linewidth=2, alpha=0.8)
            
            # Formatting
            ax.set_title(f"Day of Year: {day} (Date: {day_data['date'].iloc[0].strftime('%Y-%m-%d')})", fontsize=12)
            ax.set_ylabel('Power (kW)')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # Calculate metrics for this specific day
            day_metrics = get_metrics(day_data['actual_power_kw'], day_data[approach['pred_col']])
            metric_text = (f"Daily MAE: {day_metrics['MAE']:.1f} kW  |  MAE%: {day_metrics['MAE_%_Cap']:.1f}%\n"
                           f"Max Error: {day_metrics['Max_Err']:.1f} kW  |  Min Error: {day_metrics['Min_Err']:.1f} kW")
            ax.text(0.02, 0.85, metric_text, transform=ax.transAxes, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
            
            if i == 0:
                ax.legend(loc='upper right')
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(approach['filename'])
        plt.close()
        print(f"Saved {approach['filename']}")

    print("\nReport generation complete!")

if __name__ == "__main__":
    main()
