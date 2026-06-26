import pandas as pd
import numpy as np
import requests
import warnings
from hybrid_forecaster import HybridForecaster
warnings.filterwarnings("ignore")

PLANTS = {
    'Netherlands': {
        'csv': 'PV_measurements/pvoutput_data_netherlands1.csv',
        'lat': 51.5, 'lon': 5.5,
        'kwp': 498.42, 'tilt': 35, 'azimuth': 180,
        'power_col': 'Power (W)', 'power_scale': 1000.0,
        'date_format': 'pvoutput',
        'tz': 'Europe/Amsterdam'
    },
    'NC-1': {
        'csv': 'PV_measurements/14200_all_channels_20220101_20221231.csv',
        'lat': 35.813315, 'lon': -78.748991,
        'kwp': 14200.0, 'tilt': 20, 'azimuth': 180,
        'power_col': 'ac_power_meter_3105', 'power_scale': 1.0,
        'date_format': 'utc_measured_on',
        'tz': 'America/New_York'
    },
    'NC-2': {
        'csv': 'PV_measurements/14201_all_channels_20220101_20221231.csv',
        'lat': 35.812367, 'lon': -78.749346,
        'kwp': 1340.0, 'tilt': 20, 'azimuth': 180,
        'power_col': 'ac_power_meter_3106', 'power_scale': 1.0,
        'date_format': 'utc_measured_on',
        'tz': 'America/New_York'
    },
    'Germany': {
        'csv': 'PV_measurements/pvoutput_data_germany1.csv',
        'lat': 49.493956, 'lon': 11.075107,
        'kwp': 150.0, 'tilt': 25, 'azimuth': 180,
        'power_col': 'Power (W)', 'power_scale': 1000.0,
        'date_format': 'pvoutput',
        'tz': 'Europe/Berlin'
    }
}

def get_archived_weather_15min(lat, lon, start_date, end_date):
    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "is_day",
        "direct_radiation", "diffuse_radiation", "shortwave_radiation",
        "direct_normal_irradiance", "terrestrial_radiation"
    ]
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={','.join(variables)}&timezone=GMT"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch weather: {response.text}")
        
    data = response.json()
    hourly = data['hourly']
    
    df_weather = pd.DataFrame(hourly)
    df_weather['date_utc'] = pd.to_datetime(df_weather['time'])
    df_weather.set_index('date_utc', inplace=True)
    df_weather = df_weather.drop(columns=['time'])
    
    # Resample to 15-min and interpolate
    df_15min = df_weather.resample('15min').interpolate(method='linear')
    df_15min['is_day'] = df_15min['is_day'].round()
    
    # Required names for pvlib
    df_15min['ghi'] = df_15min['shortwave_radiation']
    df_15min['dni'] = df_15min['direct_normal_irradiance']
    df_15min['dhi'] = df_15min['diffuse_radiation']
    df_15min['temp_air'] = df_15min['temperature_2m']
    df_15min['wind_speed'] = df_15min['wind_speed_10m']
    
    df_15min['date_month'] = df_15min.index.month
    df_15min['date_day'] = df_15min.index.day
    df_15min['date_hour'] = df_15min.index.hour
    
    return df_15min

def calculate_metrics(df, pred_col, target_col='actual_power_kw', capacity=1.0):
    mask = df['is_day'] == 1
    df_day = df[mask]
    if len(df_day) == 0:
        return 0, 0, 0
    mae = np.mean(np.abs(df_day[target_col] - df_day[pred_col]))
    mae_pct_cap = (mae / capacity) * 100
    mean_actual = df_day[target_col].mean()
    mae_pct_act = (mae / mean_actual) * 100 if mean_actual > 0 else 0
    return mae, mae_pct_cap, mae_pct_act

def process_plant(name, config):
    print(f"\n{'='*50}\n Processing {name} ({config['kwp']} kWp)\n{'='*50}")
    
    df_raw = pd.read_csv(config['csv'])
    
    if config['date_format'] == 'pvoutput':
        df_raw['date_utc'] = pd.to_datetime(df_raw['Date'] + ' ' + df_raw['Time'])
        # PVOutput timestamps are typically local, but let's assume they are naive and map to UTC for simplification or localize them.
        # Actually PVOutput is local time. Let's localize and convert to UTC.
        df_raw['date_utc'] = df_raw['date_utc'].dt.tz_localize(config['tz'], nonexistent='shift_forward', ambiguous='NaT').dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        df_raw['date_utc'] = pd.to_datetime(df_raw[config['date_format']])
        
    df_raw = df_raw.dropna(subset=['date_utc'])
        
    # Scale power to kW
    df_raw['actual_power_kw'] = df_raw[config['power_col']] / config['power_scale']
    df_raw.loc[df_raw['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    
    # 15min resample
    df_raw.set_index('date_utc', inplace=True)
    df_15 = df_raw[['actual_power_kw']].resample('15min').mean().dropna()
    
    start_date = df_15.index.min().strftime('%Y-%m-%d')
    end_date = df_15.index.max().strftime('%Y-%m-%d')
    
    print(f"Fetching weather from {start_date} to {end_date}...")
    df_weather = get_archived_weather_15min(config['lat'], config['lon'], start_date, end_date)
    
    # Merge
    dataset = df_weather.copy()
    dataset = dataset.join(df_15, how='inner')
    dataset = dataset.dropna(subset=['actual_power_kw'])
    
    # Set timezone-aware index for pvlib (pvlib requires tz-aware datetime index)
    dataset.index = dataset.index.tz_localize('UTC')
    
    dataset['day_of_year'] = dataset.index.dayofyear
    train_mask = (dataset['day_of_year'] % 2 == 0)
    test_mask = ~train_mask
    
    df_train = dataset[train_mask].copy()
    df_test = dataset[test_mask].copy()
    
    print(f"Training on {len(df_train)} intervals, Testing on {len(df_test)} intervals...")
    
    forecaster = HybridForecaster(config['lat'], config['lon'], config['kwp'], config['tilt'], config['azimuth'])
    forecaster.fit(df_train, df_train['actual_power_kw'])
    
    phys_pred, ml_res, final_pred = forecaster.predict(df_test)
    df_test['pred_phys'] = phys_pred
    df_test['pred_hybrid'] = final_pred
    
    # Calculate metrics
    mae_phys, cap_phys, act_phys = calculate_metrics(df_test, 'pred_phys', capacity=config['kwp'])
    mae_hyb, cap_hyb, act_hyb = calculate_metrics(df_test, 'pred_hybrid', capacity=config['kwp'])
    
    print("\n--- RESULTS (Daylight Only) ---")
    print(f"[Physical Only] MAE: {mae_phys:.2f} kW | {cap_phys:.2f}% of Cap | {act_phys:.2f}% of Actual")
    print(f"[Hybrid Model]  MAE: {mae_hyb:.2f} kW | {cap_hyb:.2f}% of Cap | {act_hyb:.2f}% of Actual")
    return mae_phys, mae_hyb

if __name__ == "__main__":
    results = {}
    for name, config in PLANTS.items():
        try:
            p_mae, h_mae = process_plant(name, config)
            results[name] = {'Physical MAE': p_mae, 'Hybrid MAE': h_mae}
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    print("\n\n" + "="*50)
    print("FINAL BENCHMARK SUMMARY")
    print("="*50)
    for name, res in results.items():
        print(f"{name.ljust(15)} | Phys MAE: {res['Physical MAE']:.1f} kW  ->  Hybrid MAE: {res['Hybrid MAE']:.1f} kW")
