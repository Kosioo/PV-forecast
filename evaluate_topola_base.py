import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import re
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast"))
from quartz_solar_forecast.forecasts.v2 import TryolabsSolarPowerPredictor

# ================= Configuration =================
EXCEL_FILE = "Topola1.xlsx"
LATITUDE = 41.509238
LONGITUDE = 23.796700
CAPACITY_KWP = 5000.0
TILT = 38.0  
ORIENTATION = 180.0
TIMEZONE = 'Europe/Sofia'
START_DATE = "2025-06-01"
END_DATE = "2026-06-01"
OUTAGE_THRESHOLD_KW = CAPACITY_KWP * 0.05
# =================================================

def get_day_ahead_forecast(lat, lon, start_date, end_date):
    print(f"Fetching historical forecast from Open-Meteo for {start_date} to {end_date}...")
    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "is_day",
        "direct_radiation", "diffuse_radiation",
        "shortwave_radiation", "direct_normal_irradiance", "terrestrial_radiation"
    ]
    url = f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={','.join(variables)}&timezone=GMT"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch forecast: {response.text}")
        
    data = response.json()
    hourly = data['hourly']
    
    df_weather = pd.DataFrame(hourly)
    df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)
    df_weather = df_weather.set_index('time')
    df_weather = df_weather.apply(pd.to_numeric, errors='coerce')
    df_weather = df_weather.interpolate(method='linear')
    return df_weather

def calculate_metrics(df_test, pred_col, target_col='actual_power_kw', capacity=CAPACITY_KWP):
    mask = df_test['is_day'] == 1
    df_day = df_test[mask]
    
    mae = np.mean(np.abs(df_day[target_col] - df_day[pred_col]))
    mae_perc_cap = (mae / capacity) * 100.0
    
    actual_mean = df_day[target_col].mean()
    mae_perc_actual = (mae / actual_mean) * 100.0 if actual_mean > 0 else 0
    
    return mae, mae_perc_cap, mae_perc_actual

def main():
    print("=====================================================")
    print(f" Topola: BASE MODEL ONLY (No Adjustments)")
    print("=====================================================")
    
    # 1. Load Data
    print("Loading data...")
    try:
        df_raw = pd.read_excel(EXCEL_FILE, skiprows=1)
    except:
        df_raw = pd.read_csv(EXCEL_FILE, skiprows=1)
        
    df_raw.columns = ['time_str', 'capacity_mw']
    df_raw['time_str'] = df_raw['time_str'].astype(str).str.strip()
    df_raw['date_local'] = pd.to_datetime(df_raw['time_str'], errors='coerce', dayfirst=True)
    df_raw = df_raw.dropna(subset=['date_local'])
    
    def parse_mw(x):
        if isinstance(x, str):
            x = x.replace(',', '.')
            x = re.sub(r'[^\d\.]', '', x)
        try:
            return float(x) * 1000.0 # MW to kW
        except:
            return np.nan
            
    df_raw['actual_power_kw'] = df_raw['capacity_mw'].apply(parse_mw)
    df_raw = df_raw.dropna(subset=['actual_power_kw'])
    
    df_raw['date_local'] = df_raw['date_local'].dt.tz_localize(TIMEZONE, ambiguous='NaT', nonexistent='NaT')
    df_raw = df_raw.dropna(subset=['date_local'])
    df_raw['date_utc'] = df_raw['date_local'].dt.tz_convert('UTC')
    
    df_power = df_raw[['date_utc', 'actual_power_kw']].copy()
    df_power.set_index('date_utc', inplace=True)
    
    df_15min = df_power.resample('15min').mean()
    df_15min.loc[df_15min['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    df_15min = df_15min.dropna(subset=['actual_power_kw'])
    
    # 2. Fetch Weather Data
    df_weather_hourly = get_day_ahead_forecast(LATITUDE, LONGITUDE, START_DATE, END_DATE)
    df_weather_15min = df_weather_hourly.resample('15min').interpolate(method='linear')
    df_weather_15min['is_day'] = df_weather_15min['is_day'].round()
    
    # 3. Prepare Features
    predictor = TryolabsSolarPowerPredictor()
    predictor.load_model()
    
    def prepare_features(weather_df):
        df_copy = weather_df.copy().reset_index()
        df_copy = df_copy.rename(columns={'time': 'date_utc'})
        df_copy['date_local'] = df_copy['date_utc'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
        df_copy = df_copy.rename(columns={'date_local': 'date'})
        
        df_copy["orientation"] = ORIENTATION
        df_copy["tilt"] = TILT
        df_copy["kwp"] = CAPACITY_KWP
        df_copy["latitude_rounded"] = LATITUDE
        df_copy["longitude_rounded"] = LONGITUDE
        
        cleaned = predictor.clean(df_copy)
        features = cleaned.drop(columns=[predictor.DATE_COLUMN])
        expected_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                         'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                         'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                         'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                         'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
        
        return cleaned['date_utc'], features[expected_cols]
        
    dates_15min, X_15min = prepare_features(df_weather_15min)
    dates_1h, X_1h = prepare_features(df_weather_hourly)
    
    df_15min = df_15min.reset_index()
    dataset_15min = X_15min.copy()
    dataset_15min['date_utc'] = dates_15min
    dataset_15min = pd.merge(dataset_15min, df_15min, on='date_utc', how='inner')
    
    dataset_15min['date_local'] = dataset_15min['date_utc'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    dataset_15min['day_of_year'] = dataset_15min['date_local'].dt.dayofyear
    
    print("Evaluating entire year using only the Base Model...")
    
    feature_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                    'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                    'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                    'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
                    
    X_all_base = dataset_15min[feature_cols]
    
    base_model_path = os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast", "quartz_solar_forecast", "models", "model_10_202405.ubj")
    base_model = xgb.XGBRegressor()
    base_model.load_model(base_model_path)
    
    # 4. Predict
    preds_15_base = base_model.predict(X_all_base)
    
    pred_base_only_15 = preds_15_base * CAPACITY_KWP
    pred_base_only_15 = np.where(dataset_15min['is_day'] == 0, 0, pred_base_only_15)
    pred_base_only_15 = np.maximum(0, pred_base_only_15)
    
    test_15_results = dataset_15min[['date_utc', 'date_local', 'is_day', 'actual_power_kw']].copy()
    test_15_results['pred_base_only_15'] = pred_base_only_15
    
    test_1h_results = test_15_results.set_index('date_utc').resample('1h').mean().reset_index()
    test_1h_results = test_1h_results.dropna(subset=['actual_power_kw'])
    
    # 5. Evaluation
    print("\n================== BENCHMARK RESULTS (DAYLIGHT ONLY) ==================")
    print("--- 15-Minute Market ---")
    mae_15_b, mae_15_b_cap, mae_15_b_act = calculate_metrics(test_15_results, 'pred_base_only_15')
    print(f"[Base Model Only 15m] MAE: {mae_15_b:.1f} kW | MAE% Cap: {mae_15_b_cap:.1f}% | MAE% Act: {mae_15_b_act:.1f}%")
    
    print("\n--- Hourly Market ---")
    mae_1h_b, mae_1h_b_cap, mae_1h_b_act = calculate_metrics(test_1h_results, 'pred_base_only_15')
    print(f"[Base Model Only 1H]  MAE: {mae_1h_b:.1f} kW | MAE% Cap: {mae_1h_b_cap:.1f}% | MAE% Act: {mae_1h_b_act:.1f}%")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
