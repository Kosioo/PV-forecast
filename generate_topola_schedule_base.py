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
LATITUDE = 41.509238
LONGITUDE = 23.796700
CAPACITY_KWP = 5000.0
TILT = 38.0  
ORIENTATION = 180.0
TIMEZONE = 'Europe/Sofia'
FORECAST_DATE = "2026-06-26"
# =================================================

def get_forecast(lat, lon, start_date, end_date):
    print(f"Fetching Open-Meteo data for {start_date} to {end_date}...")
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

def main():
    print("=====================================================")
    print(f" Topola Schedule: BASE MODEL ONLY ({FORECAST_DATE})")
    print("=====================================================")
    
    # 1. Fetch Forecast for Today
    print(f"\nFetching live forecast for {FORECAST_DATE}...")
    df_weather_hourly_today = get_forecast(LATITUDE, LONGITUDE, FORECAST_DATE, FORECAST_DATE)
    df_weather_15min_today = df_weather_hourly_today.resample('15min').interpolate(method='linear')
    df_weather_15min_today['is_day'] = df_weather_15min_today['is_day'].round()
    
    # 2. Prepare Features Model
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
        
    dates_today, X_today_base = prepare_features(df_weather_15min_today)
    
    # 3. Load Model
    feature_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                    'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                    'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                    'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
                    
    base_model_path = os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast", "quartz_solar_forecast", "models", "model_10_202405.ubj")
    base_model = xgb.XGBRegressor()
    base_model.load_model(base_model_path)
    
    # 4. Predict Schedule
    print(f"Generating 15-minute schedule for {FORECAST_DATE}...")
    preds_base = base_model.predict(X_today_base[feature_cols])
    
    pred_base_only = preds_base * CAPACITY_KWP
    pred_base_only = np.where(df_weather_15min_today['is_day'] == 0, 0, pred_base_only)
    pred_base_only = np.maximum(0, pred_base_only)
    
    schedule_df = pd.DataFrame({
        'Date_UTC': dates_today,
        'Date_Local': dates_today.dt.tz_convert(TIMEZONE).dt.tz_localize(None),
        'Base_Model_Power_kW': np.round(pred_base_only, 2)
    })
    
    # Save CSV
    csv_filename = f"topola_schedule_base_{FORECAST_DATE.replace('-', '')}.csv"
    schedule_df.to_csv(csv_filename, index=False)
    print(f"\nSUCCESS! Saved schedule to {csv_filename}")
    
    # 5. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(schedule_df['Date_Local'], schedule_df['Base_Model_Power_kW'], color='blue', linewidth=2, label='Base Model Only')
    plt.fill_between(schedule_df['Date_Local'], schedule_df['Base_Model_Power_kW'], color='blue', alpha=0.3)
    plt.title(f"Topola 5MW Generation Schedule (Base Model Only) - {FORECAST_DATE}")
    plt.xlabel(f"Local Time ({TIMEZONE})")
    plt.ylabel("Power (kW)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_filename = f"topola_schedule_base_{FORECAST_DATE.replace('-', '')}.png"
    plt.savefig(plot_filename)
    print(f"Saved schedule plot to {plot_filename}")

if __name__ == "__main__":
    main()
