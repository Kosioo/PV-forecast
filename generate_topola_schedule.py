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
HISTORICAL_START = "2025-06-01"
HISTORICAL_END = "2026-06-01" # Topola dataset ends around here
FORECAST_DATE = "2026-06-26"
OUTAGE_THRESHOLD_KW = CAPACITY_KWP * 0.05  # 5% of capacity (~250 kW)
# =================================================

def get_forecast(lat, lon, start_date, end_date):
    print(f"Fetching Open-Meteo data for {start_date} to {end_date}...")
    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "is_day",
        "direct_radiation", "diffuse_radiation",
        "shortwave_radiation", "direct_normal_irradiance", "terrestrial_radiation", "snow_depth"
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
    print(f" Topola Live Production Schedule ({FORECAST_DATE})")
    print("=====================================================")
    
    # 1. Load Historical Actuals
    print("Loading historical actuals...")
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
    
    df_15min_hist = df_power.resample('15min').mean()
    df_15min_hist.loc[df_15min_hist['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    df_15min_hist = df_15min_hist.dropna(subset=['actual_power_kw'])
    
    # 2. Fetch Historical Weather
    df_weather_hourly_hist = get_forecast(LATITUDE, LONGITUDE, HISTORICAL_START, HISTORICAL_END)
    df_weather_15min_hist = df_weather_hourly_hist.resample('15min').interpolate(method='linear')
    df_weather_15min_hist['is_day'] = df_weather_15min_hist['is_day'].round()
    
    # 3. Prepare Features Model
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
        
    dates_hist, X_hist = prepare_features(df_weather_15min_hist)
    if 'snow_depth' in df_weather_15min_hist.columns:
        X_hist['snow_depth'] = df_weather_15min_hist['snow_depth'].values
    
    df_15min_hist = df_15min_hist.reset_index()
    dataset_hist = X_hist.copy()
    dataset_hist['date_utc'] = dates_hist
    dataset_hist = pd.merge(dataset_hist, df_15min_hist, on='date_utc', how='inner')
    
    # Smart Outage Filtering (Filter Maintenance, keep Snow)
    dataset_hist['date_local'] = dataset_hist['date_utc'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    dataset_hist['day_of_year'] = dataset_hist['date_local'].dt.dayofyear
    
    day_max_power = dataset_hist.groupby('day_of_year')['actual_power_kw'].max()
    day_max_snow = dataset_hist.groupby('day_of_year')['snow_depth'].max() if 'snow_depth' in dataset_hist.columns else pd.Series(0, index=day_max_power.index)
        
    potential_outages = day_max_power[day_max_power < OUTAGE_THRESHOLD_KW].index
    maintenance_days = [day for day in potential_outages if day_max_snow.get(day, 0) == 0]
    dataset_hist = dataset_hist[~dataset_hist['day_of_year'].isin(maintenance_days)].copy()
    
    # 4. Train Models on 100% of Data
    print(f"Training Base + Residual models on {len(dataset_hist)} historical 15-minute intervals...")
    feature_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                    'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                    'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                    'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
    
    residual_cols = feature_cols + (['snow_depth'] if 'snow_depth' in dataset_hist.columns else [])
    
    X_train_base = dataset_hist[feature_cols]
    X_train_res = dataset_hist[residual_cols]
    y_train = dataset_hist['actual_power_kw'] 
    
    base_model_path = os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast", "quartz_solar_forecast", "models", "model_10_202405.ubj")
    base_model = xgb.XGBRegressor()
    base_model.load_model(base_model_path)
    
    base_preds_train = base_model.predict(X_train_base) * CAPACITY_KWP
    residuals_train = y_train - base_preds_train
    
    residual_model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
    residual_model.fit(X_train_res, residuals_train)
    
    # 5. Fetch Forecast for Today
    print(f"\nFetching live forecast for {FORECAST_DATE}...")
    df_weather_hourly_today = get_forecast(LATITUDE, LONGITUDE, FORECAST_DATE, FORECAST_DATE)
    df_weather_15min_today = df_weather_hourly_today.resample('15min').interpolate(method='linear')
    df_weather_15min_today['is_day'] = df_weather_15min_today['is_day'].round()
    
    dates_today, X_today_base = prepare_features(df_weather_15min_today)
    X_today_res = X_today_base.copy()
    if 'snow_depth' in residual_cols:
        if 'snow_depth' in df_weather_15min_today.columns:
            X_today_res['snow_depth'] = df_weather_15min_today['snow_depth'].values
        else:
            X_today_res['snow_depth'] = 0.0
    
    # 6. Predict Schedule
    print(f"Generating 15-minute schedule for {FORECAST_DATE}...")
    preds_base = base_model.predict(X_today_base[feature_cols])
    preds_res = residual_model.predict(X_today_res[residual_cols])
    
    pred_base_only = preds_base * CAPACITY_KWP
    pred_base_only = np.where(df_weather_15min_today['is_day'] == 0, 0, pred_base_only)
    pred_base_only = np.maximum(0, pred_base_only)
    
    pred_native = (preds_base * CAPACITY_KWP) + preds_res
    pred_native = np.where(df_weather_15min_today['is_day'] == 0, 0, pred_native)
    pred_native = np.maximum(0, pred_native)
    
    schedule_df = pd.DataFrame({
        'Date_UTC': dates_today,
        'Date_Local': dates_today.dt.tz_convert(TIMEZONE).dt.tz_localize(None),
        'Base_Model_Power_kW': np.round(pred_base_only, 2),
        'Adjusted_Power_kW': np.round(pred_native, 2)
    })
    
    # Save CSV
    csv_filename = f"topola_schedule_{FORECAST_DATE.replace('-', '')}.csv"
    schedule_df.to_csv(csv_filename, index=False)
    print(f"\nSUCCESS! Saved schedule to {csv_filename}")
    
    # 7. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(schedule_df['Date_Local'], schedule_df['Base_Model_Power_kW'], color='blue', linewidth=2, linestyle='--', label='Base Model Only')
    plt.plot(schedule_df['Date_Local'], schedule_df['Adjusted_Power_kW'], color='orange', linewidth=2, label='Adjusted Power')
    plt.fill_between(schedule_df['Date_Local'], schedule_df['Adjusted_Power_kW'], color='orange', alpha=0.3)
    plt.title(f"Topola 5MW Generation Schedule ({FORECAST_DATE})")
    plt.xlabel(f"Local Time ({TIMEZONE})")
    plt.ylabel("Power (kW)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_filename = f"topola_schedule_{FORECAST_DATE.replace('-', '')}.png"
    plt.savefig(plot_filename)
    print(f"Saved schedule plot to {plot_filename}")

if __name__ == "__main__":
    main()
