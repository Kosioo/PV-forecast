import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast"))
from quartz_solar_forecast.forecasts.v2 import TryolabsSolarPowerPredictor

def get_day_ahead_forecast(lat, lon, start_date, end_date):
    print(f"Fetching historical forecast from Open-Meteo for {start_date} to {end_date}...")
    
    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "is_day",
        "direct_radiation", "diffuse_radiation",
        "shortwave_radiation", "direct_normal_irradiance", "terrestrial_radiation"
    ]
    
    # Using historical-forecast-api
    url = f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={','.join(variables)}&timezone=GMT"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch forecast: {response.text}")
        
    data = response.json()
    hourly = data['hourly']
    
    df_weather = pd.DataFrame(hourly)
    
    # Convert 'time' to datetime, and all other columns to numeric
    df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)
    df_weather = df_weather.set_index('time')
    df_weather = df_weather.apply(pd.to_numeric, errors='coerce')
    
    return df_weather

def main():
    csv_file = "PV_measurements/14200_all_channels_20220101_20221231.csv"
    print(f"Loading actual data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print("Parsing timestamps...")
    df['date_utc'] = pd.to_datetime(df['utc_measured_on'], utc=True)
    
    df_power = df[['date_utc', 'ac_power_meter_3105']].copy()
    df_power = df_power.rename(columns={'ac_power_meter_3105': 'actual_power_kw'})
    
    # Set index for resampling
    df_power.set_index('date_utc', inplace=True)
    
    # Resample to 15min by averaging
    df_15min = df_power.resample('15min').mean()
    df_15min.loc[df_15min['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    
    # Also resample to Hourly
    df_1h = df_power.resample('1h').mean()
    df_1h.loc[df_1h['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    
    # Coordinates and Specs for SAS_SF1
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    LATITUDE = 35.813315
    LONGITUDE = -78.748991
    CAPACITY_KWP = 1000.0
    TILT = 20.0
    ORIENTATION = 180.0
    
    df_weather_hourly = get_day_ahead_forecast(LATITUDE, LONGITUDE, start_date, end_date)
    
    # Interpolate empty cells if any returned by Open-Meteo
    df_weather_hourly = df_weather_hourly.interpolate(method='linear')
    
    # Upsample weather to 15min
    print("Upsampling hourly forecast to 15-minute intervals...")
    df_weather_15min = df_weather_hourly.resample('15min').interpolate(method='linear')
    df_weather_15min['is_day'] = df_weather_15min['is_day'].round()
    
    # Prepare base predictor to use its cleaner
    predictor = TryolabsSolarPowerPredictor()
    predictor.load_model()
    
    def prepare_features(weather_df):
        df_copy = weather_df.copy().reset_index()
        df_copy = df_copy.rename(columns={'time': 'date_utc'})
        
        # Convert UTC to local time (naive) so the model extracts local hour
        df_copy['date_local'] = df_copy['date_utc'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
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
    df_15min['date'] = df_15min['date_utc'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    df_15min = df_15min.drop(columns=['date_utc'])
    
    dataset_15min = X_15min.copy()
    dataset_15min['date'] = dates_15min.dt.tz_convert('US/Eastern').dt.tz_localize(None)
    dataset_15min = pd.merge(dataset_15min, df_15min, on='date', how='inner')
    
    # Drop rows with NaN target values
    dataset_15min = dataset_15min.dropna(subset=['actual_power_kw'])
    
    # Determine and filter outage days
    dataset_15min['day_of_year'] = dataset_15min['date'].dt.dayofyear
    day_max = dataset_15min.groupby('day_of_year')['actual_power_kw'].max()
    outage_days = day_max[day_max < 50].index
    print(f"Found {len(outage_days)} outage days. Removing them from training.")
    dataset_15min = dataset_15min[~dataset_15min['day_of_year'].isin(outage_days)].copy()
    
    # Split into even/odd days
    print("Splitting dataset into even/odd days...")
    train_mask = (dataset_15min['day_of_year'] % 2 == 0)
    test_mask = (dataset_15min['day_of_year'] % 2 != 0)
    
    X_train = dataset_15min[train_mask].drop(columns=['date', 'actual_power_kw', 'day_of_year'])
    y_train = dataset_15min[train_mask]['actual_power_kw'] / 1000.0 # Convert to MW
    
    X_test = dataset_15min[test_mask].drop(columns=['date', 'actual_power_kw', 'day_of_year'])
    y_test = dataset_15min[test_mask]['actual_power_kw'] / 1000.0
    # Fine-tune the model by explicitly training on the residuals
    print("Fine-tuning by training a residual model...")
    base_model_path = os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast", "quartz_solar_forecast", "models", "model_10_202405.ubj")
    
    base_model = xgb.XGBRegressor()
    base_model.load_model(base_model_path)
    
    base_preds_train = base_model.predict(X_train)
    base_preds_test = base_model.predict(X_test)
    
    residual_train = y_train - base_preds_train
    residual_test = y_test - base_preds_test
    
    residual_model = xgb.XGBRegressor()
    residual_model.fit(X_train, residual_train, eval_set=[(X_test, residual_test)], verbose=True)
    
    residual_model_path = "residual_model_14200.ubj"
    residual_model.save_model(residual_model_path)
    print(f"Saved residual model to {residual_model_path}")
    
    # Save the prepared datasets (without outage days filtering, we can filter in evaluate script)
    dataset_15min_full = X_15min.copy()
    dataset_15min_full['date'] = dates_15min.dt.tz_convert('US/Eastern').dt.tz_localize(None)
    dataset_15min_full = pd.merge(dataset_15min_full, df_15min, on='date', how='inner')
    dataset_15min_full = dataset_15min_full.dropna(subset=['actual_power_kw'])
    dataset_15min_full.to_csv('dataset_14200_15min.csv', index=False)
    
    df_1h = df_1h.reset_index()
    df_1h['date'] = df_1h['date_utc'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
    df_1h = df_1h.drop(columns=['date_utc'])
    
    dataset_1h = X_1h.copy()
    dataset_1h['date'] = dates_1h.dt.tz_convert('US/Eastern').dt.tz_localize(None)
    dataset_1h = pd.merge(dataset_1h, df_1h, on='date', how='inner')
    dataset_1h = dataset_1h.dropna(subset=['actual_power_kw'])
    dataset_1h.to_csv('dataset_14200_1h.csv', index=False)
    
    print("Data preparation and tuning complete!")

if __name__ == "__main__":
    main()
