import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from validate_model import calculate_errors

sys.path.append(os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast"))
from quartz_solar_forecast.forecasts.v2 import TryolabsSolarPowerPredictor

def get_archived_weather_hourly(lat, lon, start_date, end_date):
    print(f"Fetching historical hourly weather from Open-Meteo Archive API for {start_date} to {end_date}...")
    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "is_day",
        "direct_radiation", "diffuse_radiation", "shortwave_radiation",
        "direct_normal_irradiance", "terrestrial_radiation"
    ]
    
    # We use timezone=GMT to avoid DST duplicate label issues
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={','.join(variables)}&timezone=GMT"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch weather: {response.text}")
        
    data = response.json()
    hourly = data['hourly']
    
    df_weather = pd.DataFrame(hourly)
    df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)
    df_weather = df_weather.set_index('time')
    return df_weather

def main():
    csv_file = "PV_measurements/14201_all_channels_20220101_20221231.csv"
    print(f"Loading actual data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Keep as UTC for safe resampling
    print("Parsing timestamps...")
    df['date_utc'] = pd.to_datetime(df['utc_measured_on'], utc=True)
    
    df_power = df[['date_utc', 'ac_power_meter_3106']].copy()
    df_power = df_power.rename(columns={'ac_power_meter_3106': 'actual_power_kw'})
    
    # Set index for resampling
    df_power.set_index('date_utc', inplace=True)
    
    # Resample to 15min by averaging
    df_15min = df_power.resample('15min').mean()
    df_15min.loc[df_15min['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    
    # Also resample to Hourly
    df_1h = df_power.resample('1h').mean()
    df_1h.loc[df_1h['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    
    # Fetch 2022 Weather Data
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    LATITUDE = 35.812367
    LONGITUDE = -78.749346
    CAPACITY_KWP = 1340.0
    TILT = 20.0
    ORIENTATION = 180.0
    
    df_weather_hourly = get_archived_weather_hourly(LATITUDE, LONGITUDE, start_date, end_date)
    
    # Upsample weather to 15min
    print("Upsampling hourly weather to 15-minute intervals...")
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
        # The Tryolabs model expects the date column to be in DATE_COLUMN ('date')
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
        
        # Return UTC dates for merging, and features
        return cleaned['date_utc'], features[expected_cols]
    
    dates_15min, X_15min = prepare_features(df_weather_15min)
    dates_1h, X_1h = prepare_features(df_weather_hourly)
    
    # Target values (we need to align actual_power_kw with dates)
    df_15min = df_15min.reset_index().rename(columns={'date_utc': 'date'})
    # Convert dates to naive UTC string format so we can save to CSV safely
    df_15min['date'] = df_15min['date'].dt.tz_localize(None)
    
    # Merge targets into features to ensure perfect alignment
    dataset_15min = X_15min.copy()
    dataset_15min['date'] = dates_15min.dt.tz_localize(None)
    dataset_15min = pd.merge(dataset_15min, df_15min, on='date', how='inner')
    
    # Drop rows with NaN target values (missing measurements)
    dataset_15min = dataset_15min.dropna(subset=['actual_power_kw'])
    
    # Split into even/odd days
    print("Splitting dataset into even/odd days...")
    dataset_15min['day_of_year'] = dataset_15min['date'].dt.dayofyear
    train_mask = (dataset_15min['day_of_year'] % 2 == 0) # Even days
    test_mask = (dataset_15min['day_of_year'] % 2 != 0) # Odd days
    
    # Drop date columns for training
    X_train = dataset_15min[train_mask].drop(columns=['date', 'actual_power_kw', 'day_of_year'])
    y_train = dataset_15min[train_mask]['actual_power_kw'] / 1000.0 # Convert kW to MW target
    
    X_test = dataset_15min[test_mask].drop(columns=['date', 'actual_power_kw', 'day_of_year'])
    y_test = dataset_15min[test_mask]['actual_power_kw'] / 1000.0
    
    # Fine-tune the model
    print("Fine-tuning base XGBoost model...")
    # XGBRegressor's fit method supports xgb_model to continue training
    base_model_path = os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast", "quartz_solar_forecast", "models", "model_10_202405.ubj")
    
    tuned_model = xgb.XGBRegressor()
    # We can just fit using the existing model path as starting point
    tuned_model.fit(X_train, y_train, xgb_model=base_model_path, eval_set=[(X_test, y_test)], verbose=True)
    
    tuned_model_path = "tuned_model.ubj"
    tuned_model.save_model(tuned_model_path)
    print(f"Saved tuned model to {tuned_model_path}")
    
    # Save the prepared datasets for evaluation step
    dataset_15min.to_csv('dataset_15min.csv', index=False)
    
    # For hourly, we merge and save
    df_1h = df_1h.reset_index().rename(columns={'date_utc': 'date'})
    df_1h['date'] = df_1h['date'].dt.tz_localize(None)
    dataset_1h = X_1h.copy()
    dataset_1h['date'] = dates_1h.dt.tz_localize(None)
    dataset_1h = pd.merge(dataset_1h, df_1h, on='date', how='inner')
    dataset_1h = dataset_1h.dropna(subset=['actual_power_kw'])
    dataset_1h.to_csv('dataset_1h.csv', index=False)
    
    print("Data preparation and tuning complete!")

if __name__ == "__main__":
    main()
