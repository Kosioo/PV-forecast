import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Import the predictor
sys.path.append(os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast"))
from quartz_solar_forecast.forecasts.v2 import TryolabsSolarPowerPredictor
from validate_model import calculate_errors

def get_archived_weather_15min(lat, lon, start_date, end_date):
    print("Fetching historical hourly weather from Open-Meteo Archive API...")
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
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather = df_weather.set_index('time')
    
    # Forward fill or interpolate to 15 minute intervals
    print("Upsampling hourly weather to 15-minute intervals...")
    df_15min = df_weather.resample('15min').interpolate(method='linear')
    df_15min['is_day'] = df_15min['is_day'].round() # is_day should be 0 or 1
    
    df_15min = df_15min.reset_index()
    df_15min = df_15min.rename(columns={'time': 'date'})
    return df_15min

def main():
    csv_file = "PV_measurements/14201_all_channels_20220101_20221231.csv"
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 1. Parse and Resample Actual Data
    print("Parsing timestamps and resampling to 15-minute intervals...")
    df['date'] = pd.to_datetime(df['utc_measured_on'])
    
    df_power = df[['date', 'ac_power_meter_3106']].copy()
    df_power = df_power.rename(columns={'ac_power_meter_3106': 'actual_power_kw'})
    
    df_power.set_index('date', inplace=True)
    df_15min = df_power.resample('15min').mean()
    df_15min.loc[df_15min['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    df_15min = df_15min.reset_index()
    
    # Validate over July 2022
    start_date = "2022-07-01"
    end_date = "2022-07-07"
    print(f"Selecting validation period: {start_date} to {end_date}")
    
    mask = (df_15min['date'] >= start_date) & (df_15min['date'] < end_date)
    actuals = df_15min[mask].copy()
    
    # 2. Fetch Historical Weather & Generate Forecast
    LATITUDE = 35.812367
    LONGITUDE = -78.749346
    CAPACITY_KWP = 1340.0
    TILT = 20.0
    ORIENTATION = 180.0
    
    weather_data = get_archived_weather_15min(LATITUDE, LONGITUDE, start_date, end_date)
    weather_data["orientation"] = ORIENTATION
    weather_data["tilt"] = TILT
    weather_data["kwp"] = CAPACITY_KWP
    weather_data["latitude_rounded"] = LATITUDE
    weather_data["longitude_rounded"] = LONGITUDE
    
    print("Generating 15-min Forecast with XGBoost...")
    predictor = TryolabsSolarPowerPredictor()
    predictor.load_model()
    
    cleaned_data = predictor.clean(weather_data)
    features = cleaned_data.drop(columns=[predictor.DATE_COLUMN])
    expected_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                     'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                     'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                     'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                     'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
    features = features[expected_cols]
    
    predictions = predictor.model.predict(features)
    
    predictions_df = pd.DataFrame(predictions, columns=["prediction"])
    final_data = cleaned_data.join(predictions_df)
    final_data.loc[final_data["is_day"] == 0, "prediction"] = 0
    final_data.loc[final_data["prediction"] < 0, "prediction"] = 0
    
    forecasts = final_data[[predictor.DATE_COLUMN, "prediction"]].copy()
    
    # Tryolabs model natively outputs predictions in Megawatts (MW). 
    # We must multiply by 1000 to convert to Kilowatts (kW) to match the actuals.
    forecasts["prediction"] = forecasts["prediction"] * 1000.0
    
    forecasts = forecasts.rename(columns={"prediction": "forecast_power_kw"})
    
    # 3. Merge Actuals and Forecasts
    forecasts['date'] = pd.to_datetime(forecasts['date']).dt.tz_localize(None)
    actuals['date'] = pd.to_datetime(actuals['date'], utc=True).dt.tz_localize(None)
    
    merged = pd.merge(actuals, forecasts, on='date', how='inner')
    print(f"Successfully merged {len(merged)} 15-minute intervals.")
    
    # 4. Calculate Errors
    print("\nCalculating error metrics between Forecast and Actuals...")
    errors = calculate_errors(merged['actual_power_kw'], merged['forecast_power_kw'])
    
    print("\nValidation Results (SAS_SF2 North Carolina - July 2022):")
    print(f"Mean Absolute Error (MAE): {errors['MAE']:.2f} kW")
    print(f"Root Mean Square Error (RMSE): {errors['RMSE']:.2f} kW")
    print(f"Mean Absolute Percentage Error (MAPE): {errors['MAPE']:.2f}%")
    
    # 5. Plot
    plt.figure(figsize=(15, 6))
    plt.plot(merged['date'], merged['actual_power_kw'], label='Actual (Measured)', color='red', alpha=0.7)
    plt.plot(merged['date'], merged['forecast_power_kw'], label='Forecast (Open-Meteo + XGBoost)', color='blue', alpha=0.7)
    plt.title('Validation: Actual vs Forecast (15-min intervals) - SAS_SF2 NC (July 2022)')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("real_data_validation_plot.png")
    print("Saved plot to real_data_validation_plot.png")

if __name__ == "__main__":
    main()
