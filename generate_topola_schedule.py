import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import warnings
from hybrid_forecaster import PVLibPhysicalModel
warnings.filterwarnings("ignore")

# ================= Configuration =================
LATITUDE = 41.509238
LONGITUDE = 23.796700
CAPACITY_KWP = 5000.0
TILT = 38.0  
ORIENTATION = 180.0
TIMEZONE = 'Europe/Sofia'
# =================================================

def get_day_ahead_forecast(lat, lon, target_date_str):
    print(f"Fetching 14-day Open-Meteo forecast for {target_date_str}...")
    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
        "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "is_day",
        "direct_radiation", "diffuse_radiation", "shortwave_radiation",
        "direct_normal_irradiance", "terrestrial_radiation"
    ]
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&minutely_15={','.join(variables)}&timezone=GMT&past_days=1&forecast_days=3"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch weather: {response.text}")
        
    data = response.json()
    minutely_15 = data['minutely_15']
    
    df_weather = pd.DataFrame(minutely_15)
    df_weather['date_utc'] = pd.to_datetime(df_weather['time'])
    df_weather.set_index('date_utc', inplace=True)
    df_weather = df_weather.drop(columns=['time'])
    
    # Required names for pvlib
    df_weather['ghi'] = df_weather['shortwave_radiation']
    df_weather['dni'] = df_weather['direct_normal_irradiance']
    df_weather['dhi'] = df_weather['diffuse_radiation']
    df_weather['temp_air'] = df_weather['temperature_2m']
    df_weather['wind_speed'] = df_weather['wind_speed_10m']
    
    # Filter only for the target date
    target_date = pd.to_datetime(target_date_str).date()
    df_weather = df_weather[df_weather.index.date == target_date]
    
    return df_weather

def main():
    print("=====================================================")
    print(" Topola (5MW) Day-Ahead Schedule (Physical Model)")
    print("=====================================================")
    
    target_date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    df_weather = get_day_ahead_forecast(LATITUDE, LONGITUDE, target_date_str)
    
    # Needs to be tz-aware for pvlib
    df_weather.index = df_weather.index.tz_localize('UTC')
    
    # We only use Physical Model since we don't have enough Topola historical data for ML
    physical_model = PVLibPhysicalModel(LATITUDE, LONGITUDE, CAPACITY_KWP, TILT, ORIENTATION)
    
    print("Generating pure physical PV schedule...")
    phys_pred = physical_model.predict(df_weather)
    
    df_weather['pred_kw'] = phys_pred
    df_weather['pred_kw'] = np.maximum(0, df_weather['pred_kw'])
    df_weather['pred_kw'] = np.where(df_weather['is_day'] == 0, 0, df_weather['pred_kw'])
    
    # Convert index back to local Sofia time for export/plotting
    df_local = df_weather.copy()
    df_local.index = df_local.index.tz_convert(TIMEZONE)
    
    csv_filename = f"topola_schedule_{target_date_str.replace('-','')}_physical.csv"
    export_df = df_local[['pred_kw']].rename(columns={'pred_kw': 'Predicted_Power_kW'})
    export_df.to_csv(csv_filename)
    print(f"Schedule exported to {csv_filename}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df_local.index, df_local['pred_kw'], label='Predicted Physical Output (kW)', color='darkorange', linewidth=2.5)
    plt.fill_between(df_local.index, df_local['pred_kw'], color='orange', alpha=0.2)
    
    plt.title(f"Topola 5MW Day-Ahead Prediction - {target_date_str}", fontsize=16)
    plt.ylabel("Power Output (kW)", fontsize=12)
    plt.xlabel(f"Local Time ({TIMEZONE})", fontsize=12)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    png_filename = f"topola_schedule_{target_date_str.replace('-','')}_physical.png"
    plt.savefig(png_filename, dpi=300)
    print(f"Plot saved to {png_filename}")

if __name__ == "__main__":
    main()
