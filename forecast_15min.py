import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests_cache
from retry_requests import retry
import openmeteo_requests

# Add the cloned open-source-quartz-solar-forecast to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast"))
from quartz_solar_forecast.forecasts.v2 import TryolabsSolarPowerPredictor

class Minutely15WeatherService:
    def get_15min_weather(self, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
        variables = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "wind_direction_10m",
            "is_day",
            "direct_radiation",
            "diffuse_radiation",
            "shortwave_radiation",
            "direct_normal_irradiance",
            "terrestrial_radiation"
        ]
        
        # Build URL for 15-minute data
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&minutely_15={','.join(variables)}&start_date={start_date}&end_date={end_date}&timezone=GMT"
        
        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        response = openmeteo.weather_api(url, params={})
        
        minutely_15 = response[0].Minutely15()
        minutely_15_data = {
            "date": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=False),
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=False),
                freq=pd.Timedelta(seconds=minutely_15.Interval()),
                inclusive="left",
            )
        }
        
        for i, variable in enumerate(variables):
            minutely_15_data[variable] = minutely_15.Variables(i).ValuesAsNumpy()
            
        df = pd.DataFrame(minutely_15_data)
        
        # Add required columns that the predictor expects (from weather_service)
        df["latitude_rounded"] = latitude
        df["longitude_rounded"] = longitude
        
        return df

class Minutely15Predictor(TryolabsSolarPowerPredictor):
    def get_data(self, latitude, longitude, start_date, kwp, orientation=180, tilt=30):
        start_date_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_datetime = start_date_datetime + datetime.timedelta(days=2)
        end_date = end_date_datetime.strftime("%Y-%m-%d")

        weather_service = Minutely15WeatherService()
        weather_data = weather_service.get_15min_weather(latitude, longitude, start_date, end_date)

        weather_data["orientation"] = orientation
        weather_data["tilt"] = tilt
        weather_data["kwp"] = kwp
        return weather_data

    def predict_power_output(self, latitude, longitude, start_date, kwp, orientation=180, tilt=30):
        data = self.get_data(latitude, longitude, start_date, kwp, orientation, tilt)
        cleaned_data = self.clean(data)
        
        # Clean method leaves "date_minute" in COLUMNS_TO_DROP. But wait, we need date_hour, date_minute, etc.
        # The Tryolabs model drops date_minute! So it only uses date_hour. 
        # This is fine, it will just evaluate the 15-min weather features against the hour of the day.
        
        features = cleaned_data.drop(columns=[self.DATE_COLUMN])
        expected_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                         'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                         'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                         'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                         'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
        features = features[expected_cols]
        predictions = self.model.predict(features)
        predictions_df = pd.DataFrame(predictions, columns=["prediction"])
        final_data = cleaned_data.join(predictions_df)
        final_data.loc[final_data["is_day"] == 0, "prediction"] = 0
        final_data.loc[final_data["prediction"] < 0, "prediction"] = 0
        df = final_data[[self.DATE_COLUMN, "prediction"]]
        df = df.rename(columns={"prediction": "power_kw"})
        return df

def main():
    print("Starting 15-Minute Resolution XGBoost Forecast...")
    
    # Simon Solar Farm, Georgia
    LATITUDE = 33.6762
    LONGITUDE = -83.676
    CAPACITY_KWP = 33000.0
    TILT = 20.0
    ORIENTATION = 180.0

    now = pd.Timestamp.now()
    start_date = now.strftime("%Y-%m-%d")

    predictor = Minutely15Predictor()
    predictor.load_model()

    print(f"Fetching 15-Minute weather from Open-Meteo and generating predictions...")
    predictions_df = predictor.predict_power_output(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=start_date,
        kwp=CAPACITY_KWP,
        orientation=ORIENTATION,
        tilt=TILT
    )

    print("\nForecast successfully generated! Here are the first few rows:")
    print(predictions_df.head(10))

    csv_filename = "forecast_15min_output.csv"
    predictions_df.to_csv(csv_filename, index=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_df['date'], predictions_df['power_kw'], label='15-Min Forecast (kW)', color='green')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.title(f'15-Minute Forecast - Simon Solar Farm (33 MWp)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_15min_plot.png")
    print("Saved 15-minute forecast plot to forecast_15min_plot.png")

if __name__ == "__main__":
    main()
