import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add the cloned open-source-quartz-solar-forecast to sys.path so we can import from it directly
# This allows us to bypass pip installing it and bypass the pv-site-prediction dependency error
sys.path.append(os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast"))

from quartz_solar_forecast.forecasts.v2 import TryolabsSolarPowerPredictor

def main():
    print("Starting Open Climate Fix Quartz Solar Forecast (XGBoost Model)...")

    # =========================================================================
    # USER CONFIGURATION:
    # Update the latitude, longitude, and capacity_kwp with your 
    # actual PV plant's values.
    # =========================================================================
    LATITUDE = 51.5072
    LONGITUDE = -0.1276
    CAPACITY_KWP = 1000.0 # e.g. 1MW plant = 1000 kWp
    TILT = 30.0
    ORIENTATION = 180.0 # 180 is South-facing

    print(f"Configured PV Site: Lat {LATITUDE}, Lon {LONGITUDE}, Capacity {CAPACITY_KWP} kWp")

    # Get the current date and round down to start of the hour
    now = pd.Timestamp.now()
    start_date = now.strftime("%Y-%m-%d")

    print(f"Loading XGBoost model and downloading weights from Hugging Face if necessary...")
    predictor = TryolabsSolarPowerPredictor()
    predictor.load_model()

    print(f"Fetching weather from Open-Meteo and generating predictions for {start_date}...")
    predictions_df = predictor.predict_power_output(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=start_date,
        kwp=CAPACITY_KWP,
        orientation=ORIENTATION,
        tilt=TILT
    )

    print("\nForecast successfully generated! Here are the first few rows:")
    print(predictions_df.head())

    # Save to CSV
    csv_filename = "xgb_forecast_output.csv"
    predictions_df.to_csv(csv_filename)
    print(f"\nForecast data saved to {csv_filename}")

    # Plot the results
    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    
    # ensure the index is datetime
    if not isinstance(predictions_df.index, pd.DatetimeIndex):
        predictions_df.index = pd.to_datetime(predictions_df.index)
        
    plt.plot(predictions_df.index, predictions_df['power_kw'], label='Forecasted Power (kW)', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.title(f'Quartz Solar XGBoost Forecast\nLat: {LATITUDE}, Lon: {LONGITUDE}, Cap: {CAPACITY_KWP} kWp')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_filename = "xgb_forecast_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    main()
