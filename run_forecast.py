import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# The Open Climate Fix Quartz Solar Forecast library
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite

def main():
    print("Starting Open Climate Fix Quartz Solar Forecast...")

    # =========================================================================
    # USER CONFIGURATION:
    # Please update the latitude, longitude, and capacity_kwp with your 
    # actual PV plant's values.
    # We are using a placeholder location in London as a default test.
    # =========================================================================
    LATITUDE = 51.5072
    LONGITUDE = -0.1276
    CAPACITY_KWP = 1000.0 # e.g. 1MW plant = 1000 kWp
    
    # Create the PVSite object
    site = PVSite(latitude=LATITUDE, longitude=LONGITUDE, capacity_kwp=CAPACITY_KWP)
    print(f"Configured PV Site: Lat {LATITUDE}, Lon {LONGITUDE}, Capacity {CAPACITY_KWP} kWp")

    # Run the forecast for the next 48 hours
    # The default nwp_source is "icon" but you can also try "gfs" or "ecmwf"
    print("Running forecast using 'icon' NWP data. This will download weather data and model weights...")
    ts_now = datetime.now()
    
    predictions_df = run_forecast(site=site, ts=ts_now, nwp_source="icon")
    
    print("\nForecast successfully generated! Here are the first few rows:")
    print(predictions_df.head())

    # Save to CSV
    csv_filename = "forecast_output.csv"
    predictions_df.to_csv(csv_filename)
    print(f"\nForecast data saved to {csv_filename}")

    # Plot the results
    print("Generating plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(predictions_df.index, predictions_df['power_kw'], label='Forecasted Power (kW)')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.title(f'Quartz Solar Forecast (Lat: {LATITUDE}, Lon: {LONGITUDE}, Cap: {CAPACITY_KWP} kWp)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plot_filename = "forecast_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    main()
