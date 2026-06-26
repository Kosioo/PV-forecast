import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Ensure Visuals directory exists
os.makedirs("Visuals", exist_ok=True)

# Plant Coordinates (SAS_SF1, NC)
CENTER_LAT = 35.813
CENTER_LON = -78.749

# Unseen Test Days (Odd days from our previous tests, e.g., Day 165 is June 14, 2022)
TEST_DATES = ["2022-06-13", "2022-07-21", "2022-10-15"]

def fetch_spatial_weather(date_str):
    print(f"Fetching spatial grid for {date_str}...")
    
    # Create 10x10 grid (100 points) covering roughly 2 degrees (~200km box)
    lats = np.linspace(CENTER_LAT - 1.0, CENTER_LAT + 1.0, 10)
    lons = np.linspace(CENTER_LON - 1.0, CENTER_LON + 1.0, 10)
    
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    lat_list = grid_lat.flatten()
    lon_list = grid_lon.flatten()
    
    lat_str = ",".join([f"{lat:.3f}" for lat in lat_list])
    lon_str = ",".join([f"{lon:.3f}" for lon in lon_list])
    
    url = f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat_str}&longitude={lon_str}&start_date={date_str}&end_date={date_str}&hourly=cloud_cover,direct_radiation,temperature_2m&timezone=America/New_York"
    
    response = requests.get(url)
    data = response.json()
    
    if type(data) is list:
        # Multiple locations return a list of responses
        return data, lat_list, lon_list
    else:
        raise Exception(f"Failed to fetch data: {data}")

def plot_spatial_day(date_str):
    data_list, lat_list, lon_list = fetch_spatial_weather(date_str)
    
    # We want noon local time (which is 12:00 PM)
    # The timezone is America/New_York, so index 12 in hourly data represents 12:00 PM
    noon_index = 12
    
    cloud_covers = []
    direct_rads = []
    temps = []
    
    for loc_data in data_list:
        hourly = loc_data['hourly']
        cloud_covers.append(hourly['cloud_cover'][noon_index])
        direct_rads.append(hourly['direct_radiation'][noon_index])
        temps.append(hourly['temperature_2m'][noon_index])
        
    cloud_covers = np.array(cloud_covers)
    direct_rads = np.array(direct_rads)
    temps = np.array(temps)
    
    # Set up interpolation grid for smooth contour plots
    grid_y, grid_x = np.mgrid[CENTER_LAT-1:CENTER_LAT+1:100j, CENTER_LON-1:CENTER_LON+1:100j]
    
    grid_cloud = griddata((lat_list, lon_list), cloud_covers, (grid_y, grid_x), method='cubic')
    grid_rad = griddata((lat_list, lon_list), direct_rads, (grid_y, grid_x), method='cubic')
    grid_temp = griddata((lat_list, lon_list), temps, (grid_y, grid_x), method='cubic')
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Spatial Weather Snapshot at Noon (Day-Ahead Forecast for {date_str})", fontsize=18, fontweight='bold')
    
    # Cloud Cover Plot
    c1 = axes[0].contourf(grid_x, grid_y, grid_cloud, levels=20, cmap='Blues', alpha=0.9)
    axes[0].plot(CENTER_LON, CENTER_LAT, 'r*', markersize=15, label='SAS_SF1 Plant')
    axes[0].set_title('Cloud Cover (%)')
    fig.colorbar(c1, ax=axes[0])
    
    # Direct Radiation Plot
    c2 = axes[1].contourf(grid_x, grid_y, grid_rad, levels=20, cmap='YlOrRd', alpha=0.9)
    axes[1].plot(CENTER_LON, CENTER_LAT, 'b*', markersize=15, label='SAS_SF1 Plant')
    axes[1].set_title('Direct Radiation (W/m²)')
    fig.colorbar(c2, ax=axes[1])
    
    # Temperature Plot
    c3 = axes[2].contourf(grid_x, grid_y, grid_temp, levels=20, cmap='coolwarm', alpha=0.9)
    axes[2].plot(CENTER_LON, CENTER_LAT, 'k*', markersize=15, label='SAS_SF1 Plant')
    axes[2].set_title('Temperature (°C)')
    fig.colorbar(c3, ax=axes[2])
    
    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
    # Extract the exact feature array the model sees for the central plant
    # The center is roughly at index 44 or 45 in the flattened array, but let's just grab the one closest to CENTER_LAT/LON
    center_idx = np.argmin((lat_list - CENTER_LAT)**2 + (lon_list - CENTER_LON)**2)
    center_data = data_list[center_idx]['hourly']
    
    features_text = (
        f"--- Model Input Vector at Noon ---\n"
        f"Time: {center_data['time'][noon_index]}\n"
        f"Cloud Cover: {center_data['cloud_cover'][noon_index]} %\n"
        f"Direct Rad: {center_data['direct_radiation'][noon_index]} W/m²\n"
        f"Temperature: {center_data['temperature_2m'][noon_index]} °C\n"
        f"Lat/Lon: {lat_list[center_idx]:.2f}, {lon_list[center_idx]:.2f}\n"
        f"Capacity: 1000 kW\n"
        f"Tilt/Orient: 20°, 180°"
    )
    
    # Add text box below the plots
    plt.figtext(0.5, -0.05, features_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    save_path = f"C:/Users/konst/.gemini/antigravity/brain/e93d7397-cf9a-429a-9df7-309b8a19f218/spatial_weather_{date_str}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Also save to the repo Visuals directory as requested
    repo_path = f"Visuals/spatial_weather_{date_str}.png"
    plt.savefig(repo_path, bbox_inches='tight', dpi=150)
    
    print(f"Saved visualization to {save_path} and {repo_path}")
    plt.close()

if __name__ == '__main__':
    for date in TEST_DATES:
        plot_spatial_day(date)
