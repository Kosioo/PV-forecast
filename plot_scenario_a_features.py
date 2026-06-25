import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def main():
    # Load data for 14200
    df_15 = pd.read_csv('dataset_14200_15min.csv')
    df_15['date'] = pd.to_datetime(df_15['date'])
    df_15['day_of_year'] = df_15['date'].dt.dayofyear
    
    # Isolate test set (odd days)
    test_15 = df_15[df_15['day_of_year'] % 2 != 0].copy()
    
    # Filter outage days
    day_max = test_15.groupby('day_of_year')['actual_power_kw'].max()
    outage_days = day_max[day_max < 50].index
    test_15 = test_15[~test_15['day_of_year'].isin(outage_days)].copy()
    
    # Load Models
    base_model_path = os.path.join(os.path.dirname(__file__), "open-source-quartz-solar-forecast", "quartz_solar_forecast", "models", "model_10_202405.ubj")
    base_model = xgb.XGBRegressor()
    base_model.load_model(base_model_path)
    
    residual_model = xgb.XGBRegressor()
    residual_model.load_model('residual_model_14200.ubj')
    
    feature_cols = ['latitude_rounded', 'longitude_rounded', 'orientation', 'tilt', 'kwp',
                    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
                    'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                    'cloud_cover_high', 'wind_speed_10m', 'wind_direction_10m', 'is_day',
                    'direct_radiation', 'diffuse_radiation', 'date_month', 'date_day', 'date_hour']
                    
    # Generate 15-min predictions
    preds_15_base = base_model.predict(test_15[feature_cols])
    preds_15_res = residual_model.predict(test_15[feature_cols])
    preds_15 = (preds_15_base + preds_15_res) * 1000.0
    
    # Set to 0 at night or negative
    preds_15 = np.where(test_15['is_day'] == 0, 0, preds_15)
    preds_15 = np.maximum(0, preds_15)
    test_15['pred_native_15'] = preds_15
    
    # Scenario A: Average to Hourly
    test_15_hourly = test_15.set_index('date').resample('1h').mean()
    test_15_hourly = test_15_hourly.dropna(subset=['actual_power_kw']).reset_index()
    
    # Pick 3 interesting days
    sample_days = [151, 165, 201]  # E.g., late May/June/July odd days
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15), gridspec_kw={'width_ratios': [1.5, 1]})
    fig.suptitle('Day-Ahead Forecast Inputs & Output (Scenario A - Hourly)', fontsize=20, y=0.98)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    
    for i, day in enumerate(sample_days):
        day_data = test_15_hourly[test_15_hourly['day_of_year'] == day]
        if day_data.empty:
            continue
            
        ax_power = axes[i, 0]
        ax_weather = axes[i, 1]
        
        # Plot 1: Power (Actual vs Scenario A)
        ax_power.step(day_data['date'], day_data['actual_power_kw'], label='Actual Hourly Power', color='black', linewidth=3, where='post')
        ax_power.step(day_data['date'], day_data['pred_native_15'], label='Predicted Hourly Power (Scen A)', color='#ff7f0e', linewidth=3, linestyle='--', where='post')
        
        # Highlight Noon
        noon_time = day_data[day_data['date_hour'] == 12]['date'].values[0]
        ax_power.axvline(x=noon_time, color='red', linestyle=':', alpha=0.7, label='12:00 PM (Noon)')
        ax_power.text(noon_time, ax_power.get_ylim()[1]*0.9, ' NOON', color='red', fontweight='bold')
        
        ax_power.set_title(f"Day {day}: Hourly Power Production", fontsize=14, fontweight='bold')
        ax_power.set_ylabel("Power (kW)", fontsize=12)
        ax_power.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_power.grid(True, alpha=0.3)
        ax_power.legend(loc='upper left')
        ax_power.set_xlim([day_data['date'].min(), day_data['date'].max()])
        
        # Plot 2: Weather Features
        ax_weather.plot(day_data['date'], day_data['direct_radiation'], label='Direct Radiation (W/m²)', color='orange', linewidth=2)
        ax_weather.plot(day_data['date'], day_data['diffuse_radiation'], label='Diffuse Radiation (W/m²)', color='gold', linewidth=2)
        
        ax_weather2 = ax_weather.twinx()
        ax_weather2.plot(day_data['date'], day_data['cloud_cover'], label='Cloud Cover (%)', color='grey', linewidth=2, linestyle='-.')
        
        ax_weather.axvline(x=noon_time, color='red', linestyle=':', alpha=0.7)
        
        ax_weather.set_title(f"Day {day}: Day-Ahead Weather Inputs", fontsize=14, fontweight='bold')
        ax_weather.set_ylabel("Radiation (W/m²)", fontsize=12)
        ax_weather2.set_ylabel("Cloud Cover (%)", fontsize=12, color='grey')
        ax_weather.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_weather.grid(True, alpha=0.3)
        ax_weather.set_xlim([day_data['date'].min(), day_data['date'].max()])
        
        # Combine legends for weather
        lines, labels = ax_weather.get_legend_handles_labels()
        lines2, labels2 = ax_weather2.get_legend_handles_labels()
        ax_weather.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # Add text box for Noon features
        noon_row = day_data[day_data['date_hour'] == 12].iloc[0]
        noon_text = (f"Inputs at Noon:\n"
                     f"Dir Rad: {noon_row['direct_radiation']:.0f} W/m²\n"
                     f"Dif Rad: {noon_row['diffuse_radiation']:.0f} W/m²\n"
                     f"Cloud Cover: {noon_row['cloud_cover']:.0f}%\n"
                     f"Temp: {noon_row['temperature_2m']:.1f}°C")
        ax_weather.text(0.05, 0.45, noon_text, transform=ax_weather.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
                        fontsize=11)
        
    plt.savefig('C:/Users/konst/.gemini/antigravity/brain/e93d7397-cf9a-429a-9df7-309b8a19f218/feature_vs_power_dashboard.png', bbox_inches='tight', dpi=150)
    print("Saved feature_vs_power_dashboard.png")

if __name__ == '__main__':
    main()
