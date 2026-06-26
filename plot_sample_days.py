import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import os
import warnings
from evaluate_all_plants import PLANTS, process_plant, get_archived_weather_15min
from hybrid_forecaster import HybridForecaster
warnings.filterwarnings("ignore")

# Let's use NC-1 as the primary example since it has a massive clean 1-year dataset
PLANT_NAME = 'NC-1'
CONFIG = PLANTS[PLANT_NAME]

def main():
    print(f"Loading {PLANT_NAME} data to generate 5 random day samples...")
    
    df_raw = pd.read_csv(CONFIG['csv'])
    df_raw['date_utc'] = pd.to_datetime(df_raw[CONFIG['date_format']])
    df_raw = df_raw.dropna(subset=['date_utc'])
    
    # Scale power to kW
    df_raw['actual_power_kw'] = df_raw[CONFIG['power_col']] / CONFIG['power_scale']
    df_raw.loc[df_raw['actual_power_kw'] < 0, 'actual_power_kw'] = 0
    
    df_raw.set_index('date_utc', inplace=True)
    df_15 = df_raw[['actual_power_kw']].resample('15min').mean().dropna()
    
    start_date = df_15.index.min().strftime('%Y-%m-%d')
    end_date = df_15.index.max().strftime('%Y-%m-%d')
    
    print(f"Fetching historical weather from Open-Meteo...")
    df_weather = get_archived_weather_15min(CONFIG['lat'], CONFIG['lon'], start_date, end_date)
    
    dataset = df_weather.copy()
    dataset = dataset.join(df_15, how='inner')
    dataset = dataset.dropna(subset=['actual_power_kw'])
    
    dataset.index = dataset.index.tz_localize('UTC')
    
    dataset['day_of_year'] = dataset.index.dayofyear
    train_mask = (dataset['day_of_year'] % 2 == 0)
    test_mask = ~train_mask
    
    df_train = dataset[train_mask].copy()
    df_test = dataset[test_mask].copy()
    
    print("Training Hybrid Model...")
    forecaster = HybridForecaster(CONFIG['lat'], CONFIG['lon'], CONFIG['kwp'], CONFIG['tilt'], CONFIG['azimuth'])
    forecaster.fit(df_train, df_train['actual_power_kw'])
    
    print("Predicting on unseen days...")
    phys_pred, ml_res, final_pred = forecaster.predict(df_test)
    df_test['pred_hybrid'] = final_pred
    
    # Aggregate to 1h
    df_test_1h = df_test[['actual_power_kw', 'pred_hybrid', 'is_day']].resample('1h').mean()
    
    # Pick 5 random days
    unique_dates = np.unique(df_test.index.date)
    random.seed(42)
    sample_dates = random.sample(list(unique_dates), 5)
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 20))
    fig.suptitle(f'Hybrid Model: 5 Random Unseen Days ({PLANT_NAME})', fontsize=20, y=0.98)
    
    for i, target_date in enumerate(sample_dates):
        # 15-minute data
        mask_15 = df_test.index.date == target_date
        day_df_15 = df_test[mask_15]
        
        # 1-hour data
        mask_1h = df_test_1h.index.date == target_date
        day_df_1h = df_test_1h[mask_1h]
        
        # Convert index to local timezone for plotting
        local_idx_15 = day_df_15.index.tz_convert(CONFIG['tz'])
        local_idx_1h = day_df_1h.index.tz_convert(CONFIG['tz'])
        
        # Plot 15-minute
        ax1 = axes[i, 0]
        ax1.plot(local_idx_15, day_df_15['actual_power_kw'], label='Measured', color='black', linewidth=2)
        ax1.plot(local_idx_15, day_df_15['pred_hybrid'], label='Predicted (Hybrid)', color='darkorange', linewidth=2, linestyle='--')
        ax1.set_title(f"{target_date} (15-Minute Resolution)")
        ax1.set_ylabel("Power (kW)")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        if i == 0:
            ax1.legend()
            
        # Plot 1-hour
        ax2 = axes[i, 1]
        ax2.plot(local_idx_1h, day_df_1h['actual_power_kw'], label='Measured', color='black', linewidth=2)
        ax2.plot(local_idx_1h, day_df_1h['pred_hybrid'], label='Predicted (Hybrid)', color='blue', linewidth=2, linestyle='-.')
        ax2.set_title(f"{target_date} (1-Hour Aggregation)")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        if i == 0:
            ax2.legend()
            
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    out_file = "hybrid_5_random_days_plot.png"
    plt.savefig(out_file, dpi=200)
    print(f"\nPlot saved successfully to {out_file}!")

if __name__ == "__main__":
    main()
