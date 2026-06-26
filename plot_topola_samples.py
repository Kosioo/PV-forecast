import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
import os
import warnings
from hybrid_forecaster import HybridForecaster
from evaluate_all_plants import get_archived_weather_15min
from evaluate_topola import parse_topola_data, LATITUDE, LONGITUDE, CAPACITY_KWP, TILT, ORIENTATION, TIMEZONE
warnings.filterwarnings("ignore")

def main():
    print("Loading Topola1.xlsx data to generate 5 random day samples...")
    
    df_15 = parse_topola_data()
    
    start_date = df_15.index.min().strftime('%Y-%m-%d')
    end_date = df_15.index.max().strftime('%Y-%m-%d')
    
    print(f"Fetching historical weather from Open-Meteo ({start_date} to {end_date})...")
    df_weather = get_archived_weather_15min(LATITUDE, LONGITUDE, start_date, end_date)
    
    dataset = df_weather.copy()
    dataset.index = dataset.index.tz_localize('UTC')
    dataset = dataset.join(df_15, how='inner')
    dataset = dataset.dropna(subset=['actual_power_kw'])
    
    dataset['day_of_year'] = dataset.index.dayofyear
    train_mask = (dataset['day_of_year'] % 2 == 0)
    test_mask = ~train_mask
    
    df_train = dataset[train_mask].copy()
    df_test = dataset[test_mask].copy()
    
    print(f"Training Hybrid Model on {len(df_train)} intervals...")
    forecaster = HybridForecaster(LATITUDE, LONGITUDE, CAPACITY_KWP, TILT, ORIENTATION)
    forecaster.fit(df_train, df_train['actual_power_kw'])
    
    print(f"Predicting on {len(df_test)} unseen test intervals...")
    phys_pred, ml_res, final_pred = forecaster.predict(df_test)
    df_test['pred_hybrid'] = final_pred
    
    # Aggregate to 1h
    df_test_1h = df_test[['actual_power_kw', 'pred_hybrid', 'is_day']].resample('1h').mean()
    
    # Pick 5 random days
    unique_dates = np.unique(df_test.index.date)
    if len(unique_dates) < 5:
        print(f"WARNING: Only {len(unique_dates)} unique days available for testing. Plotting all of them.")
        sample_dates = list(unique_dates)
    else:
        random.seed(42)
        sample_dates = random.sample(list(unique_dates), 5)
    
    n_plots = len(sample_dates)
    fig, axes = plt.subplots(nrows=max(1, n_plots), ncols=2, figsize=(16, 4 * n_plots))
    if n_plots == 1:
        axes = np.array([axes]) # ensure 2D array if only 1 plot
        
    fig.suptitle('Hybrid Model: Random Unseen Days (Topola)', fontsize=20, y=0.98)
    
    for i, target_date in enumerate(sample_dates):
        # 15-minute data
        mask_15 = df_test.index.date == target_date
        day_df_15 = df_test[mask_15]
        
        # 1-hour data
        mask_1h = df_test_1h.index.date == target_date
        day_df_1h = df_test_1h[mask_1h]
        
        # Convert index to local timezone for plotting
        local_idx_15 = day_df_15.index.tz_convert(TIMEZONE)
        local_idx_1h = day_df_1h.index.tz_convert(TIMEZONE)
        
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
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_file = "topola_5_random_days_plot.png"
    plt.savefig(out_file, dpi=200)
    print(f"\nPlot saved successfully to {out_file}!")

if __name__ == "__main__":
    main()
