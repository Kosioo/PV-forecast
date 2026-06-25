import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_correction_factor(actual_recent: pd.Series, forecast_recent: pd.Series) -> float:
    """
    Calculates a multiplicative correction factor based on the mismatch 
    over the recent measurement window (e.g. last 1-2 hours).
    """
    # Only calculate factor when power is significant to avoid divide-by-zero or morning anomalies
    mask = forecast_recent > (forecast_recent.max() * 0.05)
    
    if not mask.any():
        return 1.0 # No correction at night or very low light
        
    actual_sum = actual_recent[mask].sum()
    forecast_sum = forecast_recent[mask].sum()
    
    if forecast_sum == 0:
        return 1.0
        
    factor = actual_sum / forecast_sum
    
    # Cap the correction factor to prevent wild swings (e.g., max 30% adjustment)
    factor = max(0.7, min(1.3, factor))
    return factor

def main():
    print("--- 2-Hour Ahead Intraday Correction Schedule ---")
    
    try:
        df = pd.read_csv("forecast_15min_output.csv")
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print("Please run forecast_15min.py first.")
        return

    # Simulate live environment: Let's assume the current time is 12:00 PM
    # and we just received actual production data for 10:00 AM to 12:00 PM.
    current_time_idx = df.index[df['date'].dt.hour == 12][0]
    
    # Historical window (last 2 hours)
    window_start = current_time_idx - 8 # 8 * 15min = 2 hours
    window_end = current_time_idx
    
    forecast_recent = df.loc[window_start:window_end-1, 'power_kw']
    
    # Simulate that we are slightly underproducing (e.g. 15% less due to local shading/dust)
    actual_recent = forecast_recent * 0.85 
    
    print(f"Analyzing mismatch for the past 2 hours (Indices {window_start} to {window_end-1})...")
    correction_factor = calculate_correction_factor(actual_recent, forecast_recent)
    print(f"Calculated Multiplicative Correction Factor: {correction_factor:.3f}x")
    
    # Apply correction to the next 2 hours
    future_end = current_time_idx + 8
    df['corrected_power_kw'] = df['power_kw'].copy()
    
    # Apply fading correction (tapers off over time)
    for i in range(current_time_idx, future_end):
        # Fade the factor back to 1.0 linearly over the 2 hours
        steps_out = i - current_time_idx
        fade = steps_out / 8.0 
        current_factor = correction_factor * (1 - fade) + 1.0 * fade
        
        df.loc[i, 'corrected_power_kw'] *= current_factor

    print("\nCorrected 2-Hour Ahead Schedule:")
    print(df.loc[current_time_idx:future_end-1, ['date', 'power_kw', 'corrected_power_kw']])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df.loc[window_start:future_end+8, 'date'], 
             df.loc[window_start:future_end+8, 'power_kw'], 
             label='Original Day-Ahead', linestyle='--', color='gray')
             
    plt.plot(df.loc[window_start:window_end-1, 'date'], 
             actual_recent, 
             label='Actual (Measured)', color='red', linewidth=2)
             
    plt.plot(df.loc[current_time_idx:future_end-1, 'date'], 
             df.loc[current_time_idx:future_end-1, 'corrected_power_kw'], 
             label='Corrected Intraday Schedule', color='blue', linewidth=2)
             
    plt.axvline(x=df.loc[current_time_idx, 'date'], color='black', linestyle=':', label='Current Time')
    
    plt.title('Intraday Correction based on Recent Mismatch')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("intraday_correction_plot.png")
    print("\nSaved intraday correction plot to intraday_correction_plot.png")

if __name__ == "__main__":
    main()
